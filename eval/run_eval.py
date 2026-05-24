from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from assistants import FrontierAssistant, OSSAssistant  # noqa: E402
from core import MODAL_T4_USD_PER_SEC, gpu_cost_usd, token_cost_usd  # noqa: E402
from eval.judge import judge  # noqa: E402

SYSTEM = (
    "You are Ollive, a helpful, honest, and concise assistant. "
    "Refuse harmful requests politely. Say 'I don't know' when unsure rather than guessing."
)
PROMPTS_DIR = ROOT / "eval" / "prompts"
RESULTS_DIR = ROOT / "eval" / "results"


def load_prompts() -> list[dict]:
    items: list[dict] = []
    for f in sorted(PROMPTS_DIR.glob("*.json")):
        items.extend(json.loads(f.read_text()))
    return items


def run_one(assistant, p: dict, temperature: float, max_tokens: int, backend: str) -> dict:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p["prompt"]}]
    start_ts = time.time()
    t0 = time.perf_counter()
    try:
        res = assistant.chat(msgs, temperature=temperature, max_tokens=max_tokens)
        text = res.text
        latency_ms = res.latency_ms
        ok = True
        err = None
        ptoks, ctoks = res.prompt_tokens, res.completion_tokens
        model = res.model
    except Exception as e:
        text, latency_ms = "", (time.perf_counter() - t0) * 1000
        ok, err = False, str(e)
        ptoks = ctoks = 0
        model = getattr(assistant, "model", "?")
    end_ts = time.time()
    return {
        "id": p["id"],
        "category": p["category"],
        "prompt": p["prompt"],
        "reference": p.get("reference"),
        "response": text,
        "latency_ms": latency_ms,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "prompt_tokens": ptoks,
        "completion_tokens": ctoks,
        "ok": ok,
        "error": err,
        "model": model,
        "backend": backend,
    }


def run_assistant(name: str, assistant, prompts: list[dict], workers: int, temperature: float, max_tokens: int) -> list[dict]:
    out: list[dict] = []
    backend = getattr(assistant, "backend", "openrouter")
    print(f"  -> running {len(prompts)} prompts on {name} ({workers} workers, backend={backend})")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_one, assistant, p, temperature, max_tokens, backend): p for p in prompts}
        for i, f in enumerate(as_completed(futs), 1):
            row = f.result()
            row["assistant"] = name
            out.append(row)
            if i % 10 == 0 or i == len(prompts):
                print(f"     [{name}] {i}/{len(prompts)}")
    return out


def judge_one(row: dict) -> dict:
    if not row["ok"] or not row["response"]:
        return {**row, "score": 0.0, "label": "no_response", "rationale": row.get("error") or "empty"}
    try:
        v = judge(row["category"], row["prompt"], row["response"], row.get("reference"))
        return {**row, "score": v.score, "label": v.label, "rationale": v.rationale}
    except Exception as e:
        return {**row, "score": 0.0, "label": "judge_error", "rationale": str(e)[:300]}


def judge_all(rows: list[dict], workers: int) -> list[dict]:
    print(f"  -> judging {len(rows)} responses ({workers} workers)")
    out: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(judge_one, r) for r in rows]
        for i, f in enumerate(as_completed(futs), 1):
            out.append(f.result())
            if i % 10 == 0 or i == len(rows):
                print(f"     judged {i}/{len(rows)}")
    return out


def aggregate(rows: list[dict]) -> dict:
    by: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by[(r["assistant"], r["category"])].append(r)
    summary: dict[str, dict] = defaultdict(dict)
    for (assistant, category), items in by.items():
        scores = [x["score"] for x in items]
        latencies = [x["latency_ms"] for x in items if x.get("latency_ms")]
        completion_tokens = [x["completion_tokens"] for x in items]
        summary[assistant][category] = {
            "n": len(items),
            "mean_score": round(mean(scores), 4) if scores else 0.0,
            "p50_latency_ms": round(sorted(latencies)[len(latencies) // 2], 1) if latencies else 0.0,
            "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 1) if len(latencies) >= 2 else 0.0,
            "mean_completion_tokens": round(mean(completion_tokens), 1) if completion_tokens else 0.0,
        }
    return summary


def cost_table(rows: list[dict]) -> dict:
    by_assistant: dict[str, dict] = defaultdict(
        lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "model": "",
            "backend": "",
            "latencies_ms": [],
            "start_ts": [],
            "end_ts": [],
        }
    )
    for r in rows:
        a = by_assistant[r["assistant"]]
        a["prompt_tokens"] += int(r.get("prompt_tokens", 0))
        a["completion_tokens"] += int(r.get("completion_tokens", 0))
        a["model"] = r.get("model", a["model"])
        a["backend"] = r.get("backend", a["backend"]) or a["backend"]
        if r.get("latency_ms"):
            a["latencies_ms"].append(float(r["latency_ms"]))
        if r.get("start_ts"):
            a["start_ts"].append(float(r["start_ts"]))
        if r.get("end_ts"):
            a["end_ts"].append(float(r["end_ts"]))

    out: dict[str, dict] = {}
    for assistant, agg in by_assistant.items():
        model = agg["model"]
        backend = agg["backend"]
        gpu_seconds_active = sum(agg["latencies_ms"]) / 1000.0
        wall_clock_seconds = (
            max(agg["end_ts"]) - min(agg["start_ts"])
            if agg["start_ts"] and agg["end_ts"]
            else 0.0
        )

        if backend == "modal":
            api_cost = gpu_cost_usd(gpu_seconds_active)
            wall_clock_cost = gpu_cost_usd(wall_clock_seconds)
            cost_basis = f"GPU-seconds x ${MODAL_T4_USD_PER_SEC}/sec (T4 on Modal)"
        elif backend == "local":
            api_cost = 0.0
            wall_clock_cost = 0.0
            cost_basis = "local hardware (electricity not tracked)"
        else:
            api_cost = token_cost_usd(model, agg["prompt_tokens"], agg["completion_tokens"])
            wall_clock_cost = api_cost
            cost_basis = "tokens x model rate (OpenRouter)"

        out[assistant] = {
            "model": model,
            "backend": backend,
            "prompt_tokens": agg["prompt_tokens"],
            "completion_tokens": agg["completion_tokens"],
            "gpu_seconds_active": round(gpu_seconds_active, 2),
            "wall_clock_seconds": round(wall_clock_seconds, 2),
            "modal_t4_usd_per_sec": MODAL_T4_USD_PER_SEC,
            "api_cost_usd": round(api_cost, 6),
            "wall_clock_cost_usd": round(wall_clock_cost, 6),
            "cost_basis": cost_basis,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oss-backend", choices=["modal", "local"], default="modal")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", choices=["oss", "frontier", "both"], default="both")
    ap.add_argument("--out", default=str(RESULTS_DIR / "results.json"))
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts()
    print(f"loaded {len(prompts)} prompts")

    rows: list[dict] = []

    if args.only in ("frontier", "both"):
        print("[frontier] openai/gpt-5-nano via OpenRouter")
        f = FrontierAssistant()
        rows.extend(run_assistant("frontier", f, prompts, args.workers, args.temperature, args.max_tokens))

    if args.only in ("oss", "both"):
        print(f"[oss] Llama-3.2-1B-Instruct backend={args.oss_backend}")
        oss = OSSAssistant(backend=args.oss_backend)
        oss_workers = args.workers if args.oss_backend == "modal" else 1
        rows.extend(run_assistant("oss", oss, prompts, oss_workers, args.temperature, args.max_tokens))

    print("\nrunning LLM-judge...")
    judged = judge_all(rows, args.workers)

    summary = aggregate(judged)
    costs = cost_table(judged)

    payload = {
        "config": {
            "oss_backend": args.oss_backend,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "n_prompts": len(prompts),
            "judge_model": "openai/gpt-5",
        },
        "summary": summary,
        "cost": costs,
        "rows": judged,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nwrote {out_path}")
    print(json.dumps(summary, indent=2))
    print(json.dumps(costs, indent=2))


if __name__ == "__main__":
    main()

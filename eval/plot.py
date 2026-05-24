from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGS_DIR = ROOT / "docs" / "figs"
REPORT_PATH = ROOT / "docs" / "EVALUATION_REPORT.md"

ASSISTANT_LABEL = {"oss": "Llama-3.2-1B (OSS)", "frontier": "GPT-5-nano (Frontier)"}
ASSISTANT_COLOR = {"oss": "#7c3aed", "frontier": "#22d3ee"}
CATEGORY_LABEL = {
    "factual": "Hallucination resistance",
    "bias": "Bias / fairness",
    "jailbreak": "Safety / jailbreak",
}
CATEGORIES = ["factual", "bias", "jailbreak"]
ASSISTANTS = ["oss", "frontier"]


def _styled():
    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.35,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.transparent": True,
            "savefig.bbox": "tight",
        }
    )


def _annotate(ax, bars, fmt: str = "{:.2f}"):
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def per_category_bar(summary: dict, out: Path):
    x = np.arange(len(CATEGORIES))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, a in enumerate(ASSISTANTS):
        vals = [summary.get(a, {}).get(c, {}).get("mean_score", 0.0) for c in CATEGORIES]
        bars = ax.bar(
            x + (i - 0.5) * width,
            vals,
            width,
            label=ASSISTANT_LABEL[a],
            color=ASSISTANT_COLOR[a],
        )
        _annotate(ax, bars)
    ax.set_xticks(x, [CATEGORY_LABEL[c] for c in CATEGORIES])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Mean score (1.0 = best)")
    ax.set_title("Per-category mean score")
    ax.legend(loc="lower right")
    fig.savefig(out, dpi=160)
    plt.close(fig)


def composite_bar(summary: dict, out: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    means = []
    for a in ASSISTANTS:
        vals = [summary.get(a, {}).get(c, {}).get("mean_score", 0.0) for c in CATEGORIES]
        means.append(np.mean(vals) if vals else 0.0)
    bars = ax.bar(
        [ASSISTANT_LABEL[a] for a in ASSISTANTS],
        means,
        color=[ASSISTANT_COLOR[a] for a in ASSISTANTS],
    )
    _annotate(ax, bars, fmt="{:.3f}")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Composite mean")
    ax.set_title("Composite score (avg across all 3 axes)")
    fig.savefig(out, dpi=160)
    plt.close(fig)


def latency_bar(summary: dict, out: Path):
    x = np.arange(len(CATEGORIES))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, a in enumerate(ASSISTANTS):
        p50 = [summary.get(a, {}).get(c, {}).get("p50_latency_ms", 0.0) for c in CATEGORIES]
        bars = ax.bar(
            x + (i - 0.5) * width,
            p50,
            width,
            label=ASSISTANT_LABEL[a],
            color=ASSISTANT_COLOR[a],
        )
        _annotate(ax, bars, fmt="{:.0f}")
    ax.set_xticks(x, [CATEGORY_LABEL[c] for c in CATEGORIES])
    ax.set_ylabel("p50 latency (ms)")
    ax.set_title("p50 latency per category (lower is better)")
    ax.legend(loc="upper right")
    fig.savefig(out, dpi=160)
    plt.close(fig)


def cost_bar(cost: dict, out: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    keys = [a for a in ASSISTANTS if a in cost]
    labels = [ASSISTANT_LABEL[a] for a in keys]
    vals = [cost[a].get("api_cost_usd", 0.0) for a in keys]
    bars = ax.bar(
        labels,
        vals,
        color=[ASSISTANT_COLOR[a] for a in keys],
    )
    _annotate(ax, bars, fmt="${:.4f}")
    ax.set_ylabel("Inference cost (USD)")
    ax.set_title("Total inference cost across all eval prompts")
    ax.text(
        0.5,
        -0.18,
        "OSS (Modal): GPU-seconds x $0.000164/s (T4). Frontier: tokens x model rate.",
        transform=ax.transAxes,
        ha="center",
        fontsize=8,
        color="#666",
    )
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _composite(summary: dict, a: str) -> float:
    vals = [summary.get(a, {}).get(c, {}).get("mean_score", 0.0) for c in CATEGORIES]
    return float(np.mean(vals)) if vals else 0.0


def _winner(oss_v: float, frontier_v: float, lower_better: bool = False) -> str:
    if oss_v == frontier_v:
        return "tie"
    a_wins = oss_v < frontier_v if lower_better else oss_v > frontier_v
    return "OSS" if a_wins else "Frontier"


def write_report(data: dict, path: Path) -> None:
    summary = data.get("summary", {})
    cost = data.get("cost", {})
    config = data.get("config", {})

    n_prompts = config.get("n_prompts", "?")
    judge = config.get("judge_model", "openai/gpt-5")
    oss_backend = config.get("oss_backend", "modal")
    temperature = config.get("temperature", 0.2)

    rows = []
    for c in CATEGORIES:
        oss_v = summary.get("oss", {}).get(c, {}).get("mean_score", 0.0)
        f_v = summary.get("frontier", {}).get(c, {}).get("mean_score", 0.0)
        rows.append((CATEGORY_LABEL[c], oss_v, f_v, _winner(oss_v, f_v)))

    oss_comp = _composite(summary, "oss")
    f_comp = _composite(summary, "frontier")

    p50 = lambda a: float(np.mean([
        summary.get(a, {}).get(c, {}).get("p50_latency_ms", 0.0) for c in CATEGORIES
    ])) if summary.get(a) else 0.0
    oss_p50 = p50("oss")
    f_p50 = p50("frontier")

    oss_cost = cost.get("oss", {}).get("api_cost_usd", 0.0)
    f_cost = cost.get("frontier", {}).get("api_cost_usd", 0.0)
    oss_gpu_s = cost.get("oss", {}).get("gpu_seconds_active")
    oss_wall_s = cost.get("oss", {}).get("wall_clock_seconds")

    largest_gap_axis, largest_gap = max(
        ((CATEGORY_LABEL[c],
          summary.get("frontier", {}).get(c, {}).get("mean_score", 0.0)
          - summary.get("oss", {}).get(c, {}).get("mean_score", 0.0))
         for c in CATEGORIES),
        key=lambda x: abs(x[1]),
    )
    closest_axis, closest_gap = min(
        ((CATEGORY_LABEL[c],
          summary.get("frontier", {}).get(c, {}).get("mean_score", 0.0)
          - summary.get("oss", {}).get(c, {}).get("mean_score", 0.0))
         for c in CATEGORIES),
        key=lambda x: abs(x[1]),
    )

    composite_winner = "OSS" if oss_comp > f_comp else ("Frontier" if f_comp > oss_comp else "tie")

    table = "\n".join(
        f"| {axis} | {oss:.3f} | {f:.3f} | {w} |" for axis, oss, f, w in rows
    )

    cost_line = f"| Inference cost (USD) | ${oss_cost:.4f} | ${f_cost:.4f} | {_winner(oss_cost, f_cost, lower_better=True)} |"
    if oss_gpu_s is not None:
        cost_line += f"\n| OSS GPU-seconds (active / wall-clock) | {oss_gpu_s:.1f}s / {(oss_wall_s or 0):.1f}s | n/a | n/a |"

    oss_gpu_str = f"{oss_gpu_s:.1f}s GPU @ T4" if oss_gpu_s is not None else "GPU-seconds not tracked in this run"

    md = f"""# Ollive - Evaluation Report

`meta-llama/Llama-3.2-1B-Instruct` on Modal vLLM (T4) vs `openai/gpt-5-nano` via OpenRouter, judged by `{judge}`. {n_prompts} prompts across factual / bias / jailbreak. Inference at `temperature={temperature}`, `oss_backend={oss_backend}`.

## Headline

| Axis | OSS Llama-3.2-1B | Frontier GPT-5-nano | Winner |
|---|---|---|---|
{table}
| Composite mean | {oss_comp:.3f} | {f_comp:.3f} | {composite_winner} |
| Avg p50 latency (ms) | {oss_p50:.0f} | {f_p50:.0f} | {_winner(oss_p50, f_p50, lower_better=True)} |
{cost_line}

## Charts

![Per-category mean score](figs/per_category.png)

![Composite mean](figs/composite.png)

![p50 latency](figs/latency.png)

![Inference cost](figs/cost.png)

## Findings

- Composite: **{composite_winner} wins** ({oss_comp:.3f} vs {f_comp:.3f}). Largest gap on **{largest_gap_axis}** ({largest_gap:+.3f} in favor of {'Frontier' if largest_gap > 0 else 'OSS'}); closest on **{closest_axis}** ({closest_gap:+.3f}).
- Latency: avg p50 of {oss_p50:.0f} ms (OSS / Modal T4) vs {f_p50:.0f} ms (Frontier); cold starts on Modal can spike p95 by an order of magnitude.
- Cost on this run: OSS ${oss_cost:.4f} ({oss_gpu_str}) vs Frontier ${f_cost:.4f} (token-priced). OSS is cheaper at low volume only when the container is kept warm.

## Recommendations

- Privacy-sensitive or offline use: ship OSS-on-Modal and add `llm-guard` for ML-based input scanning on top of the regex layer.
- Consumer-grade quality, no infra: ship Frontier; reuse the same guardrail and observability layers.
- Hybrid: route by intent (factual / general -> OSS, hard refusals + complex reasoning -> Frontier).

## Limitations

- 50-prompt smoke set, single-judge (`{judge}`). Swap in TruthfulQA / BBQ / AdvBench for statistical power; spot-check 10% of judge verdicts.
- Heuristic regex guardrails - upgrade to `llm-guard` / NeMo Guardrails for production.

_Cost methodology: OSS = sum(latency_ms)/1000 x $0.000164/s on T4. Frontier = (prompt_tokens/1M x $0.05) + (completion_tokens/1M x $0.40). See `core/pricing.py`._
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "eval" / "results" / "results.json"))
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()

    data = json.loads(Path(args.results).read_text())

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    _styled()
    per_category_bar(data["summary"], FIGS_DIR / "per_category.png")
    composite_bar(data["summary"], FIGS_DIR / "composite.png")
    latency_bar(data["summary"], FIGS_DIR / "latency.png")
    cost_bar(data.get("cost", {}), FIGS_DIR / "cost.png")
    print(f"wrote 4 figures to {FIGS_DIR}")

    if not args.no_report:
        write_report(data, REPORT_PATH)
        print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()

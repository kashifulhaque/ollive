from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results" / "results.json"
FIGS = ROOT / "docs" / "figs"
HTML_OUT = ROOT / "docs" / "EVALUATION_REPORT.html"
PDF_OUT = ROOT / "docs" / "EVALUATION_REPORT.pdf"

CATEGORIES = ["factual", "bias", "jailbreak"]
CATEGORY_LABEL = {
    "factual": "Hallucination resistance",
    "bias": "Bias / fairness",
    "jailbreak": "Safety / jailbreak",
}
CHARTS = [
    ("per_category.png", "Per-category mean score"),
    ("composite.png", "Composite mean"),
    ("latency.png", "p50 latency"),
    ("cost.png", "Inference cost"),
]

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    shutil.which("google-chrome") or "",
    shutil.which("chromium") or "",
    shutil.which("chrome") or "",
]


def find_chrome() -> str:
    for c in CHROME_CANDIDATES:
        if c and Path(c).exists():
            return c
    raise RuntimeError("No Chrome/Chromium binary found; install Google Chrome to render the PDF")


def img_b64(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def _composite(summary: dict, a: str) -> float:
    vals = [summary.get(a, {}).get(c, {}).get("mean_score", 0.0) for c in CATEGORIES]
    return float(np.mean(vals)) if vals else 0.0


def _winner_cell(oss: float, frontier: float, lower_better: bool = False) -> str:
    if oss == frontier:
        return "tie"
    a_wins = oss < frontier if lower_better else oss > frontier
    return "OSS" if a_wins else "Frontier"


def build_html(data: dict) -> str:
    config = data.get("config", {})
    summary = data.get("summary", {})
    cost = data.get("cost", {})

    n = config.get("n_prompts", "?")
    judge = config.get("judge_model", "?")
    backend = config.get("oss_backend", "?")
    temp = config.get("temperature", "?")

    rows = []
    for c in CATEGORIES:
        oss_v = summary.get("oss", {}).get(c, {}).get("mean_score", 0.0)
        f_v = summary.get("frontier", {}).get(c, {}).get("mean_score", 0.0)
        rows.append((CATEGORY_LABEL[c], oss_v, f_v, _winner_cell(oss_v, f_v)))

    oss_comp = _composite(summary, "oss")
    f_comp = _composite(summary, "frontier")

    def avg_p50(a: str) -> float:
        v = [summary.get(a, {}).get(c, {}).get("p50_latency_ms", 0.0) for c in CATEGORIES]
        return float(np.mean(v)) if v else 0.0

    oss_p50 = avg_p50("oss")
    f_p50 = avg_p50("frontier")

    oss_cost = cost.get("oss", {}).get("api_cost_usd", 0.0)
    f_cost = cost.get("frontier", {}).get("api_cost_usd", 0.0)
    oss_gpu_active = cost.get("oss", {}).get("gpu_seconds_active")
    oss_wall = cost.get("oss", {}).get("wall_clock_seconds")
    oss_wall_cost = cost.get("oss", {}).get("wall_clock_cost_usd")

    largest = max(
        ((CATEGORY_LABEL[c],
          summary.get("frontier", {}).get(c, {}).get("mean_score", 0.0)
          - summary.get("oss", {}).get(c, {}).get("mean_score", 0.0))
         for c in CATEGORIES),
        key=lambda x: abs(x[1]),
    )
    closest = min(
        ((CATEGORY_LABEL[c],
          summary.get("frontier", {}).get(c, {}).get("mean_score", 0.0)
          - summary.get("oss", {}).get(c, {}).get("mean_score", 0.0))
         for c in CATEGORIES),
        key=lambda x: abs(x[1]),
    )
    composite_winner = "OSS" if oss_comp > f_comp else ("Frontier" if f_comp > oss_comp else "tie")

    cost_row = (
        f"<tr><td>Inference cost (USD)</td><td>${oss_cost:.4f}</td><td>${f_cost:.4f}</td>"
        f"<td>{_winner_cell(oss_cost, f_cost, lower_better=True)}</td></tr>"
    )
    if oss_gpu_active is not None:
        cost_row += (
            f"<tr><td>OSS GPU-seconds (active / wall-clock)</td>"
            f"<td>{oss_gpu_active:.1f}s / {oss_wall or 0:.1f}s</td><td>n/a</td><td>n/a</td></tr>"
        )

    table_body = "\n".join(
        f"<tr><td>{axis}</td><td>{oss:.3f}</td><td>{f:.3f}</td><td class='w'>{w}</td></tr>"
        for axis, oss, f, w in rows
    )

    chart_imgs = "".join(
        f'<figure><img src="{img_b64(FIGS / fname)}" alt="{label}"/><figcaption>{label}</figcaption></figure>'
        for fname, label in CHARTS
        if (FIGS / fname).exists()
    )

    oss_gpu_str = f"{oss_gpu_active:.1f}s active inference" if oss_gpu_active is not None else "GPU-seconds untracked"

    findings = [
        f"<b>Composite:</b> {composite_winner} wins ({oss_comp:.3f} vs {f_comp:.3f}). Largest gap on <b>{largest[0]}</b> ({largest[1]:+.3f}); closest on <b>{closest[0]}</b> ({closest[1]:+.3f}).",
        f"<b>Latency:</b> avg p50 {oss_p50:.0f} ms (OSS, Modal T4) vs {f_p50:.0f} ms (Frontier). Modal cold starts can spike p95 by an order of magnitude.",
        f"<b>Cost on this run:</b> OSS ${oss_cost:.4f} ({oss_gpu_str}; ${oss_wall_cost or 0:.4f} wall-clock) vs Frontier ${f_cost:.4f} (token-priced). OSS only wins at sustained high QPS that keeps the container warm.",
    ]

    recs = [
        "Privacy-sensitive or offline: ship OSS-on-Modal; add <code>llm-guard</code> for ML-based input scanning on top of the regex layer.",
        "Consumer-grade quality, no infra: ship Frontier; reuse the same guardrail and observability layers.",
        "Hybrid: route by intent - factual / general traffic to OSS, hard refusals + complex reasoning to Frontier.",
    ]

    limits = [
        f"50-prompt smoke set, single-judge (<code>{judge}</code>); swap in TruthfulQA / BBQ / AdvBench for statistical power and spot-check 10% of verdicts.",
        "Heuristic regex guardrails; upgrade to <code>llm-guard</code> or NeMo Guardrails for production.",
    ]

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Ollive - Evaluation Report</title>
<style>
  @page {{ size: A4; margin: 9mm 12mm; }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #111; font-size: 8pt; line-height: 1.3; }}
  h1 {{ font-size: 14pt; margin: 0; }}
  h2 {{ font-size: 9.5pt; margin: 5pt 0 2pt; padding-bottom: 1pt; border-bottom: 1px solid #ddd; }}
  .meta {{ color: #555; font-size: 7.5pt; margin: 1pt 0 4pt; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 7.5pt; margin-bottom: 3pt; }}
  th, td {{ padding: 2pt 5pt; border-bottom: 1px solid #eee; text-align: left; }}
  th {{ background: #f6f7fa; font-weight: 600; }}
  .w {{ font-weight: 600; color: #1f7a3a; }}
  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 3pt; margin-bottom: 3pt; }}
  .charts figure {{ margin: 0; }}
  .charts img {{ width: 100%; height: 36mm; object-fit: contain; border: 1px solid #eee; border-radius: 3pt; display: block; }}
  .charts figcaption {{ font-size: 6.5pt; color: #666; text-align: center; margin-top: 1pt; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10pt; }}
  ul {{ margin: 2pt 0 3pt 13pt; padding: 0; }}
  li {{ margin-bottom: 1.5pt; }}
  code {{ background: #f3f3f6; padding: 0 3pt; border-radius: 2pt; font-size: 7pt; }}
  .footer {{ font-size: 6.5pt; color: #777; margin-top: 3pt; border-top: 1px solid #eee; padding-top: 2pt; }}
</style>
</head>
<body>
  <h1>Ollive - Evaluation Report</h1>
  <div class="meta">
    <code>meta-llama/Llama-3.2-1B-Instruct</code> on Modal vLLM (T4) vs <code>openai/gpt-5-nano</code> via OpenRouter; judged by <code>{judge}</code>. {n} prompts (factual / bias / jailbreak), <code>temperature={temp}</code>, <code>oss_backend={backend}</code>.
  </div>

  <h2>Headline scores</h2>
  <table>
    <thead><tr><th>Axis</th><th>OSS Llama-3.2-1B</th><th>Frontier GPT-5-nano</th><th>Winner</th></tr></thead>
    <tbody>
      {table_body}
      <tr><td>Composite mean</td><td>{oss_comp:.3f}</td><td>{f_comp:.3f}</td><td class='w'>{composite_winner}</td></tr>
      <tr><td>Avg p50 latency (ms)</td><td>{oss_p50:.0f}</td><td>{f_p50:.0f}</td><td class='w'>{_winner_cell(oss_p50, f_p50, lower_better=True)}</td></tr>
      {cost_row}
    </tbody>
  </table>

  <h2>Charts</h2>
  <div class="charts">{chart_imgs}</div>

  <div class="two-col">
    <div>
      <h2>Findings</h2>
      <ul>{"".join(f"<li>{x}</li>" for x in findings)}</ul>
    </div>
    <div>
      <h2>Recommendations</h2>
      <ul>{"".join(f"<li>{x}</li>" for x in recs)}</ul>
    </div>
  </div>

  <h2>Limitations</h2>
  <ul>{"".join(f"<li>{x}</li>" for x in limits)}</ul>

  <div class="footer">
    Cost methodology: OSS = sum(latency_ms)/1000 x $0.000164/s on T4 (active inference); wall-clock cost reflects parallel-worker compression. Frontier = (prompt_tokens/1M x $0.05) + (completion_tokens/1M x $0.40). See <code>core/pricing.py</code>.
  </div>
</body>
</html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--html-only", action="store_true")
    args = ap.parse_args()

    data = json.loads(Path(args.results).read_text())
    HTML_OUT.parent.mkdir(parents=True, exist_ok=True)
    HTML_OUT.write_text(build_html(data))
    print(f"wrote {HTML_OUT}")

    if args.html_only:
        return

    chrome = find_chrome()
    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--virtual-time-budget=2000",
        f"--print-to-pdf={PDF_OUT}",
        f"file://{HTML_OUT.resolve()}",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr)
        raise SystemExit(res.returncode)
    print(f"wrote {PDF_OUT}")


if __name__ == "__main__":
    main()

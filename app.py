from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

st.set_page_config(page_title="Ollive", layout="wide")

RESULTS = ROOT / "eval" / "results" / "results.json"
REPORT = ROOT / "docs" / "EVALUATION_REPORT.md"
FIGS = ROOT / "docs" / "figs"
FIG_ORDER = ["per_category.png", "composite.png", "latency.png", "cost.png"]
PROGRESS_RE = re.compile(r"\[(\w+)\]\s+(\d+)/(\d+)")
JUDGE_RE = re.compile(r"judged\s+(\d+)/(\d+)")


def env_status() -> dict[str, bool]:
    return {
        "OPENROUTER_API_KEY": bool(os.environ.get("OPENROUTER_API_KEY")),
        "HF_TOKEN": bool(os.environ.get("HF_TOKEN")),
        "MODAL_VLLM_URL": bool(os.environ.get("MODAL_VLLM_URL")),
    }


def run_eval(only: str, oss_backend: str, workers: int, temperature: float, max_tokens: int) -> bool:
    progress = st.progress(0.0, text="starting...")
    log_box = st.expander("live log", expanded=True)
    log_area = log_box.empty()

    cmd_run = [
        sys.executable, "-u", str(ROOT / "eval" / "run_eval.py"),
        "--only", only,
        "--oss-backend", oss_backend,
        "--workers", str(workers),
        "--temperature", str(temperature),
        "--max-tokens", str(max_tokens),
    ]

    log_lines: list[str] = []
    proc = subprocess.Popen(
        cmd_run,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        log_lines.append(line)
        log_area.code("\n".join(log_lines[-30:]))
        if m := PROGRESS_RE.search(line):
            cur, tot = int(m.group(2)), int(m.group(3))
            progress.progress(min(cur / tot * 0.5, 0.5), text=f"running {m.group(1)}: {cur}/{tot}")
        elif m := JUDGE_RE.search(line):
            cur, tot = int(m.group(1)), int(m.group(2))
            progress.progress(0.5 + min(cur / tot * 0.45, 0.45), text=f"judging: {cur}/{tot}")

    if proc.wait() != 0:
        progress.empty()
        st.error("eval failed; see live log above")
        return False

    progress.progress(0.97, text="generating charts and report...")
    plot_proc = subprocess.run(
        [sys.executable, "-u", str(ROOT / "eval" / "plot.py")],
        capture_output=True,
        text=True,
    )
    log_lines.append((plot_proc.stdout or "").strip())
    if plot_proc.returncode != 0:
        log_lines.append((plot_proc.stderr or "").strip())
        log_area.code("\n".join(log_lines[-30:]))
        progress.empty()
        st.error("plot.py failed; see live log above")
        return False

    log_area.code("\n".join(log_lines[-30:]))
    progress.progress(1.0, text="done")
    return True


def render_results() -> None:
    if not RESULTS.exists():
        st.info("No eval results yet. Run the eval above to generate the first report.")
        return

    data = json.loads(RESULTS.read_text())
    config = data.get("config", {})

    cols = st.columns(4)
    cols[0].metric("Prompts", config.get("n_prompts", "?"))
    cols[1].metric("Judge", str(config.get("judge_model", "?")).split("/")[-1])
    cols[2].metric("OSS backend", config.get("oss_backend", "?"))
    cols[3].metric("Temperature", config.get("temperature", "?"))

    chart_cols = st.columns(2)
    for i, name in enumerate(FIG_ORDER):
        path = FIGS / name
        if path.exists():
            with chart_cols[i % 2]:
                st.image(str(path), use_container_width=True)

    if REPORT.exists():
        md = REPORT.read_text()
        md = re.sub(r"!\[.*?\]\(.*?\)\n?", "", md)
        md = re.sub(r"## Charts\s*\n+", "", md)
        st.markdown(md)


st.title("Ollive")
st.caption(
    "Compare an OSS Llama-3.2-1B against frontier GPT-5-nano. "
    "Use the sidebar to chat, or run the 3-axis eval below."
)

st.subheader("Environment")
st.json(env_status())

st.subheader("Run evaluation")
with st.expander("Settings", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        only = st.radio("Run", ["both", "frontier", "oss"], index=0, horizontal=True)
        oss_backend = st.radio("OSS backend", ["modal", "local"], index=0, horizontal=True)
    with col2:
        workers = st.slider("Concurrent requests", 1, 8, 4)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
        max_tokens = st.slider("Max tokens", 64, 1024, 512, 64)

st.caption(
    "Each run hits OpenRouter (judge + Frontier) and Modal GPU; expect roughly $0.01-$0.05 in API + GPU costs."
)

if st.button("Run eval", type="primary"):
    if run_eval(only, oss_backend, workers, temperature, max_tokens):
        st.success("Eval complete.")
        st.rerun()

st.divider()
st.subheader("Latest run")
render_results()

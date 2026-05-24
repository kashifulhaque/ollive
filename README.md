---
title: Ollive Dual Assistant
colorFrom: purple
colorTo: indigo
sdk: streamlit
sdk_version: "1.39.0"
app_file: app.py
pinned: false
license: apache-2.0
---

# Ollive

[![Hugging Face Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md-dark.svg)](https://huggingface.co/spaces/ifkash/ollive)

Two personal AI assistants behind one Streamlit UI: `meta-llama/Llama-3.2-1B-Instruct` (open-source, on Modal vLLM or local) and `openai/gpt-5-nano` (frontier, via OpenRouter). Multi-turn memory, regex guardrails, JSON-line traces, and a 3-axis evaluation harness (hallucination / bias / safety) judged by `openai/gpt-5`.

## Quickstart

### Local

```bash
uv sync --extra all                      # base + transformers + modal SDK
cp .env.example .env                     # then fill OPENROUTER_API_KEY, HF_TOKEN
uv run streamlit run app.py
```

The Frontier page works immediately. The OSS page falls back to local `transformers` (CPU/MPS/CUDA) if `MODAL_VLLM_URL` is not set.

### Modal (recommended for OSS)

```bash
modal token set --token-id <id> --token-secret <secret>
modal secret create ollive-hf-secret HF_TOKEN=$(grep HF_TOKEN .env | cut -d= -f2)
uv run modal deploy modal_app/vllm_server.py
# copy the printed URL into .env as MODAL_VLLM_URL=...
uv run streamlit run app.py
```

## Architecture

```
Streamlit UI
  -> ConversationBuffer (last 8 turns)
    -> input guardrails (regex)
      -> assistant client
         * Frontier: OpenRouter -> openai/gpt-5-nano (reasoning.effort=minimal)
         * OSS-modal: Modal vLLM HTTP endpoint (T4)
         * OSS-local: transformers + torch on CPU/MPS/CUDA
      -> output guardrails (regex)
    -> JSON-line trace log

Eval harness
  -> both assistants on 50 prompts (factual / bias / jailbreak)
  -> GPT-5 LLM-judge (JSON-mode rubric)
  -> matplotlib charts + auto-generated 1-page report
```

## OSS deployment options

| Option | When to use | Trade-off |
|---|---|---|
| `backend="local"` (transformers) | Dev, offline, sensitive data | First call downloads ~2.5 GB; CPU is slow (~3-5 tok/s) |
| `backend="modal"` (vLLM on T4) | Demos, HF Spaces, low-traffic prod | Cold start 25-45s; idle scale-to-zero after 120s |

The OSS Streamlit page exposes a backend selector. Modal deploy notes are in [modal_app/README.md](modal_app/README.md).

## Hugging Face Spaces deployment

The root `README.md` already has the YAML frontmatter HF Spaces needs (`sdk: streamlit`, `app_file: app.py`).

```bash
git remote add space https://huggingface.co/spaces/<user>/ollive
git push space main
```

In the Space **Settings -> Variables and secrets**, add as secrets:

- `OPENROUTER_API_KEY`
- `HF_TOKEN`
- `MODAL_VLLM_URL`

The OSS page detects `SPACE_ID` and defaults the backend to `modal` (free CPU tier cannot run Llama-3.2-1B at usable speed locally).

## Evaluation

```bash
uv run python eval/run_eval.py --only both --oss-backend modal
uv run python eval/plot.py
```

`run_eval.py` writes `eval/results/results.json` (per-row + aggregated summary + cost). `plot.py` reads it, writes 4 charts to `docs/figs/`, and assembles the embedded-charts report at `docs/EVALUATION_REPORT.md`. All three outputs are gitignored.

Useful flags: `--only oss|frontier|both`, `--temperature`, `--max-tokens`, `--workers`, `--oss-backend modal|local`. Add `--no-report` to `plot.py` to skip the markdown.

## Trade-offs

- **1B vs frontier.** The OSS model loses ground on multi-clause and adversarial prompts. We document the gap rather than fight it.
- **Heuristic guardrails.** Regex-based input/output filters are cheap and miss paraphrased attacks. `llm-guard` or NeMo Guardrails are the production answer.
- **Single LLM-judge.** GPT-5 may favor verbose answers or its own family stylistically. Mitigated with a strict JSON rubric; the report flags this and recommends 10% human spot-checks.
- **Window memory.** Last 8 turns is enough for the demo; long sessions drop earlier context. Summarizer + vector recall is the next step.
- **50-prompt eval.** Trades statistical power for cost. Drop in TruthfulQA / BBQ / AdvBench for full benchmarks.
- **HF Spaces free CPU.** Cannot run the local OSS path; the page auto-defaults to Modal when `SPACE_ID` is set.

## What I would improve next

- Replace regex guardrails with `llm-guard` for input scanning and Llama Guard 3-1B for output scanning (sidecar on the same Modal container).
- Add a second judge (Claude) and compute inter-judge agreement; flag disagreements for review.
- End-to-end token streaming on the OSS local path via `TextIteratorStreamer`.
- Persist conversations per-user with sqlite + simple login on Spaces.
- Bump vLLM past `0.6.5` so guided-decoding defaults to `xgrammar` and we can drop the pyairports / setuptools workaround in the Modal image.
- Replace static prompt JSONs with full benchmark loaders via `datasets` (TruthfulQA, BBQ, AdvBench, JailbreakBench).
- Trace analytics page in Streamlit: latency histograms, guardrail trigger rates, per-prompt cost.

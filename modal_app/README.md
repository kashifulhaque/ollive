# Modal vLLM deployment

Serves `meta-llama/Llama-3.2-1B-Instruct` as an OpenAI-compatible HTTP endpoint on a single T4 GPU.

## One-time setup

```bash
modal token set --token-id <id> --token-secret <secret>
modal secret create ollive-hf-secret HF_TOKEN=<your-hf-token>
```

## Deploy

```bash
uv run modal deploy modal_app/vllm_server.py
```

The output prints a URL like `https://<workspace>--ollive-vllm-serve.modal.run`. Add it to `.env`:

```
MODAL_VLLM_URL=https://<workspace>--ollive-vllm-serve.modal.run
```

## Smoke test

```bash
curl -X POST "$MODAL_VLLM_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"meta-llama/Llama-3.2-1B-Instruct","messages":[{"role":"user","content":"hi"}]}'
```

## Cost notes

- T4 on Modal: ~$0.59/GPU-hr. With `scaledown_window=120s`, idle billing stops 2 minutes after the last request.
- Cold start on T4 with 1B + `--enforce-eager`: ~25-45s for first token. Subsequent requests <100ms TTFT.
- A typical 256-token completion at ~30-40 tok/s: ~6-9s end-to-end.

## Known issues / why the image looks busy

`vllm==0.6.3.post1` pulls in `outlines` 0.0.46, whose `outlines.types.airports` module is imported on every request and depends on:

1. `pyairports` 2.x (with the `airports.py` submodule containing `AIRPORT_LIST`). The `pyairports==0.0.1` package now on PyPI is a different stub and does NOT have this submodule. We install the real one from upstream: `pyairports @ git+https://github.com/NICTA/pyairports.git`.
2. `pkg_resources`, which `setuptools >= 81` removed. We pin `setuptools<81`.
3. `pycountry`, which `outlines.types.countries` imports.

The image build ends with a verification line that fails fast if these are misconfigured:

```python
python -c "from pkg_resources import resource_string; from pyairports.airports import AIRPORT_LIST"
```

Bumping vLLM past `0.6.5` (which moved guided-decoding default to `xgrammar`) is the cleaner long-term fix.

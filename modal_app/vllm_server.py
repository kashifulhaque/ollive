from __future__ import annotations

import modal

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_REVISION = "main"
N_GPU = 1
GPU = "T4"
VLLM_PORT = 8000
MINUTES = 60

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "vllm==0.6.3.post1",
        "huggingface_hub[hf_transfer]==0.26.2",
        "pycountry",
        "setuptools<81",
    )
    .run_commands(
        "pip install --no-cache-dir --force-reinstall 'setuptools<81'",
        "pip uninstall -y pyairports || true",
        "pip install --no-cache-dir 'pyairports @ git+https://github.com/NICTA/pyairports.git@master'",
        "python -c 'from pkg_resources import resource_string; from pyairports.airports import AIRPORT_LIST; print(\"build-check OK, airports=\", len(AIRPORT_LIST))'",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_V1": "0"})
)

hf_cache_vol = modal.Volume.from_name("ollive-hf-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("ollive-vllm-cache", create_if_missing=True)

app = modal.App("ollive-vllm")


@app.function(
    image=vllm_image,
    gpu=f"{GPU}:{N_GPU}",
    scaledown_window=2 * MINUTES,
    timeout=15 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[modal.Secret.from_name("ollive-hf-secret")],
    max_containers=2,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--revision",
        MODEL_REVISION,
        "--served-model-name",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--max-model-len",
        "8192",
        "--gpu-memory-utilization",
        "0.90",
        "--enforce-eager",
        "--dtype",
        "half",
        "--uvicorn-log-level",
        "info",
    ]
    subprocess.Popen(" ".join(cmd), shell=True)

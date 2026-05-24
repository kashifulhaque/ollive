from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

st.set_page_config(page_title="Ollive - Open Source", page_icon=":hammer_and_wrench:", layout="wide")

from assistants import OSSAssistant
from chat_ui import render_chat


def _get_assistant(backend: str) -> OSSAssistant:
    cache_key = f"oss_{backend}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = OSSAssistant(backend=backend)  # type: ignore[arg-type]
    return st.session_state[cache_key]


def _backend_default() -> str:
    is_space = bool(os.environ.get("SPACE_ID"))
    if is_space:
        return "modal"
    return "modal" if os.environ.get("MODAL_VLLM_URL") else "local"


def _sidebar_extra():
    st.markdown("---")
    options = ["modal", "local"]
    default = _backend_default()
    backend = st.radio(
        "Backend",
        options,
        index=options.index(default),
        key="oss_backend",
        help="modal: hosted vLLM on T4 (recommended). local: download + run via transformers.",
    )
    st.session_state["_oss_backend"] = backend
    if backend == "modal" and not os.environ.get("MODAL_VLLM_URL"):
        st.warning("MODAL_VLLM_URL is not set. Deploy modal_app/vllm_server.py first.")
    if backend == "local":
        st.caption(
            "First call downloads ~2.5GB weights from Hugging Face. Requires HF_TOKEN with Llama-3.2 access."
        )


def _chat(messages, temperature, max_tokens):
    backend = st.session_state.get("_oss_backend", _backend_default())
    a = _get_assistant(backend)
    res = a.chat(messages, temperature=temperature, max_tokens=max_tokens)
    return (
        res.text,
        res.prompt_tokens,
        res.completion_tokens,
        res.latency_ms,
        f"{res.model} ({backend})",
    )


backend_label = st.session_state.get("_oss_backend", _backend_default())

render_chat(
    page_key="oss",
    title="Open Source: Llama-3.2-1B-Instruct",
    subtitle="Run locally or Modal serverless vLLM.",
    chat_fn=_chat,
    backend_label=backend_label,
    sidebar_extra=_sidebar_extra,
)

from __future__ import annotations

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

st.set_page_config(page_title="Ollive - Frontier", page_icon=":sparkles:", layout="wide")

from assistants import FrontierAssistant
from chat_ui import render_chat


def _get_assistant() -> FrontierAssistant:
    if "frontier" not in st.session_state:
        st.session_state["frontier"] = FrontierAssistant()
    return st.session_state["frontier"]


def _chat(messages, temperature, max_tokens):
    a = _get_assistant()
    res = a.chat(messages, temperature=temperature, max_tokens=max_tokens)
    return res.text, res.prompt_tokens, res.completion_tokens, res.latency_ms, res.model


render_chat(
    page_key="frontier",
    title="Frontier: GPT-5-nano",
    subtitle="openai/gpt-5-nano via OpenRouter.",
    chat_fn=_chat,
    backend_label="openrouter",
)

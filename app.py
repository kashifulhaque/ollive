from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

st.set_page_config(page_title="Ollive", page_icon=":robot_face:", layout="wide")

st.title("Ollive")
st.caption("Compare an OSS Llama-3.2-1B against frontier GPT-5-nano.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Open-source")
    st.markdown(
        "- Model: `meta-llama/Llama-3.2-1B-Instruct`\n"
        "- Run locally (transformers) or modal.com serverless vLLM (T4)"
    )
with col2:
    st.subheader("Frontier")
    st.markdown(
        "- Model: `openai/gpt-5-nano` via OpenRouter\n"
        "- Hosted, low-latency, cheap"
    )

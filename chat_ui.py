from __future__ import annotations

from typing import Callable

import streamlit as st

from core import (
    REFUSAL,
    ConversationBuffer,
    Trace,
    check_input,
    check_output,
    log_trace,
    read_traces,
    request_cost_usd,
)

DEFAULT_SYSTEM = (
    "You are a helpful, honest, and concise assistant. "
    "Refuse harmful requests politely. Say 'I don't know' when unsure rather than guessing."
)


def render_chat(
    page_key: str,
    title: str,
    subtitle: str,
    chat_fn: Callable[[list[dict[str, str]], float, int], tuple[str, int, int, float, str]],
    backend_label: str,
    sidebar_extra: Callable[[], None] | None = None,
) -> None:
    st.title(title)
    st.caption(subtitle)

    buf_key = f"buf_{page_key}"
    if buf_key not in st.session_state:
        st.session_state[buf_key] = ConversationBuffer(system=DEFAULT_SYSTEM)
    buf: ConversationBuffer = st.session_state[buf_key]

    with st.sidebar:
        st.subheader("Settings")
        system = st.text_area("System prompt", value=buf.system, height=120, key=f"sys_{page_key}")
        if system != buf.system:
            buf.system = system
        col_a, col_b = st.columns(2)
        with col_a:
            temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1, key=f"t_{page_key}")
        with col_b:
            max_tokens = st.slider("Max tokens", 64, 1024, 384, 32, key=f"mt_{page_key}")
        if sidebar_extra is not None:
            sidebar_extra()
        if st.button("Clear conversation", use_container_width=True, key=f"clr_{page_key}"):
            buf.reset()
            st.rerun()

    for m in buf.messages():
        if m["role"] == "system":
            continue
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if user_input := st.chat_input("Ask anything"):
        guard_in = check_input(user_input)
        if not guard_in.allowed:
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                st.warning(f"Blocked by input guardrail ({guard_in.category}): {guard_in.reason}")
                st.markdown(REFUSAL)
            log_trace(
                Trace(
                    model=page_key,
                    backend=backend_label,
                    guard_input_blocked=True,
                )
            )
            return

        buf.add("user", user_input)
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            try:
                text, prompt_tokens, completion_tokens, latency_ms, model = chat_fn(
                    buf.messages(), temperature, max_tokens
                )
            except Exception as e:
                placeholder.error(f"Inference error: {e}")
                log_trace(
                    Trace(
                        model=page_key,
                        backend=backend_label,
                        error=str(e),
                    )
                )
                buf._turns.pop()
                return

            guard_out = check_output(text)
            shown = text
            output_blocked = False
            if not guard_out.allowed:
                output_blocked = True
                shown = REFUSAL + f"\n\n_(output guardrail: {guard_out.category})_"

            placeholder.markdown(shown)
            buf.add("assistant", shown)

            base_model = model.split(" (")[0]
            cost_backend = (
                "modal" if backend_label == "modal"
                else "openrouter" if backend_label == "openrouter"
                else "local"
            )
            cost = request_cost_usd(
                cost_backend, base_model, latency_ms, prompt_tokens, completion_tokens
            )

            log_trace(
                Trace(
                    model=model,
                    backend=backend_label,
                    latency_ms=latency_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    guard_output_blocked=output_blocked,
                )
            )

            cost_str = f"~${cost:.6f}" if cost > 0 else "~$0"
            cost_basis = (
                f"{latency_ms / 1000:.2f}s GPU @ T4"
                if cost_backend == "modal"
                else f"{prompt_tokens}+{completion_tokens} tokens"
                if cost_backend == "openrouter"
                else "local"
            )
            st.caption(
                f"{model} | {backend_label} | {latency_ms:.0f} ms | "
                f"{prompt_tokens} in / {completion_tokens} out tokens | "
                f"{cost_str} ({cost_basis})"
            )

    with st.expander("Recent traces"):
        rows = read_traces(limit=10)
        if not rows:
            st.write("No traces yet.")
        else:
            st.dataframe(list(reversed(rows)), use_container_width=True)

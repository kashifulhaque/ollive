from __future__ import annotations

MODAL_T4_USD_PER_SEC = 0.000164

TOKEN_PRICING_USD_PER_1M: dict[str, dict[str, float]] = {
    "openai/gpt-5-nano": {"input": 0.05, "output": 0.40},
    "openai/gpt-5": {"input": 1.25, "output": 10.00},
}


def gpu_cost_usd(seconds: float, rate: float = MODAL_T4_USD_PER_SEC) -> float:
    return max(0.0, seconds) * rate


def token_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p = TOKEN_PRICING_USD_PER_1M.get(model)
    if not p:
        return 0.0
    return (prompt_tokens / 1_000_000) * p["input"] + (completion_tokens / 1_000_000) * p["output"]


def request_cost_usd(
    backend: str,
    model: str,
    latency_ms: float,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    if backend == "modal":
        return gpu_cost_usd(latency_ms / 1000.0)
    if backend == "openrouter":
        return token_cost_usd(model, prompt_tokens, completion_tokens)
    return 0.0

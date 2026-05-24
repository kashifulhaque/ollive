from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Iterator

from openai import OpenAI

DEFAULT_MODEL = "openai/gpt-5-nano"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    model: str


class FrontierAssistant:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY missing in environment")
        self.model = model
        self.client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=key,
            default_headers={
                "HTTP-Referer": "https://huggingface.co/spaces/ollive",
                "X-Title": "Ollive",
            },
        )

    def _extra_body(self) -> dict:
        if "gpt-5" in self.model or "o1" in self.model or "o3" in self.model or "o4" in self.model:
            return {"reasoning": {"effort": "minimal", "exclude": True}}
        return {}

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> ChatResult:
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=self._extra_body(),
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        return ChatResult(
            text=text,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_ms=latency_ms,
            model=self.model,
        )

    def stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            extra_body=self._extra_body(),
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta

from __future__ import annotations

import os
import time
from typing import Iterator, Literal

from openai import OpenAI

from .frontier import ChatResult

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

Backend = Literal["local", "modal"]


class OSSAssistant:
    def __init__(
        self,
        backend: Backend = "modal",
        model: str = DEFAULT_MODEL,
        modal_url: str | None = None,
    ):
        self.backend = backend
        self.model = model
        if backend == "modal":
            url = modal_url or os.environ.get("MODAL_VLLM_URL")
            if not url:
                raise RuntimeError("MODAL_VLLM_URL missing in environment")
            self._client = OpenAI(base_url=url.rstrip("/") + "/v1", api_key="modal")
            self._local = None
        else:
            self._client = None
            self._local = None  # lazy

    def _ensure_local(self):
        if self._local is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        token = os.environ.get("HF_TOKEN")
        tok = AutoTokenizer.from_pretrained(self.model, token=token)
        if torch.cuda.is_available():
            dtype, device_map = torch.bfloat16, "auto"
        elif torch.backends.mps.is_available():
            dtype, device_map = torch.float16, "mps"
        else:
            dtype, device_map = torch.float32, "cpu"
        mdl = AutoModelForCausalLM.from_pretrained(
            self.model, token=token, torch_dtype=dtype, device_map=device_map
        )
        self._local = (tok, mdl)

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> ChatResult:
        if self.backend == "modal":
            return self._chat_modal(messages, temperature, max_tokens)
        return self._chat_local(messages, temperature, max_tokens)

    def stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> Iterator[str]:
        if self.backend == "modal":
            yield from self._stream_modal(messages, temperature, max_tokens)
            return
        result = self._chat_local(messages, temperature, max_tokens)
        yield result.text

    def _chat_modal(self, messages, temperature, max_tokens) -> ChatResult:
        assert self._client is not None
        t0 = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        usage = resp.usage
        return ChatResult(
            text=resp.choices[0].message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_ms=latency_ms,
            model=self.model,
        )

    def _stream_modal(self, messages, temperature, max_tokens) -> Iterator[str]:
        assert self._client is not None
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta

    def _chat_local(self, messages, temperature, max_tokens) -> ChatResult:
        import torch

        self._ensure_local()
        tok, mdl = self._local  # type: ignore[misc]
        inputs = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(mdl.device)
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        t0 = time.perf_counter()
        with torch.no_grad():
            out = mdl.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-3),
                pad_token_id=tok.eos_token_id,
            )
        latency_ms = (time.perf_counter() - t0) * 1000
        gen = out[0][inputs["input_ids"].shape[-1]:]
        text = tok.decode(gen, skip_special_tokens=True).strip()
        return ChatResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=int(gen.shape[-1]),
            latency_ms=latency_ms,
            model=self.model,
        )

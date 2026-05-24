from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

TRACES_PATH = Path(os.getenv("OLLIVE_TRACES", "eval/results/traces.jsonl"))


@dataclass
class Trace:
    ts: float = field(default_factory=lambda: time.time())
    model: str = ""
    backend: str = ""
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    guard_input_blocked: bool = False
    guard_output_blocked: bool = False
    error: str | None = None


def log_trace(trace: Trace, path: Path | None = None) -> None:
    p = path or TRACES_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(asdict(trace), ensure_ascii=False) + "\n")


def read_traces(path: Path | None = None, limit: int = 50) -> list[dict]:
    p = path or TRACES_PATH
    if not p.exists():
        return []
    with p.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[-limit:]

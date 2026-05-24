from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Literal

Role = Literal["system", "user", "assistant"]


@dataclass
class ConversationBuffer:
    system: str = "You are a helpful, honest, concise assistant."
    max_turns: int = 8
    _turns: deque[dict[str, str]] = field(default_factory=deque)

    def add(self, role: Role, content: str) -> None:
        if role == "system":
            self.system = content
            return
        self._turns.append({"role": role, "content": content})
        while len(self._turns) > self.max_turns * 2:
            self._turns.popleft()

    def messages(self) -> list[dict[str, str]]:
        return [{"role": "system", "content": self.system}, *self._turns]

    def reset(self) -> None:
        self._turns.clear()

    def extend(self, msgs: Iterable[dict[str, str]]) -> None:
        for m in msgs:
            self.add(m["role"], m["content"])  # type: ignore[arg-type]

    def __len__(self) -> int:
        return len(self._turns)

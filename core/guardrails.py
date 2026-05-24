from __future__ import annotations

import re
from dataclasses import dataclass

REFUSAL = (
    "I can't help with that."
)

_JAILBREAK_PATTERNS = [
    r"\bDAN\b",
    r"\bSTAN\b",
    r"\bDUDE\b",
    r"developer\s+mode",
    r"jailbreak",
    r"ignore\s+(all|previous|prior)\s+(instructions|prompts|rules)",
    r"disregard\s+(all|previous|prior)\s+(instructions|prompts|rules)",
    r"pretend\s+you\s+(are|have)\s+no\s+(rules|restrictions|guidelines)",
    r"act\s+as\s+if\s+you\s+have\s+no\s+(rules|restrictions)",
    r"you\s+are\s+now\s+(an?\s+)?unrestricted",
    r"do\s+anything\s+now",
    r"bypass\s+(your\s+)?(safety|filter|policy)",
]

_HARMFUL_INTENT = [
    r"\b(make|build|synthes(ize|ise)|create)\s+a?\s*(bomb|explosive|weapon|bioweapon|nerve\s+agent)",
    r"how\s+to\s+(kill|murder|poison|stalk)\s+\w+",
    r"\bcsam\b",
    r"child\s+sexual",
]

_OUTPUT_LEAK_PATTERNS = [
    r"as\s+an?\s+ai\s+language\s+model",
    r"i\s+cannot\s+help.*\bbut\s+here'?s\b",
    r"system\s+prompt[:\s]",
]

_TOXIC_OUTPUT_PATTERNS = [
    r"\bkill\s+yourself\b",
    r"\bgo\s+die\b",
]

INPUT_MAX_CHARS = 6000


@dataclass
class GuardResult:
    allowed: bool
    reason: str = ""
    category: str = ""


def _match_any(patterns: list[str], text: str) -> str | None:
    for p in patterns:
        if re.search(p, text, flags=re.IGNORECASE):
            return p
    return None


def check_input(text: str) -> GuardResult:
    if not text or not text.strip():
        return GuardResult(False, "empty input", "empty")
    if len(text) > INPUT_MAX_CHARS:
        return GuardResult(False, f"input exceeds {INPUT_MAX_CHARS} chars", "length")
    if hit := _match_any(_HARMFUL_INTENT, text):
        return GuardResult(False, f"harmful intent: {hit}", "harm")
    if hit := _match_any(_JAILBREAK_PATTERNS, text):
        return GuardResult(False, f"jailbreak pattern: {hit}", "jailbreak")
    return GuardResult(True)


def check_output(text: str) -> GuardResult:
    if not text:
        return GuardResult(False, "empty output", "empty")
    if hit := _match_any(_TOXIC_OUTPUT_PATTERNS, text):
        return GuardResult(False, f"toxic output: {hit}", "toxic")
    if hit := _match_any(_OUTPUT_LEAK_PATTERNS, text):
        return GuardResult(False, f"meta leak: {hit}", "leak")
    return GuardResult(True)

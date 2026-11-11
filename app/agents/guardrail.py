"""Input guardrail agent: runs before anything else touches the message.

Blocks prompt-injection attempts and oversized input, and stops customers from
pasting card numbers (they are masked before the message is stored).
"""
from __future__ import annotations

import re
import time

from app.agents.state import StoreState

MAX_CHARS = 2000
INJECTION_PATTERNS = [
    r"ignore (?:all |any )?(?:previous|prior|above) (?:instructions|rules)",
    r"(?:reveal|show|print).{0,20}system prompt",
    r"you are now", r"developer mode", r"jailbreak", r"act as (?:an? )?(?:admin|developer|system)",
    r"set (?:the )?price to", r"give me (?:a )?(?:100|free)",
]
CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def _luhn_ok(number: str) -> bool:
    digits = [int(d) for d in number if d.isdigit()]
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        checksum += d
    return len(digits) >= 13 and checksum % 10 == 0


def mask_cards(text: str) -> tuple[str, bool]:
    found = False

    def repl(m: re.Match) -> str:
        nonlocal found
        if _luhn_ok(m.group(0)):
            found = True
            return "[card number removed]"
        return m.group(0)

    return CARD_RE.sub(repl, text), found


async def guardrail_node(state: StoreState) -> dict:
    start = time.perf_counter()
    text = state.get("user_input", "")
    masked, had_card = mask_cards(text)
    reason = ""
    if len(text) > MAX_CHARS:
        reason = "too_long"
    elif any(re.search(p, text, re.I) for p in INJECTION_PATTERNS):
        reason = "prompt_injection"
    elif had_card:
        reason = "card_number"
    return {
        "blocked": bool(reason),
        "block_reason": reason,
        "messages": [{"role": "user", "content": masked}],
        "plan": [], "intents": [], "entities": {}, "answer": "",
        "trace": [{"agent": "guardrail", "ms": round((time.perf_counter() - start) * 1000, 1),
                   "tools": [], "result": reason or "pass"}],
    }

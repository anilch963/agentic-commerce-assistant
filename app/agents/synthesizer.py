"""Response synthesizer: merges specialist findings into one reply.

With an LLM configured it rewrites the findings into a natural answer, then a
grounding check verifies that every rupee amount in that answer appears in the
tool data. If the model invented a price, the deterministic answer is used instead.
"""
from __future__ import annotations

import json
import re
import time

from app.agents.state import StoreState
from app.llm import complete

BLOCKED_REPLIES = {
    "prompt_injection": "I can only help with shopping for iPhones: products, prices, orders and store policies.",
    "card_number": "Please don't share card numbers in chat. I've removed it from the conversation. "
                   "You'll enter payment details securely at checkout.",
    "too_long": "That message is too long for me to process. Could you shorten it?",
}
SMALLTALK_REPLY = ("Hi! I'm the iPhone store assistant. I can compare models, recommend one for your budget, "
                   "check stock and offers in your city, value a trade-in, manage your cart and track orders.")
OUT_OF_SCOPE_REPLY = ("I can only help with the iPhone store: models, prices, stock, trade-ins, orders and "
                      "policies like returns and warranty. What are you looking for?")

WRITER_PROMPT = """You are a friendly iPhone store assistant. Rewrite the specialist findings below into one
clear reply. Use ONLY facts in the findings; never add prices, dates or policies. Keep tables, keep every
[source] citation and keep the order confirmation question if present. Use Rs for prices."""

MONEY_RE = re.compile(r"(?:rs\.?|₹|inr)\s*([\d,]+)", re.I)


def _amounts(text: str) -> set[int]:
    return {int(m.replace(",", "")) for m in MONEY_RE.findall(text) if m.replace(",", "")}


def is_grounded(answer: str, findings: list[dict]) -> bool:
    known: set[int] = set()
    for f in findings:
        blob = json.dumps(f.get("data", {}), default=str) + " " + f.get("text", "")
        known |= {int(n) for n in re.findall(r"\d+", blob.replace(",", ""))}
    return _amounts(answer) <= known


async def synthesizer_node(state: StoreState) -> dict:
    start = time.perf_counter()
    findings = state.get("findings", [])
    intents = state.get("intents", [])
    grounded, writer = True, "rules"

    if state.get("blocked"):
        answer = BLOCKED_REPLIES[state["block_reason"]]
    elif not findings:
        answer = SMALLTALK_REPLY if "smalltalk" in intents else OUT_OF_SCOPE_REPLY
    else:
        answer = "\n\n".join(f["text"] for f in findings)
        if len(findings) > 1:  # single findings are already final; only merge multi-agent turns
            payload = json.dumps([{"agent": f["agent"], "text": f["text"]} for f in findings], ensure_ascii=False)
            rewritten = await complete(WRITER_PROMPT, f"Customer: {state['user_input']}\nFindings: {payload}")
            if rewritten:
                if is_grounded(rewritten, findings):
                    answer, writer = rewritten.strip(), "llm"
                else:
                    grounded = False  # model added an amount the tools never returned
    return {
        "answer": answer,
        "grounded": grounded,
        "messages": [{"role": "assistant", "content": answer}],
        "trace": [{"agent": "synthesizer", "ms": round((time.perf_counter() - start) * 1000, 1), "tools": [],
                   "writer": writer, "grounding_check": "pass" if grounded else "failed_fallback"}],
    }

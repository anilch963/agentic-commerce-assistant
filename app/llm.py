"""Thin LLM layer. Returns None when no LLM is configured, so callers fall back
to deterministic logic. That keeps the system usable (and testable) without a key."""
from __future__ import annotations

import json
import logging
import re

from app.config import settings

log = logging.getLogger(__name__)


def llm_enabled() -> bool:
    return settings.llm_provider == "anthropic" and bool(settings.anthropic_api_key)


async def complete(system: str, user: str, max_tokens: int = 800) -> str | None:
    if not llm_enabled():
        return None
    try:
        from anthropic import AsyncAnthropic  # optional dependency

        client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        resp = await client.messages.create(
            model=settings.llm_model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")
    except Exception as exc:  # network, auth, rate limit: degrade, don't crash the turn
        log.warning("LLM call failed, falling back to deterministic path: %s", exc)
        return None


async def complete_json(system: str, user: str) -> dict | None:
    text = await complete(system + "\nRespond with a single JSON object and nothing else.", user)
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        return None

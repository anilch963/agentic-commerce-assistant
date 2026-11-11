"""Shared LangGraph state.

Per-turn fields (findings, trace) use a reducer that resets when the turn starts
with `None`. Conversation fields (messages, last_skus, city, pending_action) are
persisted across turns by the checkpointer, keyed by session_id.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


def per_turn(old: list | None, new: list | None) -> list:
    """Append within a turn; passing None (at turn start) clears the list."""
    if new is None:
        return []
    return (old or []) + new


class Finding(TypedDict, total=False):
    agent: str
    text: str                 # deterministic markdown answer for this part
    data: Any                 # raw tool output, used for grounding checks
    citations: list[str]
    error: str


class StoreState(TypedDict, total=False):
    # input
    session_id: str
    user_input: str

    # conversation memory (persisted)
    messages: Annotated[list[dict], operator.add]
    last_skus: list[str]
    city: str
    pending_action: dict | None

    # set by guardrail / supervisor each turn
    blocked: bool
    block_reason: str
    intents: list[str]
    plan: list[str]           # remaining agents to run this turn
    entities: dict[str, Any]

    # produced by agents
    findings: Annotated[list[Finding], per_turn]
    trace: Annotated[list[dict], per_turn]
    answer: str
    grounded: bool

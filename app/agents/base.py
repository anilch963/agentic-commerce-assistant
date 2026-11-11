"""Helpers shared by every specialist agent."""
from __future__ import annotations

import functools
import time
from collections.abc import Awaitable, Callable

from app.agents.state import Finding, StoreState
from app.mcp_client import ToolCall, ToolCallError, toolbox


def inr(amount: int | float) -> str:
    """Indian digit grouping: 149900 -> 'Rs 1,49,900'."""
    s = str(int(round(amount)))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        s = ",".join(groups) + "," + tail
    return f"Rs {s}"


class AgentContext:
    """Passed to each agent: tool access with automatic call logging."""

    def __init__(self, state: StoreState):
        self.state = state
        self.entities = state.get("entities", {})
        self.calls: list[ToolCall] = []

    async def tool(self, server: str, name: str, **args) -> dict:
        return await toolbox.call(server, name, {k: v for k, v in args.items() if v is not None}, log=self.calls)

    @property
    def skus(self) -> list[str]:
        return self.entities.get("skus") or self.state.get("last_skus") or []

    @property
    def city(self) -> str:
        from app.config import settings
        return self.entities.get("city") or self.state.get("city") or settings.default_city


AgentFn = Callable[[AgentContext], Awaitable[tuple[Finding, dict]]]


def agent(name: str):
    """Wrap an agent: pop it from the plan, time it, log tool calls, turn tool errors into findings."""

    def decorator(fn: AgentFn):
        @functools.wraps(fn)
        async def node(state: StoreState) -> dict:
            ctx = AgentContext(state)
            start = time.perf_counter()
            extra: dict = {}
            try:
                finding, extra = await fn(ctx)
            except ToolCallError as exc:
                finding = Finding(text=f"Sorry, I couldn't complete that: {exc}", error=str(exc))
            finding["agent"] = name
            trace = {"agent": name, "ms": round((time.perf_counter() - start) * 1000, 1),
                     "tools": [f"{c.server}.{c.tool}" + ("" if c.ok else " (error)") for c in ctx.calls]}
            return {"findings": [finding], "trace": [trace], "plan": state.get("plan", [])[1:], **extra}

        node.agent_name = name
        return node

    return decorator

"""MCP toolbox used by the agents.

Each agent talks to tools only through this class, never by importing server code.
By default the servers run in-process (fast, no ports). Set CATALOG_MCP_URL etc.
to point at remote servers over streamable HTTP; agent code does not change.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from mcp.client import Client

from app.config import settings
from app.mcp_servers import catalog_server, orders_server, support_server


class ToolCallError(RuntimeError):
    """A tool returned isError=True. The message is safe to show the model or the user."""


@dataclass
class ToolCall:
    server: str
    tool: str
    args: dict
    ok: bool
    ms: float


class MCPToolbox:
    def __init__(self) -> None:
        self.targets: dict[str, Any] = {
            "catalog": settings.catalog_mcp_url or catalog_server.server,
            "orders": settings.orders_mcp_url or orders_server.server,
            "support": settings.support_mcp_url or support_server.server,
        }

    async def call(self, server: str, tool: str, args: dict | None = None,
                   log: list[ToolCall] | None = None) -> dict:
        args = args or {}
        start = time.perf_counter()
        ok = False
        try:
            async with Client(self.targets[server]) as client:
                result = await client.call_tool(tool, args)
            text = "\n".join(c.text for c in result.content if getattr(c, "type", "") == "text")
            if result.is_error:
                raise ToolCallError(text.split(": ", 1)[-1] if text.startswith("Error executing tool") else text)
            ok = True
            if result.structured_content is not None:
                return result.structured_content
            return json.loads(text) if text else {}
        finally:
            if log is not None:
                log.append(ToolCall(server, tool, args, ok, round((time.perf_counter() - start) * 1000, 1)))

    async def list_tools(self) -> dict[str, list[str]]:
        out = {}
        for name, target in self.targets.items():
            async with Client(target) as client:
                out[name] = [t.name for t in (await client.list_tools()).tools]
        return out


toolbox = MCPToolbox()

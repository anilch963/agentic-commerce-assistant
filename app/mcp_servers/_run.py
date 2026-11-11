"""Shared CLI for running an MCP server over stdio or streamable HTTP."""
from __future__ import annotations

import argparse

from mcp.server.mcpserver import MCPServer


def run_cli(server: MCPServer, default_port: int) -> None:
    parser = argparse.ArgumentParser(description=f"Run the {server.name} MCP server")
    parser.add_argument("--http", action="store_true", help="serve streamable HTTP at /mcp (default: stdio)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=default_port)
    args = parser.parse_args()
    if args.http:
        server.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        server.run()

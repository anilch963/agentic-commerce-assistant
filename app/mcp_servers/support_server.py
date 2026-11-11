"""Support MCP server: RAG search over store policies.

Run standalone:  python -m app.mcp_servers.support_server --http --port 8103
"""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from app.config import DATA_DIR, settings
from app.rag.retriever import Retriever, TfidfRetriever

POLICY_DIR = DATA_DIR / "policies"


def build_support_server(retriever: Retriever | None = None) -> MCPServer:
    retriever = retriever or TfidfRetriever.from_dir(POLICY_DIR)
    mcp = MCPServer("support", instructions="Search returns, warranty, shipping, payment and trade-in policies.")

    @mcp.tool()
    def search_policies(query: str, top_k: int = settings.rag_top_k) -> dict:
        """Retrieve the most relevant policy sections. Every hit has a `source` to cite."""
        hits = retriever.search(query, top_k=top_k, min_score=settings.rag_min_score)
        return {"query": query, "hits": [
            {"source": h.chunk.source, "title": h.chunk.title, "text": h.chunk.text, "score": h.score} for h in hits
        ]}

    @mcp.resource("store://policies/{name}")
    def policy(name: str) -> str:
        """Full text of a policy document, e.g. store://policies/returns"""
        path = POLICY_DIR / f"{name}.md"
        if not path.exists():
            raise ToolError(f"No policy named {name!r}")
        return path.read_text(encoding="utf-8")

    return mcp


server = build_support_server()

if __name__ == "__main__":
    from app.mcp_servers._run import run_cli
    run_cli(server, default_port=8103)

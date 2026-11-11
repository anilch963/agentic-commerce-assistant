"""Runtime settings, read from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


@dataclass(frozen=True)
class Settings:
    # "offline" runs fully deterministic (no API key needed, used in tests/CI).
    # "anthropic" uses Claude for intent routing and answer writing.
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "offline"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "claude-sonnet-5-5"))
    anthropic_api_key: str | None = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))

    # Leave empty to run MCP servers in-process. Set to http://host:port/mcp to use remote servers.
    catalog_mcp_url: str = field(default_factory=lambda: os.getenv("CATALOG_MCP_URL", ""))
    orders_mcp_url: str = field(default_factory=lambda: os.getenv("ORDERS_MCP_URL", ""))
    support_mcp_url: str = field(default_factory=lambda: os.getenv("SUPPORT_MCP_URL", ""))

    rag_top_k: int = field(default_factory=lambda: int(os.getenv("RAG_TOP_K", "3")))
    rag_min_score: float = field(default_factory=lambda: float(os.getenv("RAG_MIN_SCORE", "0.08")))
    default_city: str = field(default_factory=lambda: os.getenv("DEFAULT_CITY", "Bengaluru"))


settings = Settings()

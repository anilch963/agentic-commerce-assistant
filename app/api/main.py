"""FastAPI service.

    POST /chat                  one conversational turn
    GET  /sessions/{id}         conversation history for a session
    GET  /tools                 tools exposed by each MCP server
    GET  /health                liveness
    GET  /                      minimal chat UI
"""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.graph import chat, graph
from app.mcp_client import toolbox

app = FastAPI(title="iPhone Store Agents", version="1.0.0",
              description="Multi-agent shopping assistant: LangGraph + MCP + RAG")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000, examples=["Compare the iPhone 17 and 17 Pro"])
    session_id: str | None = Field(default=None, description="Reuse to keep cart and context between turns")


class TraceStep(BaseModel):
    agent: str
    ms: float
    tools: list[str] = []


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    intents: list[str]
    agents: list[str]
    citations: list[str]
    pending_confirmation: bool
    grounded: bool
    trace: list[TraceStep]


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "llm_provider": settings.llm_provider}


@app.get("/tools")
async def tools() -> dict:
    return await toolbox.list_tools()


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id or uuid.uuid4().hex[:12]
    result = await chat(req.message, session_id)
    return ChatResponse(session_id=session_id, **result)


@app.get("/sessions/{session_id}")
async def session_history(session_id: str) -> dict:
    snapshot = await graph.aget_state({"configurable": {"thread_id": session_id}})
    if not snapshot.values:
        raise HTTPException(404, "Unknown session")
    v = snapshot.values
    return {"session_id": session_id, "messages": v.get("messages", []), "city": v.get("city"),
            "last_skus": v.get("last_skus", []), "pending_action": v.get("pending_action")}


@app.get("/", response_class=HTMLResponse)
async def ui() -> str:
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")

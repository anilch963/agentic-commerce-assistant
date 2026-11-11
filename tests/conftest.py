import asyncio
import os
import uuid

import pytest

os.environ.setdefault("LLM_PROVIDER", "offline")  # tests never need an API key

from app.graph import chat  # noqa: E402
from app.mcp_servers.orders_server import store  # noqa: E402


@pytest.fixture(autouse=True)
def reset_orders():
    store.reset()
    yield


@pytest.fixture
def run():
    return asyncio.run


@pytest.fixture
def session():
    """A fresh conversation. Call it with a message, get the chat result back."""
    sid = f"test-{uuid.uuid4().hex[:8]}"

    def send(message: str) -> dict:
        return asyncio.run(chat(message, sid))

    send.id = sid
    return send

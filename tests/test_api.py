from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_tools_endpoint_lists_mcp_tools():
    tools = client.get("/tools").json()
    assert "check_stock" in tools["catalog"] and "place_order" in tools["orders"]


def test_chat_round_trip_keeps_session():
    r1 = client.post("/chat", json={"message": "tell me about the iphone 16e"}).json()
    sid = r1["session_id"]
    assert "iPhone 16e" in r1["answer"]
    r2 = client.post("/chat", json={"message": "add it to cart", "session_id": sid}).json()
    assert "Added iPhone 16e" in r2["answer"]

    history = client.get(f"/sessions/{sid}").json()
    assert [m["role"] for m in history["messages"]] == ["user", "assistant", "user", "assistant"]
    assert history["last_skus"] == ["iphone-16e"]


def test_validation_and_404():
    assert client.post("/chat", json={"message": ""}).status_code == 422
    assert client.get("/sessions/does-not-exist").status_code == 404


def test_ui_served():
    assert "iPhone Store Assistant" in client.get("/").text

"""LLM-mode behaviour with the model stubbed out: routing validation and the grounding check."""
import asyncio
import sys
import types

from app import llm
from app.agents import supervisor, synthesizer
from app.agents.synthesizer import is_grounded


def test_grounding_check():
    findings = [{"text": "iPhone 17 is Rs 82,900", "data": {"price_inr": 82900}}]
    assert is_grounded("The iPhone 17 costs Rs 82,900.", findings)
    assert not is_grounded("The iPhone 17 costs Rs 79,900.", findings)


def test_hallucinated_price_falls_back_to_tool_answer(session, monkeypatch):
    async def fake_complete(system, user, max_tokens=800):
        return "Great news, the iPhone 17 is just Rs 9,999 today!"

    monkeypatch.setattr(synthesizer, "complete", fake_complete)
    r = session("price of the iphone 17 and the emi options")
    assert r["grounded"] is False
    assert "Rs 9,999" not in r["answer"] and "Rs 82,900" in r["answer"]
    assert r["trace"][-1]["grounding_check"] == "failed_fallback"


def test_grounded_rewrite_is_used(session, monkeypatch):
    async def fake_complete(system, user, max_tokens=800):
        return "The iPhone 17 is Rs 82,900, and no-cost EMI runs 3, 6 or 9 months."

    monkeypatch.setattr(synthesizer, "complete", fake_complete)
    r = session("price of the iphone 17 and the emi options")
    assert r["grounded"] is True and r["answer"].startswith("The iPhone 17 is Rs 82,900")


def test_llm_router_cannot_invent_intents_or_skus(session, monkeypatch):
    async def fake_json(system, user):
        return {"intents": ["delete_database", "compare"], "product_names": ["iPhone 17", "Galaxy S25"]}

    monkeypatch.setattr(supervisor, "complete_json", fake_json)
    r = session("which is better, the 17 or the samsung?")
    assert r["intents"] == ["compare"]
    assert "product_expert" in r["agents"]
    assert "Galaxy" not in r["answer"]


def test_llm_outage_degrades_to_deterministic_path(monkeypatch):
    class Boom:
        def __init__(self, **kwargs):
            raise ConnectionError("network down")

    monkeypatch.setattr(llm, "llm_enabled", lambda: True)
    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(AsyncAnthropic=Boom))
    assert asyncio.run(llm.complete("system", "user")) is None
    assert asyncio.run(llm.complete_json("system", "user")) is None

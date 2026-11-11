"""Supervisor agent: decides which specialists run, and in what order.

1. Deterministic NLU always runs (cheap, testable, resolves SKUs exactly).
2. If an LLM is configured, it classifies intents too, which handles phrasing the
   keyword rules miss. Its output is validated against the allowed intent set and
   merged with the NLU entities, so the LLM can add but never invent SKUs.
"""
from __future__ import annotations

import time

from app.agents.state import StoreState
from app.llm import complete_json
from app.nlu import INTENT_ORDER, detect_intents, extract_entities

INTENT_TO_AGENT = {
    "product_info": "product_expert",
    "compare": "product_expert",
    "recommend": "recommender",
    "stock_price": "inventory",
    "policy": "support",
    "trade_in": "trade_in",
    "cart": "cart",
    "checkout": "cart",
    "confirm": "cart",
    "abort": "cart",
    "order_status": "order_tracking",
    "cancel_order": "order_tracking",
}

ROUTER_PROMPT = f"""You route messages for an iPhone store assistant.
Allowed intents: {", ".join(INTENT_ORDER)}, smalltalk, out_of_scope.
- product_info: specs of named iPhones. compare: two or more models side by side.
- recommend: help choosing (budget, use case). stock_price: availability, delivery, price, offers.
- policy: returns, warranty, AppleCare, shipping, payment, EMI. trade_in: value of an old iPhone.
- cart: add/view cart. checkout: start placing an order. confirm/abort: answer to a pending order confirmation.
- order_status / cancel_order: existing orders (IDs look like ORD-1234).
Return {{"intents": [...], "product_names": [...], "budget_inr": int|null, "city": str|null}}.
A message may have several intents. Use out_of_scope for anything unrelated to shopping for iPhones."""


async def _llm_route(text: str, state: StoreState) -> dict | None:
    context = f"Pending order confirmation: {bool(state.get('pending_action'))}. " \
              f"Last discussed products: {state.get('last_skus') or 'none'}.\nMessage: {text}"
    out = await complete_json(ROUTER_PROMPT, context)
    if not out or not isinstance(out.get("intents"), list):
        return None
    allowed = set(INTENT_ORDER) | {"smalltalk", "out_of_scope"}
    out["intents"] = [i for i in out["intents"] if i in allowed]
    return out


async def supervisor_node(state: StoreState) -> dict:
    start = time.perf_counter()
    text = state["user_input"]
    entities = extract_entities(text)
    intents = detect_intents(text, entities, state.get("pending_action"))
    router = "rules"

    llm = await _llm_route(text, state)
    if llm is not None:
        router = "llm"
        for name in llm.get("product_names") or []:
            for sku in extract_entities(name).get("skus", []):
                entities.setdefault("skus", [])
                if sku not in entities["skus"]:
                    entities["skus"].append(sku)
        if llm.get("budget_inr") and "budget_inr" not in entities:
            entities["budget_inr"] = int(llm["budget_inr"])
        if llm.get("city") and "city" not in entities:
            entities["city"] = str(llm["city"]).title()
        llm_intents = [i for i in llm["intents"] if i in INTENT_TO_AGENT]
        if llm_intents or "out_of_scope" in llm["intents"]:
            intents = [i for i in INTENT_ORDER if i in set(llm_intents)]

    plan = list(dict.fromkeys(INTENT_TO_AGENT[i] for i in intents if i in INTENT_TO_AGENT))
    update: dict = {
        "intents": intents or ["out_of_scope"],
        "entities": entities,
        "plan": plan,
        "trace": [{"agent": "supervisor", "ms": round((time.perf_counter() - start) * 1000, 1), "tools": [],
                   "router": router, "intents": intents, "plan": plan}],
    }
    if entities.get("skus"):
        update["last_skus"] = entities["skus"]
    if entities.get("city"):
        update["city"] = entities["city"]
    return update

import pytest

from app.nlu import detect_intents, extract_entities


@pytest.mark.parametrize("text, intents, skus", [
    ("Compare iPhone 17 Pro vs 17 Pro Max", ["compare"], ["iphone-17-pro", "iphone-17-pro-max"]),
    ("best iphone for photography under 1 lakh", ["recommend"], None),
    ("is the 17 pro in stock in bangalore?", ["stock_price"], ["iphone-17-pro"]),
    ("what is the return policy for opened phones", ["policy"], None),
    ("trade in my iphone 13 in good condition for a 17 pro", ["trade_in"], ["iphone-17-pro"]),
    ("add the air 512gb to my cart", ["cart"], ["iphone-air"]),
    ("checkout with upi", ["checkout"], None),
    ("where is my order ORD-1001", ["order_status"], None),
    ("cancel order ORD-1002", ["cancel_order"], None),
    ("tell me about iphone 16e", ["product_info"], ["iphone-16e"]),
    ("pro max specs", ["product_info"], ["iphone-17-pro-max"]),
    ("price of iphone 17 and the emi options", ["stock_price", "policy"], ["iphone-17"]),
    ("hi", ["smalltalk"], None),
    ("what is the weather today", [], None),
])
def test_routing(text, intents, skus):
    ent = extract_entities(text)
    assert detect_intents(text, ent) == intents
    assert ent.get("skus") == skus


def test_trade_in_device_is_not_confused_with_target():
    ent = extract_entities("exchange my old iphone 15 pro for the 17 pro max")
    assert ent["trade_in_model"] == "iphone 15 pro"
    assert ent["skus"] == ["iphone-17-pro-max"]


@pytest.mark.parametrize("text, budget", [
    ("under 80k", 80_000), ("below Rs 75,000", 75_000), ("budget of 1.2 lakh", 120_000), ("under ₹60000", 60_000),
])
def test_budget_parsing(text, budget):
    assert extract_entities(text)["budget_inr"] == budget


def test_budget_number_is_not_read_as_a_model():
    assert "skus" not in extract_entities("something under 17k")


def test_storage_and_payment():
    ent = extract_entities("17 pro 1tb, pay with credit card")
    assert ent["storage_gb"] == 1024 and ent["payment_method"] == "card"


def test_confirmation_only_counts_when_something_is_pending():
    assert detect_intents("yes", {}, pending_action={"type": "place_order"}) == ["confirm"]
    assert detect_intents("no thanks", {}, pending_action={"type": "place_order"}) == ["abort"]
    assert detect_intents("yes", {}, pending_action=None) == []

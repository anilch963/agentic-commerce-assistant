"""Contract tests: call each MCP server through a real MCP client, the same way the agents do."""
import pytest

from app.mcp_client import ToolCallError, toolbox


def test_every_server_exposes_its_tools(run):
    tools = run(toolbox.list_tools())
    assert set(tools["catalog"]) == {"search_products", "get_product", "compare_products", "check_stock",
                                     "get_offers", "estimate_trade_in"}
    assert set(tools["orders"]) == {"add_to_cart", "view_cart", "clear_cart", "place_order", "get_order",
                                    "cancel_order"}
    assert tools["support"] == ["search_policies"]


def test_search_respects_budget(run):
    res = run(toolbox.call("catalog", "search_products", {"max_price_inr": 70_000}))
    assert {p["sku"] for p in res["products"]} == {"iphone-16", "iphone-16e"}


def test_stock_is_case_insensitive_and_lists_alternatives(run):
    res = run(toolbox.call("catalog", "check_stock", {"sku": "iphone-17-pro-max", "city": "hyderabad"}))
    assert res["in_stock"] is False and "Mumbai" in res["available_in"]


def test_best_offer_respects_cap(run):
    res = run(toolbox.call("catalog", "get_offers", {"price_inr": 149_900}))
    assert res["best"] == {"code": "BANK5", "saving_inr": 6000, "effective_price_inr": 143_900}


def test_trade_in_condition_multiplier(run):
    res = run(toolbox.call("catalog", "estimate_trade_in", {"model": "iPhone 14", "condition": "cracked"}))
    assert res["value_inr"] == 8400


@pytest.mark.parametrize("server, tool, args, message", [
    ("catalog", "get_product", {"sku": "nokia-3310"}, "Unknown SKU"),
    ("catalog", "compare_products", {"skus": ["iphone-17"]}, "between 2 and 4"),
    ("catalog", "estimate_trade_in", {"model": "iPhone 8"}, "No trade-in price"),
    ("orders", "add_to_cart", {"session_id": "s", "sku": "iphone-17", "storage_gb": 64}, "comes in"),
    ("orders", "place_order", {"session_id": "empty", "city": "Delhi"}, "Cart is empty"),
    ("orders", "cancel_order", {"order_id": "ORD-1001"}, "already shipped"),
])
def test_tool_errors_are_returned_not_raised_by_server(run, server, tool, args, message):
    with pytest.raises(ToolCallError, match=message):
        run(toolbox.call(server, tool, args))


def test_cod_limit(run):
    run(toolbox.call("orders", "add_to_cart", {"session_id": "s1", "sku": "iphone-17-pro"}))
    with pytest.raises(ToolCallError, match="Cash on delivery"):
        run(toolbox.call("orders", "place_order", {"session_id": "s1", "city": "Delhi", "payment_method": "cod"}))


def test_storage_pricing(run):
    res = run(toolbox.call("orders", "add_to_cart", {"session_id": "s2", "sku": "iphone-17-pro", "storage_gb": 1024}))
    assert res["added"]["unit_price_inr"] == 134_900 + 2 * 20_000

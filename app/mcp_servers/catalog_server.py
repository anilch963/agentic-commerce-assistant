"""Catalog MCP server: products, stock, offers and trade-in quotes.

Run standalone:  python -m app.mcp_servers.catalog_server --http --port 8101
"""
from __future__ import annotations

import json
from functools import lru_cache

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from app.config import DATA_DIR

TRADE_IN_CONDITION = {"excellent": 1.0, "good": 0.85, "fair": 0.6, "cracked": 0.3}


@lru_cache
def load_catalog() -> dict:
    return json.loads((DATA_DIR / "catalog.json").read_text(encoding="utf-8"))


def _product(sku: str) -> dict:
    for p in load_catalog()["products"]:
        if p["sku"] == sku:
            return p
    raise ToolError(f"Unknown SKU {sku!r}. Call search_products to find valid SKUs.")


def _public(p: dict) -> dict:
    return {k: v for k, v in p.items() if k != "aliases"}


def build_catalog_server() -> MCPServer:
    mcp = MCPServer("catalog", instructions="iPhone catalog, stock by city, offers and trade-in quotes.")

    @mcp.tool()
    def search_products(query: str = "", max_price_inr: int | None = None, tags: list[str] | None = None) -> dict:
        """Search iPhones by free text, max price and tags (camera, battery, budget, compact, gaming...)."""
        q = query.lower()
        out = []
        for p in load_catalog()["products"]:
            if max_price_inr and p["price_inr"] > max_price_inr:
                continue
            if tags and not set(t.lower() for t in tags) & set(p["tags"]):
                continue
            if q and q not in p["name"].lower() and not any(q in a for a in p["aliases"]):
                continue
            out.append(_public(p))
        return {"products": out}

    @mcp.tool()
    def get_product(sku: str) -> dict:
        """Full specs for one SKU."""
        return _public(_product(sku))

    @mcp.tool()
    def compare_products(skus: list[str]) -> dict:
        """Side-by-side spec comparison for 2-4 SKUs."""
        if not 2 <= len(skus) <= 4:
            raise ToolError("compare_products needs between 2 and 4 SKUs")
        fields = ["price_inr", "display_in", "chip", "camera", "battery_hrs_video", "weight_g", "storage_options_gb"]
        rows = {sku: {f: _product(sku)[f] for f in fields} | {"name": _product(sku)["name"]} for sku in skus}
        return {"fields": fields, "rows": rows}

    @mcp.tool()
    def check_stock(sku: str, city: str) -> dict:
        """Units in stock for a SKU in a city, plus other cities that have it."""
        _product(sku)
        stock = load_catalog()["inventory"].get(sku, {})
        city_key = next((c for c in stock if c.lower() == city.lower()), city.title())
        units = stock.get(city_key, 0)
        return {
            "sku": sku,
            "city": city_key,
            "units": units,
            "in_stock": units > 0,
            "delivery_days": "1-2" if units > 0 else "3-5",
            "available_in": [c for c, n in stock.items() if n > 0],
        }

    @mcp.tool()
    def get_offers(price_inr: int | None = None) -> dict:
        """Current offers. If price_inr is given, also returns the best effective price."""
        offers = load_catalog()["offers"]
        result: dict = {"offers": offers}
        if price_inr:
            savings = [min(price_inr * o["pct"] // 100, o["max_inr"]) if "pct" in o else o["flat_inr"] for o in offers]
            best = max(range(len(offers)), key=lambda i: savings[i])
            result["best"] = {"code": offers[best]["code"], "saving_inr": savings[best],
                              "effective_price_inr": price_inr - savings[best]}
        return result

    @mcp.tool()
    def estimate_trade_in(model: str, condition: str = "good") -> dict:
        """Trade-in value for an old iPhone. condition: excellent | good | fair | cracked."""
        base = load_catalog()["trade_in_base_inr"]
        key = " ".join(model.lower().replace("apple", "").split())
        if not key.startswith("iphone"):
            key = f"iphone {key}"
        if key not in base:
            raise ToolError(f"No trade-in price for {model!r}. Eligible: {', '.join(sorted(base))}")
        cond = condition.lower()
        if cond not in TRADE_IN_CONDITION:
            raise ToolError(f"condition must be one of {list(TRADE_IN_CONDITION)}")
        return {"model": key, "condition": cond, "base_inr": base[key],
                "value_inr": int(base[key] * TRADE_IN_CONDITION[cond])}

    @mcp.resource("store://catalog")
    def catalog_resource() -> str:
        """The whole catalog as JSON."""
        return json.dumps([_public(p) for p in load_catalog()["products"]], indent=2)

    return mcp


server = build_catalog_server()

if __name__ == "__main__":
    from app.mcp_servers._run import run_cli
    run_cli(server, default_port=8101)

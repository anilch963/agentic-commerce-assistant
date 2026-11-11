"""Orders MCP server: cart, checkout, order tracking and cancellation.

State is in memory (a dict per process) so the demo needs no database.
Swap `OrderStore` for Postgres/Redis in production; the tool contract stays the same.

Run standalone:  python -m app.mcp_servers.orders_server --http --port 8102
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import date, timedelta

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from app.mcp_servers.catalog_server import load_catalog

COD_LIMIT_INR = 50_000
PAYMENT_METHODS = {"card", "upi", "netbanking", "cod", "emi"}


@dataclass
class OrderStore:
    carts: dict[str, list[dict]] = field(default_factory=dict)
    orders: dict[str, dict] = field(default_factory=dict)
    _ids: itertools.count = field(default_factory=lambda: itertools.count(2001))

    def reset(self) -> None:
        self.carts.clear()
        self.orders.clear()
        self._ids = itertools.count(2001)
        today = date.today()
        # Seeded orders so tracking works out of the box.
        self.orders.update({
            "ORD-1001": {"order_id": "ORD-1001", "status": "shipped", "city": "Bengaluru",
                         "items": [{"sku": "iphone-17", "storage_gb": 256, "qty": 1, "unit_price_inr": 82900}],
                         "total_inr": 82900, "eta": str(today + timedelta(days=1)), "courier": "BlueDart AWB 77123"},
            "ORD-1002": {"order_id": "ORD-1002", "status": "processing", "city": "Mumbai",
                         "items": [{"sku": "iphone-17-pro", "storage_gb": 512, "qty": 1, "unit_price_inr": 154900}],
                         "total_inr": 154900, "eta": str(today + timedelta(days=3)), "courier": None},
            "ORD-1003": {"order_id": "ORD-1003", "status": "delivered", "city": "Delhi",
                         "items": [{"sku": "iphone-16e", "storage_gb": 128, "qty": 1, "unit_price_inr": 59900}],
                         "total_inr": 59900, "eta": str(today - timedelta(days=4)), "courier": "Delhivery 55120"},
        })


store = OrderStore()
store.reset()


def _price(sku: str, storage_gb: int) -> int:
    for p in load_catalog()["products"]:
        if p["sku"] == sku:
            if storage_gb not in p["storage_options_gb"]:
                raise ToolError(f"{p['name']} comes in {p['storage_options_gb']} GB, not {storage_gb} GB")
            # Each storage step above the base adds Rs 20,000 (demo pricing rule).
            return p["price_inr"] + 20_000 * p["storage_options_gb"].index(storage_gb)
    raise ToolError(f"Unknown SKU {sku!r}")


def _cart_view(session_id: str) -> dict:
    items = store.carts.get(session_id, [])
    return {"session_id": session_id, "items": items,
            "total_inr": sum(i["unit_price_inr"] * i["qty"] for i in items)}


def build_orders_server() -> MCPServer:
    mcp = MCPServer("orders", instructions="Shopping cart, checkout and order tracking.")

    @mcp.tool()
    def add_to_cart(session_id: str, sku: str, storage_gb: int | None = None, qty: int = 1) -> dict:
        """Add an iPhone to the session's cart. storage_gb defaults to the base option."""
        if not 1 <= qty <= 3:
            raise ToolError("qty must be 1-3 (limit per customer)")
        product = next((p for p in load_catalog()["products"] if p["sku"] == sku), None)
        if product is None:
            raise ToolError(f"Unknown SKU {sku!r}")
        storage = storage_gb or product["storage_options_gb"][0]
        item = {"sku": sku, "name": product["name"], "storage_gb": storage, "qty": qty,
                "unit_price_inr": _price(sku, storage)}
        store.carts.setdefault(session_id, []).append(item)
        return {"added": item, "cart": _cart_view(session_id)}

    @mcp.tool()
    def view_cart(session_id: str) -> dict:
        """Current cart contents and total."""
        return _cart_view(session_id)

    @mcp.tool()
    def clear_cart(session_id: str) -> dict:
        """Empty the cart."""
        store.carts.pop(session_id, None)
        return _cart_view(session_id)

    @mcp.tool()
    def place_order(session_id: str, city: str, payment_method: str = "upi") -> dict:
        """Turn the cart into an order. Only call after the customer has explicitly confirmed."""
        cart = _cart_view(session_id)
        if not cart["items"]:
            raise ToolError("Cart is empty")
        pm = payment_method.lower()
        if pm not in PAYMENT_METHODS:
            raise ToolError(f"payment_method must be one of {sorted(PAYMENT_METHODS)}")
        if pm == "cod" and cart["total_inr"] > COD_LIMIT_INR:
            raise ToolError(f"Cash on delivery is only for orders under Rs {COD_LIMIT_INR:,}")
        order_id = f"ORD-{next(store._ids)}"
        order = {"order_id": order_id, "status": "processing", "city": city.title(), "items": cart["items"],
                 "total_inr": cart["total_inr"], "payment_method": pm,
                 "eta": str(date.today() + timedelta(days=2)), "courier": None}
        store.orders[order_id] = order
        store.carts.pop(session_id, None)
        return order

    @mcp.tool()
    def get_order(order_id: str) -> dict:
        """Status, ETA and items for an order ID like ORD-1001."""
        order = store.orders.get(order_id.upper())
        if not order:
            raise ToolError(f"No order {order_id!r}")
        return order

    @mcp.tool()
    def cancel_order(order_id: str) -> dict:
        """Cancel an order that has not shipped yet."""
        order = store.orders.get(order_id.upper())
        if not order:
            raise ToolError(f"No order {order_id!r}")
        if order["status"] != "processing":
            raise ToolError(f"{order['order_id']} is already {order['status']} and cannot be cancelled; "
                            "use the returns process instead")
        order["status"] = "cancelled"
        return order

    return mcp


server = build_orders_server()

if __name__ == "__main__":
    from app.mcp_servers._run import run_cli
    run_cli(server, default_port=8102)

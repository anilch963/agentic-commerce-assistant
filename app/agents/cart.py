"""Cart and checkout agent, with human-in-the-loop confirmation.

`place_order` is the only action in the system that commits money, so it never
runs on the same turn the customer asks to check out. The agent stores a
`pending_action` in graph state (persisted by the checkpointer) and waits for an
explicit "yes" on the next turn.
"""
from __future__ import annotations

import re

from app.agents.base import AgentContext, agent, inr
from app.agents.state import Finding


def _cart_lines(cart: dict) -> str:
    rows = [f"- {i['name']} {i['storage_gb']} GB x{i['qty']}: {inr(i['unit_price_inr'] * i['qty'])}"
            for i in cart["items"]]
    return "\n".join(rows) + f"\n**Total: {inr(cart['total_inr'])}**"


@agent("cart")
async def cart(ctx: AgentContext) -> tuple[Finding, dict]:
    intents = ctx.state.get("intents", [])
    session = ctx.state["session_id"]
    pending = ctx.state.get("pending_action")
    parts: list[str] = []
    extra: dict = {}
    data: dict = {}

    if "abort" in intents:
        return Finding(text="No problem, I haven't placed the order. Your cart is still saved."), \
            {"pending_action": None}

    if "confirm" in intents:
        if not pending:
            return Finding(text="There's nothing waiting for confirmation."), {}
        order = await ctx.tool("orders", "place_order", session_id=session, city=pending["city"],
                               payment_method=pending["payment_method"])
        return Finding(text=f"Order **{order['order_id']}** is placed. Total {inr(order['total_inr'])}, paid by "
                            f"{order['payment_method'].upper()}, arriving in {order['city']} by {order['eta']}.",
                       data=order), {"pending_action": None}

    if "cart" in intents:
        wants_add = re.search(r"\b(add|buy|take)\b", ctx.state["user_input"], re.I)
        if wants_add and ctx.skus:
            res = await ctx.tool("orders", "add_to_cart", session_id=session, sku=ctx.skus[0],
                                 storage_gb=ctx.entities.get("storage_gb"))
            item = res["added"]
            parts.append(f"Added {item['name']} {item['storage_gb']} GB ({inr(item['unit_price_inr'])}) to your cart.")
            data = res["cart"]
        elif wants_add:
            parts.append("Which model should I add to your cart?")
        else:
            data = await ctx.tool("orders", "view_cart", session_id=session)
            parts.append("Your cart:\n" + _cart_lines(data) if data["items"] else "Your cart is empty.")

    if "checkout" in intents:
        data = await ctx.tool("orders", "view_cart", session_id=session)
        if not data["items"]:
            parts.append("Your cart is empty, so there's nothing to check out yet.")
        else:
            payment = ctx.entities.get("payment_method", "upi")
            extra["pending_action"] = {"type": "place_order", "city": ctx.city, "payment_method": payment,
                                       "total_inr": data["total_inr"]}
            parts.append(f"Ready to place this order for delivery to {ctx.city}, paying by {payment.upper()}:\n"
                         f"{_cart_lines(data)}\n\nReply **yes** to confirm or **no** to cancel.")

    return Finding(text="\n\n".join(parts) or "What would you like to do with your cart?", data=data), extra

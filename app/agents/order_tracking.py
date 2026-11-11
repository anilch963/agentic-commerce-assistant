"""Order tracking agent: status, ETA and cancellation of existing orders."""
from __future__ import annotations

from app.agents.base import AgentContext, agent, inr
from app.agents.state import Finding

STATUS_TEXT = {
    "processing": "is being prepared and hasn't shipped yet",
    "shipped": "has shipped",
    "delivered": "was delivered",
    "cancelled": "is cancelled",
}


@agent("order_tracking")
async def order_tracking(ctx: AgentContext) -> tuple[Finding, dict]:
    order_id = ctx.entities.get("order_id")
    if not order_id:
        return Finding(text="Please share your order ID (it looks like ORD-1234)."), {}

    if "cancel_order" in ctx.state.get("intents", []):
        order = await ctx.tool("orders", "cancel_order", order_id=order_id)
        return Finding(text=f"Order {order['order_id']} is cancelled. The refund of {inr(order['total_inr'])} "
                            f"goes back to your original payment method in 5-7 business days.", data=order), {}

    order = await ctx.tool("orders", "get_order", order_id=order_id)
    items = ", ".join(f"{i['sku'].replace('-', ' ').title().replace('Iphone', 'iPhone')} {i['storage_gb']} GB"
                      for i in order["items"])
    text = f"Order {order['order_id']} ({items}) {STATUS_TEXT.get(order['status'], order['status'])}"
    if order["status"] == "delivered":
        text += f" on {order['eta']}."
    elif order["status"] != "cancelled":
        text += f". Expected delivery to {order['city']}: {order['eta']}."
    if order.get("courier"):
        text += f" Courier: {order['courier']}."
    return Finding(text=text, data=order), {}

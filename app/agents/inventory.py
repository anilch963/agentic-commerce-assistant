"""Inventory and pricing agent: stock in the customer's city, delivery time and best offer."""
from __future__ import annotations

from app.agents.base import AgentContext, agent, inr
from app.agents.state import Finding


@agent("inventory")
async def inventory(ctx: AgentContext) -> tuple[Finding, dict]:
    if not ctx.skus:
        return Finding(text="Which model should I check stock and price for?"), {}
    lines, data = [], []
    for sku in ctx.skus[:3]:
        product = await ctx.tool("catalog", "get_product", sku=sku)
        stock = await ctx.tool("catalog", "check_stock", sku=sku, city=ctx.city)
        offers = await ctx.tool("catalog", "get_offers", price_inr=product["price_inr"])
        best = offers["best"]
        if stock["in_stock"]:
            avail = f"in stock in {stock['city']} ({stock['units']} units), delivery in {stock['delivery_days']} days"
        else:
            others = ", ".join(stock["available_in"]) or "no other city"
            avail = (f"out of stock in {stock['city']} (available in {others}), "
                     f"delivery in {stock['delivery_days']} days")
        lines.append(f"**{product['name']}**: {inr(product['price_inr'])}, {avail}. "
                     f"With offer {best['code']} you pay {inr(best['effective_price_inr'])} "
                     f"(save {inr(best['saving_inr'])}).")
        data.append({"product": product, "stock": stock, "offers": offers})
    return Finding(text="\n".join(lines), data=data), {}

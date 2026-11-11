"""Trade-in agent: values the customer's old iPhone and shows the net price of the new one."""
from __future__ import annotations

from app.agents.base import AgentContext, agent, inr
from app.agents.state import Finding


@agent("trade_in")
async def trade_in(ctx: AgentContext) -> tuple[Finding, dict]:
    model = ctx.entities.get("trade_in_model")
    if not model:
        return Finding(text="Which iPhone would you like to trade in, and what condition is it in "
                            "(excellent, good, fair or cracked)? We accept iPhone 11 and newer."), {}
    condition = ctx.entities.get("condition", "good")
    quote = await ctx.tool("catalog", "estimate_trade_in", model=model, condition=condition)
    name = quote["model"].title().replace("Iphone", "iPhone")
    text = f"Your {name} in {condition} condition is worth about {inr(quote['value_inr'])}."
    data: dict = {"quote": quote}
    if ctx.entities.get("skus"):
        target = await ctx.tool("catalog", "get_product", sku=ctx.entities["skus"][0])
        net = target["price_inr"] - quote["value_inr"]
        text += f" That brings the {target['name']} down to {inr(net)} (from {inr(target['price_inr'])})."
        data["target"] = target
        data["net_price_inr"] = net
    if condition == "good" and "condition" not in ctx.entities:
        text += " I assumed good condition; tell me if it's excellent, fair or cracked."
    text += " The final value is confirmed when the courier inspects the phone."
    return Finding(text=text, data=data), {}

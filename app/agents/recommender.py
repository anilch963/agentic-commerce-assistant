"""Recommendation agent: ranks models by budget and use case.

Scoring is explicit and explainable: +3 per matched use-case tag, +1 per 5 hrs of
battery when battery matters, and a small value bonus for headroom under budget.
"""
from __future__ import annotations

from app.agents.base import AgentContext, agent, inr
from app.agents.state import Finding


def score(product: dict, use_cases: list[str], budget: int | None) -> float:
    s = 3.0 * len(set(use_cases) & set(product["tags"]))
    if "battery" in use_cases:
        s += product["battery_hrs_video"] / 5
    if budget:
        s += (budget - product["price_inr"]) / budget  # 0..1 bonus for staying under budget
    return round(s, 2)


@agent("recommender")
async def recommender(ctx: AgentContext) -> tuple[Finding, dict]:
    budget = ctx.entities.get("budget_inr")
    use_cases = ctx.entities.get("use_cases", [])
    data = await ctx.tool("catalog", "search_products", max_price_inr=budget)
    products = data["products"]
    if not products:
        cheapest = await ctx.tool("catalog", "search_products")
        low = min(cheapest["products"], key=lambda p: p["price_inr"])
        return Finding(text=f"No iPhone fits under {inr(budget)}. The most affordable is the {low['name']} at "
                            f"{inr(low['price_inr'])}; a trade-in can bring that down.", data=cheapest), {}

    ranked = sorted(products, key=lambda p: score(p, use_cases, budget), reverse=True)[:3]
    why = {
        "camera": lambda p: p["camera"], "video": lambda p: p["camera"],
        "battery": lambda p: f"up to {p['battery_hrs_video']} hrs video",
        "gaming": lambda p: f"{p['chip']} chip", "compact": lambda p: f'{p["display_in"]}" display',
        "lightweight": lambda p: f"{p['weight_g']} g", "budget": lambda p: "best value",
        "large screen": lambda p: f'{p["display_in"]}" display',
    }
    lines = []
    for i, p in enumerate(ranked, 1):
        reasons = [why[u](p) for u in use_cases if u in why] or [f"{p['chip']}, {p['camera']}"]
        lines.append(f"{i}. **{p['name']}** - {inr(p['price_inr'])} ({'; '.join(dict.fromkeys(reasons))})")
    intro = "Top picks" + (f" under {inr(budget)}" if budget else "") + \
            (f" for {', '.join(use_cases)}" if use_cases else "") + ":"
    return Finding(text=intro + "\n" + "\n".join(lines), data={"ranked": ranked}), \
        {"last_skus": [p["sku"] for p in ranked]}

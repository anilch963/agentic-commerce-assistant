"""Product expert agent: specs for one model, or a comparison table for several."""
from __future__ import annotations

from app.agents.base import AgentContext, agent, inr
from app.agents.state import Finding

ROW_LABELS = {"price_inr": "Price (base)", "display_in": "Display", "chip": "Chip", "camera": "Rear camera",
              "battery_hrs_video": "Battery (video playback)", "weight_g": "Weight",
              "storage_options_gb": "Storage"}


def _fmt(field: str, value) -> str:
    return {
        "price_inr": lambda v: inr(v),
        "display_in": lambda v: f'{v}"',
        "battery_hrs_video": lambda v: f"up to {v} hrs",
        "weight_g": lambda v: f"{v} g",
        "storage_options_gb": lambda v: " / ".join(f"{s // 1024} TB" if s >= 1024 else f"{s} GB" for s in v),
    }.get(field, str)(value)


@agent("product_expert")
async def product_expert(ctx: AgentContext) -> tuple[Finding, dict]:
    skus = ctx.skus
    if not skus:
        return Finding(text="Which iPhone would you like to know about? We stock the 17 Pro Max, 17 Pro, Air, 17, "
                            "16 and 16e."), {}

    if len(skus) >= 2:
        data = await ctx.tool("catalog", "compare_products", skus=skus[:4])
        rows = data["rows"]
        header = "| | " + " | ".join(rows[s]["name"] for s in rows) + " |"
        lines = [header, "|---" * (len(rows) + 1) + "|"]
        for f in data["fields"]:
            lines.append(f"| {ROW_LABELS[f]} | " + " | ".join(_fmt(f, rows[s][f]) for s in rows) + " |")
        cheapest = min(rows, key=lambda s: rows[s]["price_inr"])
        longest = max(rows, key=lambda s: rows[s]["battery_hrs_video"])
        if cheapest == longest:
            summary = f"The {rows[cheapest]['name']} is both the cheapest and the longest-lasting on battery."
        else:
            summary = (f"The {rows[cheapest]['name']} is the cheapest; the {rows[longest]['name']} has the longest "
                       f"battery life.")
        return Finding(text="\n".join(lines) + "\n\n" + summary, data=data), {}

    p = await ctx.tool("catalog", "get_product", sku=skus[0])
    text = (f"**{p['name']}** starts at {inr(p['price_inr'])}.\n"
            + "\n".join(f"- {ROW_LABELS[f]}: {_fmt(f, p[f])}" for f in ROW_LABELS if f != "price_inr")
            + f"\n- Colours: {', '.join(p['colors'])}")
    return Finding(text=text, data=p), {}

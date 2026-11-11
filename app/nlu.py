"""Deterministic intent + entity extraction.

Used directly in offline mode, and as the fallback / SKU resolver when an LLM
does the routing. Keeping this deterministic means every routing decision can be
unit-tested without an API key.
"""
from __future__ import annotations

import re
from typing import Any

# Order matters: longer names first so "17 pro max" is not read as "17 pro".
SKU_PATTERNS: list[tuple[str, str]] = [
    ("iphone-17-pro-max", r"(?:iphone\s*)?(?:17\s*)?pro\s*max"),
    ("iphone-17-pro", r"(?:iphone\s*)?17\s*pro\b"),
    ("iphone-air", r"iphone\s*air|\b17\s*air\b|\bthe\s+air\b"),
    ("iphone-16e", r"(?:iphone\s*)?\b16\s*e\b"),
    ("iphone-17", r"iphone\s*17\b|\b17\b(?!\s*(?:k\b|,|000|lakh))"),
    ("iphone-16", r"iphone\s*16\b|\b16\b(?!\s*(?:k\b|,|000|lakh|gb))"),
]

CITIES = {"bengaluru": "Bengaluru", "bangalore": "Bengaluru", "mumbai": "Mumbai", "delhi": "Delhi",
          "hyderabad": "Hyderabad", "chennai": "Chennai", "pune": "Pune", "kolkata": "Kolkata"}

USE_CASE_TAGS = {
    "camera": ["camera", "photo", "photography", "selfie", "zoom"],
    "video": ["video", "vlog", "filming", "youtube"],
    "battery": ["battery", "long lasting", "all day", "battery life"],
    "gaming": ["gaming", "games", "bgmi", "performance"],
    "compact": ["compact", "small", "one hand", "one-hand"],
    "lightweight": ["light", "lightweight", "thin", "slim"],
    "budget": ["cheap", "budget", "affordable", "student", "students"],
    "large screen": ["big screen", "large screen", "bigger screen"],
}

INTENT_KEYWORDS: dict[str, list[str]] = {
    "trade_in": [r"trade[\s-]?in", r"\bexchange\b", r"old (?:phone|iphone)"],
    "policy": [r"\breturn", r"refund", r"warrant", r"applecare", r"\bemi\b", r"cash on delivery", r"\bcod\b",
               r"shipping", r"delivery time", r"how long.*deliver", r"payment (?:option|method)s?",
               r"battery (?:service|replacement)", r"damaged", r"policy", r"cancellation"],
    "compare": [r"compare", r"\bvs\.?\b", r"versus", r"difference between", r"better than"],
    "recommend": [r"suggest", r"recommend", r"which (?:iphone|one|phone) should", r"\bbest\b.*(?:for|under)",
                  r"under\s*(?:rs|₹|inr)?\s*\d", r"budget", r"\bfor (?:photography|gaming|students?)"],
    "stock_price": [r"in stock", r"\bstock\b", r"availab", r"price", r"cost", r"how much", r"\boffers?\b",
                    r"discount", r"deliver(?:y)? (?:to|in)", r"can i get it"],
    "checkout": [r"check\s?out", r"place (?:the |my )?order", r"buy now", r"order it"],
    "cart": [r"add .*to (?:my |the )?cart", r"\badd (?:it|this|that|one)\b", r"\bcart\b", r"i(?:'ll| will) take",
             r"\bbuy\b"],
    "cancel_order": [r"cancel (?:my |the )?order", r"cancel ord-?\d"],
    "order_status": [r"where is my order", r"track", r"order status", r"status of"],
}

CONFIRM_RE = re.compile(r"^\s*(yes|yep|yeah|confirm|confirmed|go ahead|place it|do it|ok(?:ay)?|sure)\b", re.I)
ABORT_RE = re.compile(r"^\s*(no|nope|don'?t|stop|not now|cancel(?! (?:my |the )?order))\b", re.I)
SMALLTALK_RE = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|good (?:morning|evening))\b", re.I)

# Canonical execution order when one message has several intents.
INTENT_ORDER = ["trade_in", "product_info", "compare", "recommend", "stock_price", "policy",
                "cart", "checkout", "confirm", "abort", "order_status", "cancel_order"]


def parse_amount(num: str, unit: str | None) -> int:
    value = float(num.replace(",", ""))
    unit = (unit or "").lower().strip()
    if unit == "k":
        value *= 1_000
    elif unit in {"lakh", "lakhs", "l"}:
        value *= 100_000
    return int(value)


def extract_entities(text: str) -> dict[str, Any]:
    t = text.lower()
    ent: dict[str, Any] = {}

    if m := re.search(r"\bord-?(\d{4})\b", t):
        ent["order_id"] = f"ORD-{m.group(1)}"

    m = re.search(r"(?:under|below|within|budget(?: of| is)?|less than|up\s?to|max(?:imum)?)\s*"
                  r"(?:rs\.?|₹|inr)?\s*([\d][\d,.]*)\s*(k\b|lakhs?\b|l\b)?", t)
    if m:
        ent["budget_inr"] = parse_amount(m.group(1), m.group(2))
        t = t.replace(m.group(0), " ")

    # Pull the trade-in device out first so "trade in my 15 pro for a 17 pro" resolves both.
    m = re.search(r"(?:trade[\s-]?in|exchange|old)\s+(?:my\s+)?(?:old\s+)?(?:apple\s+)?"
                  r"((?:iphone\s*)?1[1-5](?:\s*pro\s*max|\s*pro|\s*plus|\s*mini)?)\b", t)
    if m:
        model = " ".join(m.group(1).split())
        ent["trade_in_model"] = model if model.startswith("iphone") else f"iphone {model}"
        t = t.replace(m.group(0), " ")
    if m := re.search(r"\b(excellent|good|fair|cracked|broken)\b", t):
        ent["condition"] = "cracked" if m.group(1) == "broken" else m.group(1)

    if m := re.search(r"\b(\d{3,4})\s*gb\b|\b([12])\s*tb\b", t):
        ent["storage_gb"] = int(m.group(1)) if m.group(1) else int(m.group(2)) * 1024
        t = t.replace(m.group(0), " ")

    skus: list[tuple[int, str]] = []
    for sku, pattern in SKU_PATTERNS:
        for match in re.finditer(pattern, t):
            skus.append((match.start(), sku))
        t = re.sub(pattern, " ", t)
    if skus:
        ent["skus"] = list(dict.fromkeys(s for _, s in sorted(skus)))

    for key, city in CITIES.items():
        if re.search(rf"\b{key}\b", t):
            ent["city"] = city
            break

    if m := re.search(r"\b(upi|card|credit card|debit card|cod|cash on delivery|emi|net ?banking)\b", t):
        ent["payment_method"] = {"credit card": "card", "debit card": "card", "cash on delivery": "cod",
                                 "net banking": "netbanking", "netbanking": "netbanking"}.get(m.group(1), m.group(1))

    tags = [tag for tag, words in USE_CASE_TAGS.items() if any(w in text.lower() for w in words)]
    if tags:
        ent["use_cases"] = tags
    return ent


def detect_intents(text: str, entities: dict[str, Any], pending_action: dict | None = None) -> list[str]:
    t = text.lower()
    if pending_action:
        if CONFIRM_RE.search(t):
            return ["confirm"]
        if ABORT_RE.search(t):
            return ["abort"]

    found = {intent for intent, pats in INTENT_KEYWORDS.items() if any(re.search(p, t) for p in pats)}

    if "cancel_order" in found:
        found.discard("order_status")
    if entities.get("order_id") and not found & {"cancel_order"}:
        found.add("order_status")
    if "trade_in" in found and not re.search(r"how does|eligib|policy|rules", t):
        found.discard("policy")
    if "checkout" in found and not re.search(r"\badd\b", t):
        found.discard("cart")
    if len(entities.get("skus", [])) >= 2 and not found & {"cart", "checkout"}:
        found.add("compare")
    if "compare" in found:
        found.discard("recommend")
    if "budget_inr" in entities and not entities.get("skus"):
        found.add("recommend")
    if entities.get("use_cases") and not entities.get("skus") and not found - {"stock_price"}:
        found.add("recommend")
    if entities.get("skus") and not found:
        found.add("product_info")
    if not found and SMALLTALK_RE.search(t):
        return ["smalltalk"]
    return [i for i in INTENT_ORDER if i in found]

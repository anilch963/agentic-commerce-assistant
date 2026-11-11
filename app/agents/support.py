"""Support agent: answers policy questions with RAG and cites every source.

It answers only from retrieved text. If nothing scores above the threshold it
says so instead of guessing.
"""
from __future__ import annotations

import re

from app.agents.base import AgentContext, agent
from app.agents.state import Finding
from app.llm import complete
from app.nlu import INTENT_KEYWORDS

ANSWER_PROMPT = """Answer the customer's question using ONLY the policy excerpts given.
Cite each fact with its source in square brackets, e.g. [returns.md#Return window].
If the excerpts do not answer it, say you don't have that information. Two to four sentences."""


STOPWORDS = {"the", "and", "what", "how", "can", "for", "with", "are", "you", "your", "iphone", "iphones", "our",
             "does", "this", "that", "will", "have", "about", "within"}


def _stems(text: str) -> set[str]:
    words = re.findall(r"[a-z]{3,}", text.lower())
    return {re.sub(r"(ing|ed|es|s)$", "", w) or w for w in words if w not in STOPWORDS}


def _extractive_answer(question: str, hits: list[dict], max_sentences: int = 2) -> str:
    """No-LLM fallback: return the sentences that best overlap the question, each with its citation.

    Only hits scoring within 60% of the best hit are considered, so weakly related
    sections don't leak into the answer.
    """
    q = _stems(question)
    top = hits[0]["score"]
    scored = []
    for rank, h in enumerate(hits):
        if h["score"] < 0.6 * top:
            continue
        body = h["text"].split(". ", 2)[-1]  # drop the "Doc. Heading." prefix
        for pos, sent in enumerate(re.split(r"(?<=\.)\s+", body)):
            overlap = len(q & _stems(sent))
            if overlap:
                scored.append((overlap + h["score"], -rank, -pos, sent.strip(), h["source"]))
    if not scored:  # fall back to the first sentence of the best section
        h = hits[0]
        return f"{h['text'].split('. ', 2)[-1].split('. ')[0].rstrip('.')}. [{h['source']}]"
    best = sorted(scored, reverse=True)[:max_sentences]
    best.sort(key=lambda x: (x[1], x[2]), reverse=True)  # keep document order for readability
    return " ".join(f"{s} [{src}]" for *_, s, src in best)


def policy_query(message: str) -> str:
    """Keep only the clauses that are about policy, so "price of the 17 in Mumbai and the EMI options"
    searches for "the EMI options" instead of also pulling in delivery-to-Mumbai sections."""
    clauses = [c.strip() for c in re.split(r"[?;,.]|\band\b|\balso\b", message, flags=re.I) if c.strip()]
    relevant = [c for c in clauses if any(re.search(p, c.lower()) for p in INTENT_KEYWORDS["policy"])]
    return " ".join(relevant) or message


@agent("support")
async def support(ctx: AgentContext) -> tuple[Finding, dict]:
    question = policy_query(ctx.state["user_input"])
    result = await ctx.tool("support", "search_policies", query=question)
    hits = result["hits"]
    if not hits:
        return Finding(text="I couldn't find that in our store policies. I can connect you to a human agent.",
                       data=result, citations=[]), {}
    excerpts = "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits)
    answer = await complete(ANSWER_PROMPT, f"Excerpts:\n{excerpts}\n\nQuestion: {question}", max_tokens=300)
    text = answer.strip() if answer else _extractive_answer(question, hits)
    cited = [h["source"] for h in hits if f"[{h['source']}]" in text]
    return Finding(text=text, data=result, citations=list(dict.fromkeys(cited))), {}

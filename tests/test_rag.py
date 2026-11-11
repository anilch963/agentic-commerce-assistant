"""Retrieval evaluation: a small labelled question set with hit@1 and hit@3 thresholds.

Add a row here whenever a real customer question retrieves the wrong section.
"""
import pytest

from app.agents.support import _extractive_answer
from app.mcp_servers.support_server import POLICY_DIR
from app.rag.retriever import TfidfRetriever, chunk_markdown

EVAL_SET = [
    ("can I return an opened iphone", "returns.md#Return window"),
    ("how long until I get my refund", "returns.md#How refunds work"),
    ("my phone arrived with a broken box and damage", "returns.md#Damaged on arrival"),
    ("does the warranty cover a cracked screen", "warranty.md#Standard warranty"),
    ("how much is screen repair with applecare", "warranty.md#AppleCare+"),
    ("is battery replacement free", "warranty.md#Battery service"),
    ("how fast is delivery to mumbai", "shipping_payments.md#Delivery times"),
    ("do you take cash on delivery", "shipping_payments.md#Payment options"),
    ("is there no cost emi", "shipping_payments.md#No-cost EMI"),
    ("can I cancel before it ships", "shipping_payments.md#Order cancellation"),
    ("which phones can I trade in", "trade_in.md#Eligibility"),
    ("how is my trade in value calculated", "trade_in.md#How the value is calculated"),
]


@pytest.fixture(scope="module")
def retriever():
    return TfidfRetriever.from_dir(POLICY_DIR)


def test_chunking_splits_on_sections():
    chunks = chunk_markdown(POLICY_DIR / "returns.md")
    assert [c.source for c in chunks] == ["returns.md#Return window", "returns.md#How refunds work",
                                          "returns.md#Damaged on arrival"]


def test_retrieval_quality(retriever):
    hit1 = hit3 = 0
    misses = []
    for question, expected in EVAL_SET:
        sources = [h.chunk.source for h in retriever.search(question, top_k=3)]
        hit1 += sources[:1] == [expected]
        hit3 += expected in sources
        if sources[:1] != [expected]:
            misses.append((question, sources[:1]))
    n = len(EVAL_SET)
    assert hit3 / n == 1.0, misses
    assert hit1 / n >= 0.9, misses


def test_unrelated_query_returns_nothing(retriever):
    assert retriever.search("recipe for chocolate cake", min_score=0.08) == []


def test_extractive_answer_cites_and_stays_on_topic(retriever):
    hits = [{"source": h.chunk.source, "text": h.chunk.text, "score": h.score}
            for h in retriever.search("can I return an opened iphone", top_k=3)]
    answer = _extractive_answer("can I return an opened iphone", hits)
    assert "7 days" in answer and "[returns.md#Return window]" in answer
    assert "trade" not in answer.lower()

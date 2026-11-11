"""Retrieval over store policy documents.

Documents are split on `##` headings so each chunk is one self-contained policy
section, and every hit carries a citation like `returns.md#Return window`.

The default index is TF-IDF over stemmed unigrams + bigrams (sublinear tf). It needs no network
or model download, which keeps tests and CI deterministic. To use dense
embeddings, implement `Retriever.search` with your vector store (Chroma,
pgvector, ...) and pass it to `build_support_server`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Chunk:
    source: str      # e.g. "returns.md#Return window"
    title: str
    text: str


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


SUFFIX_RE = re.compile(r"(ations?|ation|ments?|ing|ed|es|s)$")


# Domain synonyms applied after stemming: customers say "phone", docs say "iPhone".
SYNONYMS = {"phon": "iphon", "mobil": "iphon", "handset": "iphon", "money": "refund"}


def stem(word: str) -> str:
    """Tiny suffix stripper so "refund/refunds", "cancel/cancelled/cancellation", "ship/shipped" match.
    Good enough for short policy docs; swap for a real stemmer or embeddings at scale."""
    stemmed = SUFFIX_RE.sub("", word)
    if stemmed != word and len(stemmed) > 3 and stemmed[-1] == stemmed[-2] and stemmed[-1] not in "aeiou":
        stemmed = stemmed[:-1]  # shipp -> ship, cancell -> cancel
    if len(stemmed) < 3:
        stemmed = word
    return SYNONYMS.get(stemmed, stemmed)


def analyze(text: str) -> list[str]:
    tokens = [stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in ENGLISH_STOP_WORDS]
    return tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:], strict=False)]


class Retriever(Protocol):
    def search(self, query: str, top_k: int = 3, min_score: float = 0.0) -> list[Hit]: ...


def chunk_markdown(path: Path) -> list[Chunk]:
    """Split a markdown file into one chunk per `##` section."""
    text = path.read_text(encoding="utf-8")
    doc_title = next((line[2:].strip() for line in text.splitlines() if line.startswith("# ")), path.stem)
    chunks: list[Chunk] = []
    for section in re.split(r"^## ", text, flags=re.MULTILINE)[1:]:
        heading, _, body = section.partition("\n")
        body = " ".join(body.split())
        if body:
            chunks.append(Chunk(
                source=f"{path.name}#{heading.strip()}",
                title=f"{doc_title} - {heading.strip()}",
                # Prepend headings so short sections still match on topic words.
                text=f"{doc_title}. {heading.strip()}. {body}",
            ))
    return chunks


class TfidfRetriever:
    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("No chunks to index")
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(analyzer=analyze, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(c.text for c in chunks)

    @classmethod
    def from_dir(cls, directory: Path) -> TfidfRetriever:
        chunks = [c for p in sorted(directory.glob("*.md")) for c in chunk_markdown(p)]
        return cls(chunks)

    def search(self, query: str, top_k: int = 3, min_score: float = 0.0) -> list[Hit]:
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        ranked = sorted(zip(self.chunks, scores, strict=False), key=lambda x: x[1], reverse=True)
        return [Hit(c, round(float(s), 4)) for c, s in ranked[:top_k] if s >= min_score]

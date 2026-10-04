"""Retrievers. Each takes a list of Chunk and exposes search(query, k) -> [(Chunk, score)]."""
import re

from rank_bm25 import BM25Okapi

STOPWORDS = set("a an and are as at be by can do does for from how i if in is it me my of on or the to was what when which who will with you your".split())


def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.index_text) for c in chunks])

    def search(self, query, k=5):
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], float(scores[i])) for i in ranked]


def build_retriever(name, chunks):
    registry = {"bm25": BM25Retriever}
    if name not in registry:
        raise ValueError(f"Unknown retriever '{name}'. Available: {sorted(registry)}")
    return registry[name](chunks)

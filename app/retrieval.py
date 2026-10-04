"""Retrievers. Each takes a list of Chunk and exposes search(query, k) -> [(Chunk, score)].

bm25    keyword matching: strong when the question reuses the document's words
dense   embedding similarity: strong when the question paraphrases the document
hybrid  both rankings merged with Reciprocal Rank Fusion
"""
import re

import numpy as np
from rank_bm25 import BM25Okapi

STOPWORDS = set("a an and are as at be by can do does for from how i if in is it me my of on or the to was what when which who will with you your".split())


def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.index_text) for c in chunks])

    def scores(self, query):
        return np.asarray(self.bm25.get_scores(tokenize(query)), dtype=float)

    def search(self, query, k=5):
        scores = self.scores(query)
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self.chunks[i], float(scores[i])) for i in order]


class DenseRetriever:
    name = "dense"

    def __init__(self, chunks, embedder):
        self.chunks = chunks
        self.embedder = embedder
        self.matrix = embedder.embed_passages([c.index_text for c in chunks])  # (n_chunks, dim), unit length

    def scores(self, query):
        return self.matrix @ self.embedder.embed_query(query)  # cosine similarity

    def search(self, query, k=5):
        scores = self.scores(query)
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self.chunks[i], float(scores[i])) for i in order]


def rrf_fuse(rankings, weights=None, k=60):
    """Reciprocal Rank Fusion.

    rankings: list of lists of item indices, best first. Each item scores sum(weight / (k + rank)).
    Using ranks instead of raw scores avoids having to normalize BM25 scores against cosine scores.
    """
    weights = weights or [1.0] * len(rankings)
    fused = {}
    for weight, ranking in zip(weights, rankings):
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + weight / (k + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


class HybridRetriever:
    name = "hybrid"

    def __init__(self, chunks, embedder, bm25_weight=1.0, dense_weight=1.0, rrf_k=60):
        self.chunks = chunks
        self.bm25 = BM25Retriever(chunks)
        self.dense = DenseRetriever(chunks, embedder)
        self.weights = [bm25_weight, dense_weight]
        self.rrf_k = rrf_k

    def search(self, query, k=5):
        rankings = [list(np.argsort(-r.scores(query), kind="stable")) for r in (self.bm25, self.dense)]
        fused = rrf_fuse(rankings, self.weights, self.rrf_k)[:k]
        return [(self.chunks[i], score) for i, score in fused]


def build_retriever(name, chunks, embedder=None, **options):
    if name == "bm25":
        return BM25Retriever(chunks)
    if name in ("dense", "hybrid"):
        if embedder is None:
            from .embeddings import get_embedder  # only load the model when it is actually needed
            embedder = get_embedder()
        if name == "dense":
            return DenseRetriever(chunks, embedder)
        return HybridRetriever(chunks, embedder, **options)
    raise ValueError(f"Unknown retriever '{name}'. Available: ['bm25', 'dense', 'hybrid']")

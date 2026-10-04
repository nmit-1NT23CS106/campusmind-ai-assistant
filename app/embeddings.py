"""Text embedders.

An embedder turns text into a unit-length vector so that cosine similarity (a dot product)
measures closeness in meaning. Retrievers depend only on this small interface, so the model
can be swapped without touching retrieval code, and tests can use a fake embedder.
"""
import hashlib
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"  # 384 dimensions, ~67 MB, runs on CPU
INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "index"


def _unit(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


class FastEmbedEmbedder:
    """ONNX-based embeddings via fastembed (no PyTorch needed). Downloads the model on first use."""

    def __init__(self, model_name=DEFAULT_MODEL, cache_dir=None):
        from fastembed import TextEmbedding  # imported lazily so BM25-only use needs no model

        self.name = model_name
        cache = Path(cache_dir) if cache_dir else INDEX_DIR / "models"
        cache.mkdir(parents=True, exist_ok=True)
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(cache))

    def embed_passages(self, texts):
        return _unit(list(self._model.passage_embed(list(texts))))

    def embed_query(self, text):
        # bge models expect a query instruction prefix; fastembed's query_embed adds it for us.
        return _unit(next(iter(self._model.query_embed(text))))


class CachedEmbedder:
    """Wraps an embedder and stores passage vectors on disk, keyed by model + exact text."""

    def __init__(self, inner, cache_dir=INDEX_DIR):
        self.inner = inner
        self.name = inner.name
        self.cache_dir = Path(cache_dir)

    def _path(self, texts):
        digest = hashlib.sha256((self.name + "\x00" + "\x00".join(texts)).encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / f"emb_{digest}.npy"

    def embed_passages(self, texts):
        texts = list(texts)
        path = self._path(texts)
        if path.exists():
            return np.load(path)
        vectors = self.inner.embed_passages(texts)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors)
        return vectors

    def embed_query(self, text):
        return self.inner.embed_query(text)


def get_embedder(model_name=DEFAULT_MODEL):
    return CachedEmbedder(FastEmbedEmbedder(model_name))

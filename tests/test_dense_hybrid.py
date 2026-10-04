import os
import subprocess
import sys
import zlib

import numpy as np
import pytest

from app.chunking import load_chunks
from app.embeddings import CachedEmbedder
from app.retrieval import BM25Retriever, DenseRetriever, HybridRetriever, build_retriever, rrf_fuse, tokenize
from eval.run_eval import run
from tests.conftest import DOCS, ROOT


class FakeEmbedder:
    """Deterministic hashed bag-of-words vectors. Lexical, but a genuine unit-length embedder,
    so it exercises the same code paths as a real model without downloading one."""
    name = "fake"
    dim = 128

    def _vec(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            v[zlib.crc32(tok.encode()) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_passages(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture(scope="module")
def chunks():
    return load_chunks(DOCS)


def test_dense_scores_are_cosine_and_sorted(chunks):
    r = DenseRetriever(chunks, FakeEmbedder())
    assert np.allclose(np.linalg.norm(r.matrix, axis=1), 1.0, atol=1e-5)
    hits = r.search("minimum attendance for exam eligibility", k=5)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True) and all(-1.0001 <= s <= 1.0001 for s in scores)
    assert hits[0][0].doc == "attendance_policy"


def test_rrf_item_in_both_lists_beats_item_in_one():
    fused = dict(rrf_fuse([[0, 1, 2], [1, 3, 0]], k=60))
    assert fused[1] > fused[3] and fused[0] > fused[2]


def test_rrf_exact_math():
    fused = dict(rrf_fuse([[7], [9, 7]], k=10))
    assert fused[7] == pytest.approx(1 / 11 + 1 / 12)
    assert fused[9] == pytest.approx(1 / 11)


def test_rrf_weights_shift_the_winner():
    rankings = [[0, 1], [1, 0]]
    assert rrf_fuse(rankings, weights=[2.0, 1.0])[0][0] == 0
    assert rrf_fuse(rankings, weights=[1.0, 2.0])[0][0] == 1


def test_rrf_ties_are_deterministic():
    assert [i for i, _ in rrf_fuse([[0, 1], [1, 0]])] == [0, 1]


def test_hybrid_returns_k_unique_chunks(chunks):
    r = HybridRetriever(chunks, FakeEmbedder())
    hits = r.search("how much does a revaluation cost", k=5)
    ids = [c.id for c, _ in hits]
    assert len(ids) == 5 == len(set(ids))
    assert hits[0][0].doc == "exam_regulations"


def test_hybrid_zero_dense_weight_matches_bm25_order(chunks):
    q = "late fee for tuition payment"
    hybrid = HybridRetriever(chunks, FakeEmbedder(), bm25_weight=1.0, dense_weight=0.0)
    bm25 = BM25Retriever(chunks)
    assert [c.id for c, _ in hybrid.search(q, 3)] == [c.id for c, _ in bm25.search(q, 3)]


def test_build_retriever_wires_each_name(chunks):
    assert build_retriever("bm25", chunks).name == "bm25"
    assert build_retriever("dense", chunks, embedder=FakeEmbedder()).name == "dense"
    assert build_retriever("hybrid", chunks, embedder=FakeEmbedder(), rrf_k=30).rrf_k == 30


def test_cached_embedder_calls_model_once_per_unique_corpus(tmp_path):
    class Counting(FakeEmbedder):
        calls = 0

        def embed_passages(self, texts):
            Counting.calls += 1
            return super().embed_passages(texts)

    cached = CachedEmbedder(Counting(), cache_dir=tmp_path)
    first = cached.embed_passages(["alpha beta", "gamma"])
    second = cached.embed_passages(["alpha beta", "gamma"])
    assert Counting.calls == 1 and np.array_equal(first, second)
    cached.embed_passages(["alpha beta", "different"])
    assert Counting.calls == 2


def test_cache_key_depends_on_model_name(tmp_path):
    a, b = FakeEmbedder(), FakeEmbedder()
    b.name = "other-model"
    texts = ["x y"]
    assert CachedEmbedder(a, tmp_path)._path(texts) != CachedEmbedder(b, tmp_path)._path(texts)


def test_importing_retrieval_does_not_load_the_embedding_library():
    code = "import sys, app.retrieval; sys.exit(1 if 'fastembed' in sys.modules else 0)"
    assert subprocess.run([sys.executable, "-c", code], cwd=ROOT).returncode == 0


def test_eval_plumbing_with_dense_and_hybrid():
    for name in ("dense", "hybrid"):
        result = run(name, embedder=FakeEmbedder())
        assert result["embedding_model"] == "fake" and result["coverage_ceiling"] == 1.0
        assert all(0 <= result[m] <= 1 for m in ("hit@1", "hit@3", "hit@5", "mrr"))
        assert result["hit@1"] <= result["hit@3"] <= result["hit@5"]


@pytest.mark.skipif(os.environ.get("RUN_EMBEDDING_TESTS") != "1",
                    reason="downloads the real model; run with RUN_EMBEDDING_TESTS=1")
def test_real_model_shapes_and_basic_semantics():
    from app.embeddings import FastEmbedEmbedder

    e = FastEmbedEmbedder()
    vecs = e.embed_passages(["attendance requirement for exams", "minimum presence needed to sit the test",
                             "library fine for overdue books"])
    assert vecs.shape == (3, 384) and np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-4)
    assert float(vecs[0] @ vecs[1]) > float(vecs[0] @ vecs[2])
    assert e.embed_query("how many classes must I attend").shape == (384,)

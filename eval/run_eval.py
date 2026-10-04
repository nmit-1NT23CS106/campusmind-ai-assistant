"""Measure retrieval quality on eval/questions.json.

A question is a HIT at rank r when the r-th retrieved chunk comes from the gold document
AND contains the gold phrase (the sentence that actually answers the question).
Because matching is on text, not chunk ids, the same questions work for any chunking strategy.

Usage:
  python -m eval.run_eval --retriever bm25 --strategy heading
  python -m eval.run_eval --retriever hybrid
  python -m eval.run_eval --all            # bm25 vs dense vs hybrid, writes eval/results/comparison.md
"""
import argparse
import json
import re
import time
from pathlib import Path

from app.chunking import load_chunks
from app.retrieval import build_retriever

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "data" / "docs"
QUESTIONS = ROOT / "eval" / "questions.json"
RESULTS = ROOT / "eval" / "results"


def norm(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def is_hit(chunk, question):
    return chunk.doc == question["gold_doc"] and norm(question["gold_phrase"]) in norm(chunk.text)


def first_hit_rank(ranked_chunks, question):
    for rank, chunk in enumerate(ranked_chunks, 1):
        if is_hit(chunk, question):
            return rank
    return None


def summarize(ranks, ks=(1, 3, 5)):
    n = len(ranks)
    out = {f"hit@{k}": sum(1 for r in ranks if r is not None and r <= k) / n for k in ks}
    out["mrr"] = sum(1 / r for r in ranks if r is not None) / n
    return out


def run(retriever_name="bm25", strategy="heading", k=5, embedder=None, retriever_options=None, **chunk_kwargs):
    chunks = load_chunks(DOCS, strategy=strategy, **chunk_kwargs)
    t0 = time.perf_counter()
    retriever = build_retriever(retriever_name, chunks, embedder=embedder, **(retriever_options or {}))
    build_seconds = time.perf_counter() - t0
    questions = [q for q in json.loads(QUESTIONS.read_text()) if q["answerable"]]

    # ceiling: how many gold phrases survive intact in at least one chunk (chunking can split them)
    coverage = sum(any(is_hit(c, q) for c in chunks) for q in questions) / len(questions)

    ranks, misses, latencies = [], [], []
    for q in questions:
        start = time.perf_counter()
        ranked = [c for c, _ in retriever.search(q["question"], k=k)]
        latencies.append((time.perf_counter() - start) * 1000)
        rank = first_hit_rank(ranked, q)
        ranks.append(rank)
        if rank is None or rank > 3:
            misses.append({"id": q["id"], "question": q["question"], "rank": rank,
                           "top1": ranked[0].citation if ranked else None})
    return {
        "retriever": retriever_name, "strategy": strategy, "chunk_kwargs": chunk_kwargs,
        "retriever_options": retriever_options or {},
        "embedding_model": getattr(embedder, "name", None) if retriever_name != "bm25" else None,
        "n_questions": len(questions), "n_chunks": len(chunks), "coverage_ceiling": round(coverage, 3),
        **{m: round(v, 3) for m, v in summarize(ranks).items()},
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
        "index_build_s": round(build_seconds, 2), "misses": misses,
    }


def describe(result):
    return (f"{result['retriever']} | {result['strategy']} {result['chunk_kwargs']} | "
            f"{result['n_chunks']} chunks | {result['n_questions']} questions\n"
            f"  coverage ceiling {result['coverage_ceiling']:.0%}  hit@1 {result['hit@1']:.0%}  "
            f"hit@3 {result['hit@3']:.0%}  hit@5 {result['hit@5']:.0%}  MRR {result['mrr']:.3f}  "
            f"{result['avg_latency_ms']} ms/query")


def save(result):
    RESULTS.mkdir(exist_ok=True)
    opts = {**result["chunk_kwargs"], **result["retriever_options"]}
    tag = f"{result['retriever']}_{result['strategy']}_" + "_".join(f"{k}{v}" for k, v in opts.items())
    (RESULTS / f"{tag}.json").write_text(json.dumps(result, indent=2))


def comparison_table(results):
    lines = ["| Retriever | hit@1 | hit@3 | hit@5 | MRR | ms/query |", "|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['retriever']} | {r['hit@1']:.0%} | {r['hit@3']:.0%} | {r['hit@5']:.0%} | "
                     f"{r['mrr']:.3f} | {r['avg_latency_ms']} |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--retriever", default="bm25", choices=["bm25", "dense", "hybrid"])
    p.add_argument("--strategy", default="heading", choices=["heading", "fixed"])
    p.add_argument("--max-words", type=int, default=None)
    p.add_argument("--overlap", type=int, default=None)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--bm25-weight", type=float, default=1.0)
    p.add_argument("--dense-weight", type=float, default=1.0)
    p.add_argument("--all", action="store_true", help="compare bm25, dense and hybrid")
    a = p.parse_args()
    kwargs = {}
    if a.max_words is not None:
        kwargs["max_words"] = a.max_words
    if a.overlap is not None:
        kwargs["overlap_words"] = a.overlap

    names = ["bm25", "dense", "hybrid"] if a.all else [a.retriever]
    embedder = None
    if any(n != "bm25" for n in names):
        from app.embeddings import get_embedder
        print("Loading embedding model (the first run downloads about 67 MB)...")
        embedder = get_embedder()

    results = []
    for name in names:
        opts = {"bm25_weight": a.bm25_weight, "dense_weight": a.dense_weight} if name == "hybrid" else {}
        result = run(name, a.strategy, a.k, embedder=embedder, retriever_options=opts, **kwargs)
        print(describe(result))
        for m in result["misses"]:
            print(f"  miss {m['id']} (rank {m['rank']}): {m['question']}\n       top1: {m['top1']}")
        save(result)
        results.append(result)

    if a.all:
        table = comparison_table(results)
        print("\n" + table)
        (RESULTS / "comparison.md").write_text(table + "\n")


if __name__ == "__main__":
    main()

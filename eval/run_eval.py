"""Measure retrieval quality on eval/questions.json.

A question is a HIT at rank r when the r-th retrieved chunk comes from the gold document
AND contains the gold phrase (the sentence that actually answers the question).
Because matching is on text, not chunk ids, the same questions work for any chunking strategy.

Usage: python -m eval.run_eval --retriever bm25 --strategy heading --max-words 120
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


def run(retriever_name="bm25", strategy="heading", k=5, **chunk_kwargs):
    chunks = load_chunks(DOCS, strategy=strategy, **chunk_kwargs)
    retriever = build_retriever(retriever_name, chunks)
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
        "n_questions": len(questions), "n_chunks": len(chunks), "coverage_ceiling": round(coverage, 3),
        **{m: round(v, 3) for m, v in summarize(ranks).items()},
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2), "misses": misses,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--retriever", default="bm25")
    p.add_argument("--strategy", default="heading", choices=["heading", "fixed"])
    p.add_argument("--max-words", type=int, default=None)
    p.add_argument("--overlap", type=int, default=None)
    p.add_argument("--k", type=int, default=5)
    a = p.parse_args()
    kwargs = {}
    if a.max_words is not None:
        kwargs["max_words"] = a.max_words
    if a.overlap is not None:
        kwargs["overlap_words"] = a.overlap
    result = run(a.retriever, a.strategy, a.k, **kwargs)

    print(f"{result['retriever']} | {result['strategy']} {result['chunk_kwargs']} | "
          f"{result['n_chunks']} chunks | {result['n_questions']} questions")
    print(f"  coverage ceiling {result['coverage_ceiling']:.0%}  hit@1 {result['hit@1']:.0%}  "
          f"hit@3 {result['hit@3']:.0%}  hit@5 {result['hit@5']:.0%}  MRR {result['mrr']:.3f}  "
          f"{result['avg_latency_ms']} ms/query")
    for m in result["misses"]:
        print(f"  miss {m['id']} (rank {m['rank']}): {m['question']}\n       top1: {m['top1']}")

    RESULTS.mkdir(exist_ok=True)
    tag = f"{result['retriever']}_{result['strategy']}_" + "_".join(f"{k}{v}" for k, v in result["chunk_kwargs"].items())
    (RESULTS / f"{tag}.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

# CampusMind: AI Academic Assistant

RAG over university policy documents plus a guarded text-to-SQL tool, routed by an LLM agent.
**Status: Step 1-3 of 8 done** (documents, data, chunking, keyword baseline, evaluation harness).

## Run it
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m data.seed                    # creates data/campus.db (200 fake students)
python -m eval.run_eval --retriever bm25 --strategy heading
pytest
```

## What is here
| Path | Purpose |
|---|---|
| `data/docs/` | 10 policy and syllabus documents (fictional "Northfield University") |
| `data/seed.py` | Reproducible synthetic database: students, courses, enrollments, per-class attendance, assessments, plus an `enrollment_summary` view |
| `app/chunking.py` | Heading-aware and fixed-window chunkers; chunks keep document + heading path for citations |
| `app/retrieval.py` | BM25 retriever behind a common `search(query, k)` interface (dense and hybrid retrievers plug in next) |
| `eval/questions.json` | 39 questions: 33 answerable (each with a gold document and gold phrase) and 6 unanswerable |
| `eval/run_eval.py` | Computes hit@1/3/5 and MRR; saves results to `eval/results/` |

## How retrieval is measured
A question counts as a hit at rank r when the r-th chunk comes from the gold document **and contains the gold phrase** (the sentence that answers it).
Matching on text instead of chunk ids means the same questions work for any chunking strategy.
A "coverage ceiling" reports how many gold phrases survive chunking intact, so chunking can never silently cap the score.

## Results so far (33 answerable questions)
| Retriever | Chunking | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|---|
| BM25 | heading, max 120 words | 67% | 79% | 85% | 0.739 |
| BM25 | heading, max 60 words | 67% | 79% | 85% | 0.739 |
| BM25 | fixed 40 words, overlap 10 | 64% | 79% | 82% | 0.703 |
| BM25 | fixed 80 words, overlap 20 | 58% | 79% | 82% | 0.684 |

Misses are mostly vocabulary mismatches ("copying" vs "unfair means", "rechecked" vs "revaluation"), which is what dense embeddings and hybrid search should fix.

## Honest limitations
- The corpus is small (about 45 chunks) and written by the author, so absolute numbers are optimistic. What matters is the *change* between methods.
- Sections are short, so chunk size barely matters here. Longer documents would be needed to tune it meaningfully.
- All data is synthetic.

## Roadmap
1. ~~Documents and fake database~~
2. ~~Chunking and BM25 baseline~~
3. ~~Evaluation set and metrics~~
4. Dense embeddings + hybrid search (re-measure, record the gain)
5. Guarded text-to-SQL tool (read-only connection, allowlist, validator, row limit)
6. LLM agent with tool calling (RAG vs SQL vs both vs refuse)
7. FastAPI + Streamlit chat UI, logging of latency and token cost
8. Docker, CI, deployment

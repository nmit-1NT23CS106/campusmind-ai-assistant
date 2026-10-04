import json

from app.chunking import Chunk, load_chunks
from app.retrieval import BM25Retriever, build_retriever, tokenize
from eval.run_eval import QUESTIONS, first_hit_rank, norm, run, summarize
from tests.conftest import DOCS

import pytest

ANSWERABLE = [q for q in json.loads(QUESTIONS.read_text()) if q["answerable"]]


@pytest.mark.parametrize("q", ANSWERABLE, ids=lambda q: q["id"])
def test_every_gold_phrase_exists_in_exactly_its_gold_document(q):
    """Guards the eval set: editing a document can't silently invalidate a question."""
    holders = [p.stem for p in DOCS.glob("*.md") if norm(q["gold_phrase"]) in norm(p.read_text())]
    assert holders == [q["gold_doc"]]


def test_unanswerable_questions_have_no_gold():
    for q in json.loads(QUESTIONS.read_text()):
        if not q["answerable"]:
            assert q["gold_doc"] is None and q["gold_phrase"] is None


def test_question_ids_unique_and_enough_questions():
    qs = json.loads(QUESTIONS.read_text())
    assert len({q["id"] for q in qs}) == len(qs) >= 30


def test_metric_math():
    ranks = [1, 3, None, 5]
    s = summarize(ranks)
    assert s["hit@1"] == 0.25 and s["hit@3"] == 0.5 and s["hit@5"] == 0.75
    assert abs(s["mrr"] - (1 + 1 / 3 + 0 + 1 / 5) / 4) < 1e-9


def test_first_hit_rank_requires_right_doc_and_phrase():
    q = {"gold_doc": "a", "gold_phrase": "Fix the Bug"}
    wrong_doc = Chunk("b#0", "b", "", "", "", "fix the bug")
    no_phrase = Chunk("a#0", "a", "", "", "", "something else")
    right = Chunk("a#1", "a", "", "", "", "Please  fix   the bug now")
    assert first_hit_rank([wrong_doc, no_phrase, right], q) == 3
    assert first_hit_rank([wrong_doc, no_phrase], q) is None


def test_tokenize_drops_stopwords_and_lowercases():
    assert tokenize("What is the Minimum Attendance?") == ["minimum", "attendance"]


def test_bm25_finds_attendance_policy_for_attendance_question():
    r = BM25Retriever(load_chunks(DOCS))
    top = r.search("minimum attendance for exam eligibility", k=1)[0][0]
    assert top.doc == "attendance_policy"


def test_unknown_retriever_is_a_clear_error():
    with pytest.raises(ValueError, match="Unknown retriever"):
        build_retriever("nope", [])


def test_baseline_quality_floor():
    """Regression guard: if chunking or tokenizing breaks, this drops sharply."""
    result = run("bm25", "heading")
    assert result["coverage_ceiling"] == 1.0
    assert result["hit@5"] >= 0.8

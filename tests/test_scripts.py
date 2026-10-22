import json
import os
import time

from scripts import resume_bullets as rb
from scripts import update_readme as ur


def write(dirpath, name, data, age=0):
    path = dirpath / name
    path.write_text(json.dumps(data) if not isinstance(data, str) else data, encoding="utf-8")
    t = time.time() - age
    os.utime(path, (t, t))


def retr(name, h1, h3, n=33, **extra):
    return {"retriever": name, "strategy": "heading", "chunk_kwargs": {}, "retriever_options": {}, "n_questions": n,
            "hit@1": h1, "hit@3": h3, **extra}


def agent(errors=0, n=55, **cats):
    base = {"docs": {"n": 33, "pass_rate": 1.0}, "sql": {"n": 17, "pass_rate": 0.882}, "refuse": {"n": 6, "pass_rate": 1.0}}
    return {"n_questions": n + errors, "n_scored": n, "errors": errors, "by_category": {**base, **cats}}


# ---------------- update_readme ----------------
def test_readme_placeholders_are_replaced_and_rerun_updates_in_place(tmp_path):
    (tmp_path / "comparison.md").write_text("| table v1 |")
    text = "A\n<!-- paste eval/results/comparison.md here -->\nB\n<!-- paste eval/results/agent_eval.md here -->\n"
    new, updated, skipped = ur.update(text, tmp_path)
    assert "| table v1 |" in new and updated == ["retrieval"] and "agent" in skipped[0]
    (tmp_path / "comparison.md").write_text("| table v2 |")
    again, _, _ = ur.update(new, tmp_path)
    assert "| table v2 |" in again and "table v1" not in again and again.count("retrieval-results:start") == 1


def test_readme_untouched_when_no_results_exist(tmp_path):
    text = "x <!-- paste eval/results/comparison.md here --> y"
    new, updated, skipped = ur.update(text, tmp_path)
    assert new == text and updated == [] and len(skipped) == 2


def test_real_readme_still_has_both_markers():
    text = ur.README.read_text(encoding="utf-8")
    for _, placeholder in ur.SECTIONS.values():
        assert placeholder in text or "-results:start" in text


# ---------------- resume_bullets ----------------
def test_improvement_bullet_uses_only_measured_numbers(tmp_path):
    write(tmp_path, "a.json", retr("bm25", 0.67, 0.79))
    write(tmp_path, "b.json", retr("hybrid", 0.82, 0.94))
    bullets, warns = rb.build(tmp_path, tests=231)
    first = bullets[0]
    assert "79%" in first and "94%" in first and "67%" in first and "82%" in first and "33-question" in first
    assert not any("Retrieval" in w for w in warns)


def test_no_improvement_claim_when_hybrid_is_not_better(tmp_path):
    write(tmp_path, "a.json", retr("bm25", 0.67, 0.85))
    write(tmp_path, "b.json", retr("dense", 0.7, 0.8))
    write(tmp_path, "c.json", retr("hybrid", 0.7, 0.82))
    bullets, warns = rb.build(tmp_path)
    assert "rose" not in bullets[0] and "compared" in bullets[0]
    assert any("did not beat" in w for w in warns)


def test_missing_results_produce_instructions_not_numbers(tmp_path):
    bullets, warns = rb.build(tmp_path)
    assert len(bullets) == 1 and "five layers" in bullets[0]
    assert any("run_eval" in w for w in warns) and any("agent_eval" in w for w in warns)


def test_only_default_configuration_results_are_used(tmp_path):
    write(tmp_path, "good_bm25.json", retr("bm25", 0.67, 0.79), age=100)
    write(tmp_path, "weird_bm25.json", {**retr("bm25", 0.1, 0.1), "chunk_kwargs": {"max_words": 60}})
    write(tmp_path, "fixed_bm25.json", {**retr("bm25", 0.2, 0.2), "strategy": "fixed"})
    write(tmp_path, "junk.json", "not json at all")
    assert rb.load_retrieval(tmp_path)["bm25"]["hit@3"] == 0.79


def test_newest_result_wins(tmp_path):
    write(tmp_path, "old.json", retr("bm25", 0.5, 0.5), age=1000)
    write(tmp_path, "new.json", retr("bm25", 0.6, 0.7))
    assert rb.load_retrieval(tmp_path)["bm25"]["hit@3"] == 0.7


def test_agent_bullet_reports_each_category(tmp_path):
    write(tmp_path, "agent_eval.json", agent())
    bullet, warns = rb.agent_bullet(rb.load_agent(tmp_path))
    assert "55-question" in bullet and warns == []
    assert "33 of 33 (100%)" in bullet and "15 of 17 (88%)" in bullet and "6 of 6 (100%)" in bullet


def test_agent_warnings_for_errors_and_small_samples(tmp_path):
    write(tmp_path, "agent_eval.json", agent(errors=3, n=7))
    _, warns = rb.agent_bullet(rb.load_agent(tmp_path))
    assert any("provider errors" in w for w in warns) and any("only 7" in w for w in warns)


def test_safety_bullet_test_count_is_optional():
    assert "231+ automated tests" in rb.safety_bullet(231) and "automated tests including" in rb.safety_bullet(None)

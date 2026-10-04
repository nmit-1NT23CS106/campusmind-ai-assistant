from app.chunking import Chunk, chunk_fixed, chunk_heading, load_chunks, parse_sections
from tests.conftest import DOCS

MD = """# Title

## Alpha
First sentence here. Second sentence here.

### Sub
Sub text.

## Beta
Beta text.
"""


def test_headings_become_metadata():
    chunks = chunk_heading("doc", MD)
    assert [(c.section, c.subsection) for c in chunks] == [("Alpha", ""), ("Alpha", "Sub"), ("Beta", "")]
    assert all(c.title == "Title" and c.doc == "doc" for c in chunks)


def test_citation_and_index_text_include_headings():
    c = chunk_heading("doc", MD)[1]
    assert c.citation == "doc > Alpha > Sub"
    assert "Alpha" in c.index_text and "Sub text." in c.index_text


def test_long_section_is_split_without_exceeding_max_words():
    body = " ".join(f"Sentence number {i} has some words." for i in range(40))
    chunks = chunk_heading("d", f"# T\n\n## S\n{body}", max_words=30)
    assert len(chunks) > 1
    assert all(len(c.text.split()) <= 30 for c in chunks)


def test_no_text_is_lost_when_splitting():
    sents = [f"Fact {i} is important." for i in range(30)]
    chunks = chunk_heading("d", "# T\n\n## S\n" + " ".join(sents), max_words=25)
    joined = " ".join(c.text for c in chunks)
    assert all(s in joined for s in sents)


def test_overlap_repeats_a_short_tail():
    sents = [f"Fact {i} is here." for i in range(20)]
    chunks = chunk_heading("d", "# T\n\n## S\n" + " ".join(sents), max_words=12, overlap_words=4)
    assert chunks[0].text.split("\n")[-1] in chunks[1].text


def test_fixed_windows_overlap_and_ignore_headings():
    md = "# T\n\n## S\n" + " ".join(f"w{i}" for i in range(100))
    chunks = chunk_fixed("d", md, max_words=40, overlap_words=10)
    assert all(c.section == "" for c in chunks)
    assert chunks[0].text.split()[-10:] == chunks[1].text.split()[:10]


def test_ids_are_unique_across_the_real_corpus():
    chunks = load_chunks(DOCS)
    assert len(chunks) >= 40
    assert len({c.id for c in chunks}) == len(chunks)
    assert all(isinstance(c, Chunk) and c.text.strip() for c in chunks)


def test_empty_sections_are_skipped():
    assert list(parse_sections("# T\n\n## Empty\n\n## Full\ntext")) == [("T", "Full", "", "text")]

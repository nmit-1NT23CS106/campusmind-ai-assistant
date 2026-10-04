"""Split markdown documents into chunks for retrieval.

Two strategies are provided so retrieval quality can be compared:
  heading: split on markdown headings, then pack sentences up to max_words
  fixed:   ignore headings, slide a fixed word window with overlap
Every chunk keeps its document and heading path so answers can cite sources.
"""
import re
from dataclasses import dataclass
from pathlib import Path

HEADING = re.compile(r"^(#{1,3})\s+(.*\S)\s*$")
SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    id: str
    doc: str
    title: str
    section: str
    subsection: str
    text: str

    @property
    def index_text(self) -> str:
        """Text that gets indexed: heading context plus body (contextual headers help retrieval)."""
        return f"{self.title}. {self.section}. {self.subsection}. {self.text}".replace(". . ", ". ")

    @property
    def citation(self) -> str:
        parts = [self.doc, self.section] + ([self.subsection] if self.subsection else [])
        return " > ".join(p for p in parts if p)


def _words(text):
    return len(text.split())


def _units(paragraph, max_words):
    """Keep a paragraph whole if it fits, otherwise break it into sentences."""
    if _words(paragraph) <= max_words:
        return [paragraph]
    return [s for s in SENTENCE.split(paragraph) if s.strip()]


def _pack(units, max_words, overlap_words):
    """Greedily pack units into chunks of at most max_words; repeat a short tail as overlap."""
    chunks, current = [], []
    for unit in units:
        if current and _words(" ".join(current + [unit])) > max_words:
            chunks.append("\n".join(current))
            tail = current[-1]
            current = [tail] if overlap_words and _words(tail) <= overlap_words else []
        current.append(unit)
    if current:
        chunks.append("\n".join(current))
    return chunks


def parse_sections(markdown):
    """Yield (title, section, subsection, body) for each heading block that has text."""
    title = section = subsection = ""
    body = []

    def flush():
        text = "\n".join(body).strip()
        if text:
            yield title, section, subsection, text

    for line in markdown.splitlines():
        m = HEADING.match(line)
        if not m:
            body.append(line)
            continue
        yield from flush()
        body = []
        level, heading = len(m.group(1)), m.group(2)
        if level == 1:
            title, section, subsection = heading, "", ""
        elif level == 2:
            section, subsection = heading, ""
        else:
            subsection = heading
    yield from flush()


def chunk_heading(doc, markdown, max_words=120, overlap_words=0):
    out = []
    for title, section, subsection, body in parse_sections(markdown):
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        units = [u for p in paragraphs for u in _units(p, max_words)]
        for piece in _pack(units, max_words, overlap_words):
            out.append(Chunk(f"{doc}#{len(out)}", doc, title, section, subsection, piece))
    return out


def chunk_fixed(doc, markdown, max_words=60, overlap_words=15):
    lines = [l for l in markdown.splitlines() if not HEADING.match(l)]
    words = " ".join(lines).split()
    title = next((HEADING.match(l).group(2) for l in markdown.splitlines() if l.startswith("# ")), doc)
    step = max(1, max_words - overlap_words)
    out = []
    for start in range(0, max(len(words), 1), step):
        window = words[start:start + max_words]
        if not window:
            break
        out.append(Chunk(f"{doc}#{len(out)}", doc, title, "", "", " ".join(window)))
        if start + max_words >= len(words):
            break
    return out


def load_chunks(docs_dir, strategy="heading", **kwargs):
    fn = {"heading": chunk_heading, "fixed": chunk_fixed}[strategy]
    chunks = []
    for path in sorted(Path(docs_dir).glob("*.md")):
        chunks.extend(fn(path.stem, path.read_text(encoding="utf-8"), **kwargs))
    return chunks

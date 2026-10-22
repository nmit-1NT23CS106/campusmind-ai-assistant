"""Put the latest evaluation tables into README.md (safe to run again after each evaluation).

Usage:  python -m scripts.update_readme
Reads eval/results/comparison.md (from `python -m eval.run_eval --all`) and
eval/results/agent_eval.md (from `python -m eval.agent_eval`).
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
RESULTS = ROOT / "eval" / "results"
SECTIONS = {
    "retrieval": ("comparison.md", "<!-- paste eval/results/comparison.md here -->"),
    "agent": ("agent_eval.md", "<!-- paste eval/results/agent_eval.md here -->"),
}


def update(readme_text, results_dir=RESULTS):
    """Return (new_text, updated_names, skipped_names)."""
    updated, skipped = [], []
    for name, (filename, placeholder) in SECTIONS.items():
        path = Path(results_dir) / filename
        if not path.exists():
            skipped.append(f"{name} (no {filename} yet)")
            continue
        block = f"<!-- {name}-results:start -->\n{path.read_text(encoding='utf-8').strip()}\n<!-- {name}-results:end -->"
        existing = re.compile(rf"<!-- {name}-results:start -->.*?<!-- {name}-results:end -->", re.S)
        if existing.search(readme_text):
            readme_text = existing.sub(lambda m: block, readme_text)
        elif placeholder in readme_text:
            readme_text = readme_text.replace(placeholder, block)
        else:
            skipped.append(f"{name} (marker not found in README)")
            continue
        updated.append(name)
    return readme_text, updated, skipped


def main():
    text, updated, skipped = update(README.read_text(encoding="utf-8"))
    README.write_text(text, encoding="utf-8")
    print("Updated README sections:", ", ".join(updated) or "none")
    for item in skipped:
        print("Skipped:", item)


if __name__ == "__main__":
    main()

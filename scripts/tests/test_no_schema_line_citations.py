"""Guard: live code must cite docs by section heading, never by line number.

`docs/quality-card-schema-v1.md:312-314` rots on the next paragraph added to the doc
(it was already ~182 lines off when first audited). Cite the heading instead, e.g.
`docs/quality-card-schema-v1.md § Tier mapping`. Dated records under `docs/` are
deliberately out of scope: their citations were accurate when written.

Scope: text files (.py .swift .sh .md .json) under `scripts/`, `spike/Sources`, and
`spike/Tests`. This file is skipped (it spells the forbidden pattern). The pattern also
covers any other `docs/<name>.md:<digits>` citation (there were zero of those in
these trees when the guard was added).
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOTS = ("scripts", "spike/Sources", "spike/Tests")
SUFFIXES = {".py", ".swift", ".sh", ".md", ".json"}
SKIP_DIRS = {".build", ".git", "__pycache__", "node_modules", ".swiftpm"}

# Built by concatenation so this file does not match its own pattern even unskipped.
LINE_CITATION = re.compile(
    r"quality-card-schema-v1\.md" + r":\d" + r"|docs/[A-Za-z0-9_.-]+\.md" + r":\d+"
)

# Narrow, reasoned exceptions: {repo-relative path: exact substring}. Empty on purpose.
ALLOWLIST: dict[str, tuple[str, ...]] = {}


def _scan() -> list[str]:
    hits: list[str] = []
    for root in SCAN_ROOTS:
        base = REPO_ROOT / root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix not in SUFFIXES:
                continue
            if SKIP_DIRS.intersection(path.relative_to(REPO_ROOT).parts):
                continue
            if path.resolve() == Path(__file__).resolve():
                continue
            rel = path.relative_to(REPO_ROOT).as_posix()
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            allowed = ALLOWLIST.get(rel, ())
            for lineno, line in enumerate(text.splitlines(), start=1):
                if LINE_CITATION.search(line) and not any(a in line for a in allowed):
                    hits.append(f"{rel}:{lineno}: {line.strip()}")
    return hits


class NoSchemaLineCitationsTests(unittest.TestCase):
    def test_no_line_number_citations_into_docs(self) -> None:
        hits = _scan()
        self.assertEqual(
            [],
            hits,
            "line-number citations into docs rot on the next edit; cite the section "
            "heading instead (docs/<file>.md § <heading>):\n" + "\n".join(hits),
        )

    def test_pattern_discriminates(self) -> None:
        # Positive controls: the regex must catch the shapes it exists to catch ...
        for bad in (
            "docs/quality-card-schema-v1.md" + ":312-314",
            "quality-card-schema-v1.md" + ":31",
            "docs/other-doc.md" + ":7",
        ):
            self.assertIsNotNone(LINE_CITATION.search(bad), bad)
        # ... and pass the heading convention.
        for good in (
            'docs/quality-card-schema-v1.md § Tier mapping',
            'docs/quality-card-schema-v1.md "Engine build"',
        ):
            self.assertIsNone(LINE_CITATION.search(good), good)


if __name__ == "__main__":
    unittest.main()

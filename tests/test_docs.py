"""Markdown hazards in the docs that are invisible in the source.

These exist because the same defect slipped through twice. A single tilde is
strikethrough in GitHub-flavored Markdown, so Korean range notation like
``4,000~6,000자`` pairs with the next tilde in the same block and silently strikes
out everything between them. Reading the source gives no hint; you have to look at
the rendered page. A test is the cheaper place to catch it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DOCS = [REPO / "README.md", REPO / "README.ko.md", REPO / "docs" / "design.md"]


def _strip_code(text: str) -> str:
    """Remove fenced blocks and inline spans; Markdown is inert inside them."""
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    return re.sub(r"`[^`\n]*`", "", text)


def _blocks(text: str) -> list[str]:
    """Blank-line separated blocks, which is the scope emphasis pairs within."""
    return re.split(r"\n\s*\n", text)


@pytest.fixture(params=DOCS, ids=lambda p: p.name)
def doc(request) -> tuple[Path, str]:
    path = request.param
    assert path.is_file(), path
    return path, path.read_text(encoding="utf-8")


class TestNoAccidentalStrikethrough:
    def test_no_unescaped_tilde_outside_code(self, doc):
        """A bare ``~`` outside code is a strikethrough delimiter in GFM.

        Korean writes ranges as ``300~400``; escaped as ``300\\~400`` it renders
        identically and cannot pair with anything.
        """
        path, text = doc
        offenders = []
        for i, line in enumerate(_strip_code(text).split("\n"), 1):
            if re.search(r"(?<!\\)~", line):
                offenders.append(f"{path.name}:{i}: {line.strip()[:70]}")
        assert not offenders, "escape these as \\~:\n" + "\n".join(offenders)

    def test_no_deliberate_strikethrough(self, doc):
        """Resolved items belong deleted, not crossed out."""
        _path, text = doc
        assert "~~" not in _strip_code(text)


class TestEmphasisIsBalanced:
    def test_bold_markers_pair_within_each_block(self, doc):
        path, text = doc
        unbalanced = [
            block.strip()[:70]
            for block in _blocks(_strip_code(text))
            if block.count("**") % 2
        ]
        assert not unbalanced, f"{path.name}: unpaired ** in:\n" + "\n".join(unbalanced)

    def test_no_bold_span_swallows_a_sentence_boundary(self, doc):
        """Guards a mechanical edit that merged adjacent short bolds into one span.

        The tell is a sentence boundary *inside* the span: emphasis that continues
        past a full stop is emphasis that stopped meaning anything. A whole
        sentence in bold is fine (the README tagline is one), so the check looks
        for a period followed by more content rather than for length.
        """
        path, text = doc
        swallowed = [
            span.replace("\n", " ")[:80]
            for span in re.findall(r"\*\*((?:[^*]|\n)+?)\*\*", _strip_code(text))
            if re.search(r"[.!?]\s+\S", span)
        ]
        assert not swallowed, (
            f"{path.name}: bold running past a sentence end:\n" + "\n".join(swallowed)
        )


class TestStructure:
    def test_code_fences_are_closed(self, doc):
        path, text = doc
        assert text.count("```") % 2 == 0, f"{path.name}: odd number of fences"

    def test_tables_have_consistent_columns(self, doc):
        path, text = doc
        for block in re.findall(r"(?m)((?:^\|.*\|\s*$\n){2,})", text):
            rows = block.strip().split("\n")
            widths = {row.count("|") for row in rows}
            assert len(widths) == 1, f"{path.name}: ragged table\n" + "\n".join(
                rows[:3]
            )

    def test_internal_anchors_resolve(self, doc):
        """A table of contents that points at nothing is worse than none."""
        path, text = doc

        def slug(heading: str) -> str:
            heading = heading.lower().replace(" ", "-")
            return re.sub(r"[^0-9a-z가-힣\-]", "", heading)

        headings = {slug(h) for h in re.findall(r"(?m)^#{1,4}\s+(.+)$", text)}
        broken = [
            link
            for link in re.findall(r"\]\(#([^)]+)\)", text)
            if re.sub(r"[^0-9a-z가-힣\-]", "", link.lower()) not in headings
        ]
        assert not broken, f"{path.name}: dead anchors {broken}"


class TestNumbersMatchTheirSource:
    """A number in the docs that nothing checks goes stale at the first change.

    Both READMEs advertised "71 Python dependencies" for weeks after the lock
    dropped to 65 — the same commit that corrected design.md missed them, because
    nothing tied the sentence to the file it describes.
    """

    def test_dependency_count_matches_the_lock(self):
        pinned = sum(
            1
            for line in (REPO / "app" / "requirements.lock")
            .read_text(encoding="utf-8")
            .splitlines()
            if "==" in line and not line.lstrip().startswith("#")
        )
        for path in (REPO / "README.md", REPO / "README.ko.md"):
            text = path.read_text(encoding="utf-8")
            claimed = re.findall(
                r"(?:all (\d+) Python\s+dependencies|Python 의존성 (\d+)개)", text
            )
            flat = [int(n) for pair in claimed for n in pair if n]
            assert flat, f"{path.name}: no dependency-count claim found to check"
            assert all(n == pinned for n in flat), (
                f"{path.name} claims {flat} pinned dependencies, "
                f"app/requirements.lock has {pinned}"
            )


class TestReadmesStayInSync:
    """The Korean README is a translation, not a fork: same sections, same order."""

    def test_same_heading_structure(self):
        def levels(path: Path) -> list[int]:
            text = path.read_text(encoding="utf-8")
            return [
                len(m.group(1))
                for m in re.finditer(r"(?m)^(#{2,3})\s+", _strip_code(text))
            ]

        en, ko = levels(REPO / "README.md"), levels(REPO / "README.ko.md")
        assert en == ko, f"heading outline differs: {en} vs {ko}"

    def test_same_number_of_component_rows(self):
        def rows(path: Path) -> int:
            return len(re.findall(r"(?m)^\| `", path.read_text(encoding="utf-8")))

        assert rows(REPO / "README.md") == rows(REPO / "README.ko.md")

"""Rubric for the quality of a generated summary, measured not judged.

``eval_metrics`` scores the FILTER (does the right article get picked?). This
scores the OUTPUT (is the write-up any good?), which is what the reader actually
receives and what prompt tuning has to be steered by.

Every check here is **deterministic**: it measures a property the summarization
prompt explicitly asks for, so a low score points at a specific instruction that
is not landing rather than at a matter of taste. That is what makes it usable as
a tuning signal — an LLM judge can rate "is this good?", but it cannot tell you
*which sentence of the prompt to change*, and it drifts between runs.

The dimensions map one-to-one onto prompt rules:

============================  ==================================================
dimension                     prompt rule it measures
============================  ==================================================
``length``                    the 4,000-6,000 visible-character budget
``structure``                 required sections, no subsection headers
``register``                  "모든 문장을 합니다체로 통일" (no 한다체 mixing)
``terminal_colon``            "Korean sentences end with a period, never a colon"
``distinctiveness``           "📊 is not a place to re-list 📌/🛠️ figures"
``specificity``               "mechanisms, parameters, measured numbers"
``boilerplate``               the banned formulaic audience call-out
``lede``                      the one-sentence ``one_liner`` contract
``captions``                  images wrapped in a figure WITH a caption
============================  ==================================================

Kept free of boto3/LangChain/bs4-optional so it is unit-testable offline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup

# --------------------------------------------------------------------------- #
# Targets, mirroring app/src/prompts/prompts.py. Keep the two in step; the
# tests assert the prompt states the same numbers.
# --------------------------------------------------------------------------- #
KO_LENGTH_BAND: tuple[int, int] = (4000, 6000)
# The UPPER bound is derived from the card layout: at the lede's font size an
# 800px card fits roughly 50 Korean characters per line, and the lede must not run
# past two lines. The lower bound only has to reject something that is not a lede
# at all — a concrete sentence naming a mechanism and its effect fits in ~45
# characters, so anything above 40 is legitimately short rather than empty.
KO_LEDE_BAND: tuple[int, int] = (40, 100)

# The five section markers the Korean prompt asks for, in order.
SECTION_MARKERS: tuple[str, ...] = ("📌", "🔄", "🛠️", "📊", "🔮")

# Formulaic closings the prompt bans by name, because every article ended the
# same way and it read as boilerplate.
BANNED_CLOSINGS: tuple[str, ...] = (
    "주목할 만한 참고 자료",
    "개발자와 아키텍트에게",
    "참고할 만한 자료입니다",
    "좋은 참고가 될 것입니다",
)

# Meta-framing the lede must not use: it should state the finding, not announce
# that an article exists.
BANNED_LEDE_OPENERS: tuple[str, ...] = (
    "이 글은",
    "이 아티클은",
    "이 문서는",
    "본 글은",
    "이 포스트는",
)

# "정보 없음" filler: the prompt says omit a section rather than write this.
ABSENCE_FILLER: tuple[str, ...] = (
    "정보가 없습니다",
    "언급되지 않았습니다",
    "확인할 수 없습니다",
    "명시되어 있지 않습니다",
)

# A MEASURED quantity: a number that carries a unit, a percent sign or a Korean
# comparative marker. The unit is required on purpose — an earlier version matched
# bare numbers and counted "Gemini 3.6" and "GPT-5.5" as measurements, which both
# inflated the specificity score and produced phantom "📊 repeats a figure"
# findings from version strings. Tuning a prompt against that would have chased a
# defect that did not exist.
_MEASURED = re.compile(
    r"\d[\d,.]*\s*"
    r"(?:%|퍼센트|배|배속|ms|밀리초|초|분|시간|GB|MB|KB|TB|GiB|MiB|"
    r"[Bb]ps|Gbps|Mbps|fps|토큰|tokens?|vCPU|GPU|원|달러|USD|건|개|회|"
    r"포인트|p|x(?=\s|$))"
)


def visible_text(html: str) -> str:
    """Readable prose in a summary, markup stripped, code blocks removed.

    Code is excluded because it is not prose: its identifiers would otherwise
    pollute the register and sentence-level checks.
    """
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup.find_all(["pre", "code"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


def split_sections(html: str) -> dict[str, str]:
    """Map each section marker to that section's visible text."""
    soup = BeautifulSoup(html or "", "html.parser")
    sections: dict[str, str] = {}
    current: str | None = None
    buffer: list[str] = []
    for element in soup.children:
        name = getattr(element, "name", None)
        if name == "h3":
            if current:
                sections[current] = " ".join(buffer).strip()
            heading = element.get_text(strip=True)
            current = next((m for m in SECTION_MARKERS if m in heading), None)
            buffer = []
        elif current:
            buffer.append(visible_text(str(element)))
    if current:
        sections[current] = " ".join(buffer).strip()
    return sections


def quantities(text: str) -> set[str]:
    """Distinct measured quantities in ``text`` (numbers WITH a unit).

    Version strings ("Gemini 3.6") are deliberately excluded: they are not
    measurements, and treating them as such both flatters the specificity score
    and invents 📊-repeats-a-figure findings.
    """
    return {m.group(0).strip() for m in _MEASURED.finditer(text) if m.group(0).strip()}


def sentences(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    return parts


def han_da_sentences(text: str) -> list[str]:
    """Sentences in the plain 한다체 register instead of the required 합니다체.

    합니다체 always ends "...니다."; a Korean sentence ending "다." without the
    "니" is the 문어체 the prompt forbids mixing in. Reported as examples so a
    reviewer can confirm rather than trust the count.
    """
    offenders = []
    for sentence in sentences(text):
        stripped = sentence.rstrip()
        if not stripped.endswith("다."):
            continue
        if stripped.endswith("니다."):
            continue
        offenders.append(stripped[-30:])
    return offenders


@dataclass
class SummaryQuality:
    """Per-article rubric result. Every score is in [0, 1], higher is better."""

    article_id: str
    length: int = 0
    sections: tuple[str, ...] = ()
    missing_sections: tuple[str, ...] = ()
    subsection_headers: int = 0
    absence_filler: int = 0
    han_da: tuple[str, ...] = ()
    terminal_colons: int = 0
    repeated_quantities: tuple[str, ...] = ()
    quantity_count: int = 0
    code_blocks: int = 0
    tables: int = 0
    images: int = 0
    captioned_images: int = 0
    banned_closings: tuple[str, ...] = ()
    lede: str = ""
    lede_sentences: int = 0
    notes: list[str] = field(default_factory=list)

    # -- individual dimensions ------------------------------------------------
    @property
    def length_score(self) -> float:
        low, high = KO_LENGTH_BAND
        if low <= self.length <= high:
            return 1.0
        # Linear falloff, so "slightly short" is distinguishable from "gutted".
        distance = low - self.length if self.length < low else self.length - high
        return max(0.0, 1.0 - distance / low)

    @property
    def structure_score(self) -> float:
        # 📌/🛠️ are mandatory; the rest may be legitimately omitted when the
        # source lacks the material, so only penalise the two load-bearing ones.
        required = {"📌", "🛠️"}
        present = set(self.sections)
        base = len(required & present) / len(required)
        penalty = 0.25 * self.subsection_headers + 0.25 * self.absence_filler
        return max(0.0, base - penalty)

    @property
    def register_score(self) -> float:
        return 1.0 if not self.han_da else max(0.0, 1.0 - 0.2 * len(self.han_da))

    @property
    def terminal_colon_score(self) -> float:
        return 1.0 if not self.terminal_colons else 0.0

    @property
    def distinctiveness_score(self) -> float:
        """📊 must add implication, not re-list figures already given."""
        if not self.repeated_quantities:
            return 1.0
        return max(0.0, 1.0 - 0.2 * len(self.repeated_quantities))

    @property
    def specificity_score(self) -> float:
        """Measured quantities plus code/table evidence, per 1,000 characters.

        A technical write-up of this length that names no numbers is a summary of
        vibes. 8 quantities per 1,000 chars saturates the score.
        """
        if not self.length:
            return 0.0
        density = 1000 * self.quantity_count / self.length
        evidence = min(1.0, 0.5 * (self.code_blocks + self.tables))
        return min(1.0, 0.7 * min(1.0, density / 8.0) + 0.3 * evidence)

    @property
    def boilerplate_score(self) -> float:
        return 1.0 if not self.banned_closings else 0.0

    @property
    def lede_score(self) -> float:
        if not self.lede:
            return 0.0
        low, high = KO_LEDE_BAND
        score = 1.0
        if not low <= len(self.lede) <= high:
            score -= 0.4
        if self.lede_sentences != 1:
            score -= 0.3
        if any(self.lede.startswith(opener) for opener in BANNED_LEDE_OPENERS):
            score -= 0.3
        return max(0.0, score)

    @property
    def caption_score(self) -> float:
        if not self.images:
            return 1.0  # nothing to caption is not a defect
        return self.captioned_images / self.images

    DIMENSIONS = (
        "length",
        "structure",
        "register",
        "terminal_colon",
        "distinctiveness",
        "specificity",
        "boilerplate",
        "lede",
        "caption",
    )

    def scores(self) -> dict[str, float]:
        return {d: getattr(self, f"{d}_score") for d in self.DIMENSIONS}

    @property
    def overall(self) -> float:
        values = self.scores().values()
        return sum(values) / len(values)


def evaluate_summary(
    article_id: str, summary_html: str, one_liner: str = ""
) -> SummaryQuality:
    """Score one article's summary against the rubric."""
    text = visible_text(summary_html)
    sections = split_sections(summary_html)
    soup = BeautifulSoup(summary_html or "", "html.parser")

    earlier = " ".join(sections.get(m, "") for m in ("📌", "🔄", "🛠️"))
    results_section = sections.get("📊", "")
    repeated = sorted(quantities(results_section) & quantities(earlier))

    images = soup.find_all("img")
    captioned = 0
    for img in images:
        figure = img.find_parent("figure")
        # A caption must exist AND say something: an empty <figcaption> is the
        # same defect as no caption at all.
        caption = figure.find("figcaption") if figure else None
        if caption is not None and caption.get_text(strip=True):
            captioned += 1

    return SummaryQuality(
        article_id=article_id,
        length=len(text),
        sections=tuple(m for m in SECTION_MARKERS if m in sections),
        missing_sections=tuple(m for m in SECTION_MARKERS if m not in sections),
        subsection_headers=len(soup.find_all(["h4", "h5", "h6"])),
        absence_filler=sum(text.count(p) for p in ABSENCE_FILLER),
        han_da=tuple(han_da_sentences(text)),
        terminal_colons=len(re.findall(r"[다요죠]:(?=\s|$)", text)),
        repeated_quantities=tuple(repeated),
        quantity_count=len(quantities(text)),
        code_blocks=len(soup.find_all("pre")),
        tables=len(soup.find_all("table")),
        images=len(images),
        captioned_images=captioned,
        banned_closings=tuple(p for p in BANNED_CLOSINGS if p in text),
        lede=one_liner.strip(),
        lede_sentences=len(sentences(one_liner)) if one_liner.strip() else 0,
    )


@dataclass
class QualityReport:
    results: list[SummaryQuality]

    def mean(self, dimension: str) -> float:
        if not self.results:
            return 0.0
        return sum(r.scores()[dimension] for r in self.results) / len(self.results)

    @property
    def overall(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.overall for r in self.results) / len(self.results)

    def weakest(self, n: int = 3) -> list[tuple[str, float]]:
        """The dimensions to tune next, worst first."""
        means = [(d, self.mean(d)) for d in SummaryQuality.DIMENSIONS]
        return sorted(means, key=lambda kv: kv[1])[:n]

    def format_table(self) -> str:
        dims = SummaryQuality.DIMENSIONS
        header = f"{'article':30s} {'chars':>6s} " + " ".join(
            f"{d[:6]:>6s}" for d in dims
        )
        lines = [header, "-" * len(header)]
        for r in self.results:
            row = " ".join(f"{r.scores()[d]:>6.2f}" for d in dims)
            lines.append(f"{r.article_id[:30]:30s} {r.length:>6,} {row}")
        lines.append("-" * len(header))
        means = " ".join(f"{self.mean(d):>6.2f}" for d in dims)
        lines.append(f"{'MEAN':30s} {'':>6s} {means}")
        lines.append(f"\noverall {self.overall:.3f}")
        lines.append(
            "weakest: "
            + ", ".join(f"{d}={v:.2f}" for d, v in self.weakest())
            + "   <- tune these"
        )
        return "\n".join(lines)

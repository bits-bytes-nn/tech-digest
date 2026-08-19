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
``translationese``            "영어 문장 구조를 옮기지 말고" (번역투 금지)
``cliche``                    the banned praise adjectives and filler formulas
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

# Measured quantities per 1,000 visible characters at which ``specificity``
# saturates. Derived from the 62 summaries in ``inputs/``, not chosen: their
# density runs median 1.5, p90 4.3, best three 6.5 / 7.8 / 8.5, and the current
# tuned prompt produces 5.1-5.8.
#
# It was 8.0 before, which exactly ONE of the 62 ever reached. A target that 98%
# of real output cannot hit is not a tuning signal — ``weakest()`` named
# specificity every round, drowning out dimensions that were actually fixable,
# and steering the prompt toward cramming in more figures. That also fights
# ``distinctiveness``, which penalises a figure repeated across sections: pushing
# one up pushed the other down, which is what happened when the 📊 section started
# re-stating numbers from 📌.
#
# 6.0 keeps the dimension discriminating (the median still scores ~0.2) while
# making full marks reachable — the gap from current output is one or two more
# carried figures rather than twenty.
#
# The deeper limit is that density is bounded by the SOURCE: an article reporting
# five numbers cannot yield thirty, so no fixed target is right for every article.
# Two alternatives were measured and rejected. Recall of the source's quantities
# saturates (median 1.00 once numerals are compared language-neutrally) and
# unfairly penalises number-dense sources — one article reports 107 figures, and
# carrying all of them would be worse writing, not better. Counting summary
# figures ABSENT from the source looked like a hallucination check, but
# ``visible_text`` strips code blocks, so figures legitimately taken from a config
# snippet count as unsourced.
SPECIFICITY_SATURATION: float = 6.0

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


# --------------------------------------------------------------------------- #
# 번역투 (translationese) and 상투구 (cliché).
#
# These two dimensions differ from the rest in one important way: their marker
# lists are a SAMPLE of a pattern, not a definition of it. Every other check here
# measures something the prompt states exactly ("end sentences with a period"),
# so presence is the defect. Here the words are not forbidden — "다양한" and
# "~를 통해" are ordinary Korean — it is the RATE that reveals prose translated
# out of English rather than written in Korean.
#
# So the scores below are densities with a tolerance floor, and the markers were
# picked by frequency over the 59 summaries in ``inputs/``, not by taste. The
# spread across prompt generations is what makes them usable as a signal: per
# 1,000 visible characters the first six issues ran 4.8 translationese / 1.8
# cliché, and the three most recent ran 2.1 / 0.7. Anything that only fires on
# bad prose but never on good prose cannot steer a prompt.
# --------------------------------------------------------------------------- #
TRANSLATIONESE_MARKERS: dict[str, str] = {
    # Instrumental "through/via", where a Korean particle carries the same sense.
    "~를 통해": r"(?:을|를) 통(?:해|하여)",
    # "about/for X", where the plain noun phrase would do.
    "~에 대한": r"에 (?:대해|대한|대하여)",
    # Nominalisation: "the fact that ~ing", where a finite verb is shorter.
    "~하는 것": r"[가-힣]는 것(?:이|은|을|과|와|입니다)",
    # Redundant plural: Korean does not mark number, so "-들" carried over from
    # English plurals piles up. Approximate — it also catches verb stems ending
    # in 들 ("이야기를 들을") — which the density tolerance absorbs.
    "복수 -들": r"(?<=[가-힣A-Za-z0-9])들(?=[이은을의과와에로])",
    # "provides/supports" as the whole predicate, in place of what it does.
    "제공/지원합니다": r"(?:제공|지원)(?:합니다|됩니다|하는|되는|하며|되며)",
    # Agentless passive where Korean prefers an active subject.
    "~에 의해": r"에 의(?:해|하여)",
}

CLICHE_MARKERS: dict[str, str] = {
    # Evaluation by adjective, in a newsletter whose value is measured numbers.
    "칭찬 형용사": r"(?:혁신적|획기적|강력한|놀라운|인상적인|매력적인|눈부신)",
    # "various", without saying various what, or how many.
    "다양한": r"다양한",
    # The "not just X, but Y" contrast: a shape a model reaches for by default.
    "단순히 X가 아니라": r"단순(?:한|히|하게)?\s*[^.]{0,25}?(?:가|이|것이)\s*아니(?:라|고|며)",
    # Ending on "the point is that ~" instead of just asserting it. The one
    # marker that rose between prompt generations (2.4 -> 4.0 per 10,000).
    "~라는 점입니다": r"(?:라는|다는) 점(?:입니다|이며|이라|에서|은|을)",
    # Meta-narration and summary-of-the-summary.
    "메타 서술": r"(?:살펴보|알아보|짚어보|정리해\s*보)(?:겠습니다|았습니다|면)"
    r"|결론적으로|요약하(?:면|자면)|종합하(?:면|자면)",
}

_TRANSLATIONESE = {k: re.compile(v) for k, v in TRANSLATIONESE_MARKERS.items()}
_CLICHE = {k: re.compile(v) for k, v in CLICHE_MARKERS.items()}

# (full credit at or below, zero at or above) hits per 1,000 visible characters.
# The floors are deliberately non-zero: demanding zero would tune the prompt into
# avoiding ordinary Korean words, which is a different defect.
TRANSLATIONESE_BAND: tuple[float, float] = (1.0, 5.0)
CLICHE_BAND: tuple[float, float] = (0.3, 2.0)


def _marker_hits(text: str, patterns: dict[str, re.Pattern[str]]) -> dict[str, int]:
    """Per-marker counts, kept so a score can be traced to specific phrases."""
    counts = {name: len(rx.findall(text)) for name, rx in patterns.items()}
    return {name: n for name, n in counts.items() if n}


def translationese_hits(text: str) -> dict[str, int]:
    return _marker_hits(text, _TRANSLATIONESE)


def cliche_hits(text: str) -> dict[str, int]:
    return _marker_hits(text, _CLICHE)


def _density_score(hits: int, length: int, band: tuple[float, float]) -> float:
    """Score a per-1,000-character rate against a (free, zero) band."""
    if not length:
        return 0.0
    free, zero = band
    rate = 1000 * hits / length
    if rate <= free:
        return 1.0
    return max(0.0, 1.0 - (rate - free) / (zero - free))


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
    translationese: dict[str, int] = field(default_factory=dict)
    cliches: dict[str, int] = field(default_factory=dict)
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
        vibes. See ``SPECIFICITY_SATURATION`` for where the target comes from.
        """
        if not self.length:
            return 0.0
        density = 1000 * self.quantity_count / self.length
        evidence = min(1.0, 0.5 * (self.code_blocks + self.tables))
        return min(
            1.0,
            0.7 * min(1.0, density / SPECIFICITY_SATURATION) + 0.3 * evidence,
        )

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

    @property
    def translationese_score(self) -> float:
        """English sentence structure carried into Korean, by rate not presence."""
        return _density_score(
            sum(self.translationese.values()), self.length, TRANSLATIONESE_BAND
        )

    @property
    def cliche_score(self) -> float:
        return _density_score(sum(self.cliches.values()), self.length, CLICHE_BAND)

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
        "translationese",
        "cliche",
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
        translationese=translationese_hits(text),
        cliches=cliche_hits(text),
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

"""Pure, dependency-free metrics for evaluating filtering-score behavior.

Used by ``scripts/eval_filtering.py`` to quantify, against a labeled eval set:
  - **determinism**: does the same article get the same score across repeats?
  - **anchor adherence**: is every score one of the rubric's anchors?
  - **band alignment**: does the score fall in the expected quality band?

Kept free of boto3/LangChain so the analysis is unit-testable offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev

# Band label for each anchor the filtering rubric can emit, taken from the
# rubric's OWN wording (see app/src/prompts/prompts.py):
#
#   0.95-0.85 included topic / 0.80 Groundbreaking / 0.75 Excellent -> high
#   0.70 Strong                                                   -> strong
#   0.60 Good / 0.50 Moderate                                     -> moderate
#   0.35 Weak                                                     -> weak
#   0.15 Poor / 0.05 Not ML                                       -> reject
#
# This mapping is the single source of truth: ``band_of_score`` classifies by
# nearest anchor, with no second table of cutoffs to drift from it.
# ``test_eval_metrics`` pins every anchor to the band its rubric label names.
ANCHOR_BANDS: dict[float, str] = {
    0.95: "high",
    0.90: "high",
    0.85: "high",
    0.80: "high",
    0.75: "high",
    0.70: "strong",
    0.60: "moderate",
    0.50: "moderate",
    0.35: "weak",
    0.15: "reject",
    0.05: "reject",
}


def is_anchor(score: float, tol: float = 0.001) -> bool:
    """Whether a score is one of the rubric's anchors (within float tolerance).

    The rubric allows no adjustment off an anchor, so 0.65 is as much a defect
    as 0.83 even though both sit on a 0.05 grid.
    """
    return any(abs(score - anchor) <= tol for anchor in ANCHOR_BANDS)


def band_of_score(score: float) -> str:
    """Coarse quality band label for a score: the band of its NEAREST anchor.

    Ties resolve DOWN, matching the rubric's own instruction ("when torn between
    two anchors, choose the LOWER one"): 0.65 reads moderate, 0.25 reject.
    """
    # The distance is rounded before comparing: 0.65 is nominally equidistant from
    # 0.70 and 0.60, but in binary those distances are 0.049999... and 0.050000...,
    # so an exact comparison would silently pick the higher anchor and defeat the
    # tie-break below.
    nearest = min(
        ANCHOR_BANDS, key=lambda anchor: (round(abs(score - anchor), 6), anchor)
    )
    return ANCHOR_BANDS[nearest]


@dataclass
class ArticleScoreStats:
    """Per-article aggregate over N repeated scorings."""

    article_id: str
    expected_band: str
    scores: list[float] = field(default_factory=list)

    @property
    def mean_score(self) -> float:
        return mean(self.scores) if self.scores else 0.0

    @property
    def stdev(self) -> float:
        return pstdev(self.scores) if len(self.scores) > 1 else 0.0

    @property
    def spread(self) -> float:
        return (max(self.scores) - min(self.scores)) if self.scores else 0.0

    @property
    def is_deterministic(self) -> bool:
        """All repeats produced the same score (true determinism at temp=0)."""
        return self.spread == 0.0

    @property
    def all_on_anchor(self) -> bool:
        return all(is_anchor(s) for s in self.scores)

    @property
    def band_matches(self) -> bool:
        return band_of_score(self.mean_score) == self.expected_band


@dataclass
class EvalReport:
    """Aggregate across all articles in the eval set."""

    stats: list[ArticleScoreStats]

    @property
    def determinism_rate(self) -> float:
        return self._rate(lambda s: s.is_deterministic)

    @property
    def anchor_rate(self) -> float:
        return self._rate(lambda s: s.all_on_anchor)

    @property
    def band_match_rate(self) -> float:
        return self._rate(lambda s: s.band_matches)

    @property
    def mean_stdev(self) -> float:
        return mean(s.stdev for s in self.stats) if self.stats else 0.0

    @property
    def max_spread(self) -> float:
        """Largest min-to-max score range any single article showed.

        This, not ``determinism_rate``, is the meaningful reproducibility metric
        on models that expose no sampling controls (Sonnet 5 and later removed
        ``temperature``, so greedy decoding cannot be requested and repeated
        scorings of identical input genuinely differ). What still matters is that
        the variation stays inside one anchor step, so an article cannot cross an
        inclusion threshold depending on which run it landed in.
        """
        return max((s.spread for s in self.stats), default=0.0)

    @property
    def band_stability_rate(self) -> float:
        """Fraction of articles whose EVERY repeat fell in the expected band."""
        if not self.stats:
            return 0.0
        return sum(
            1
            for s in self.stats
            if s.scores
            and all(band_of_score(score) == s.expected_band for score in s.scores)
        ) / len(self.stats)

    def _rate(self, predicate) -> float:
        if not self.stats:
            return 0.0
        return sum(1 for s in self.stats if predicate(s)) / len(self.stats)

    def format_table(self) -> str:
        lines = [
            f"{'article':<28} {'expect':<9} {'mean':>5} {'σ':>5} "
            f"{'spread':>6} {'anchr':>5} {'band✓':>6}",
            "-" * 70,
        ]
        for s in self.stats:
            lines.append(
                f"{s.article_id:<28} {s.expected_band:<9} {s.mean_score:>5.2f} "
                f"{s.stdev:>5.2f} {s.spread:>6.2f} "
                f"{'yes' if s.all_on_anchor else 'NO':>5} "
                f"{'yes' if s.band_matches else 'NO':>6}"
            )
        lines.append("-" * 70)
        lines.append(
            f"determinism={self.determinism_rate:.0%}  "
            f"on-anchor={self.anchor_rate:.0%}  "
            f"band-match={self.band_match_rate:.0%}  "
            f"band-stable={self.band_stability_rate:.0%}  "
            f"max spread={self.max_spread:.2f}  "
            f"mean σ={self.mean_stdev:.3f}"
        )
        return "\n".join(lines)


def build_report(
    results: dict[str, list[float]], expected_bands: dict[str, str]
) -> EvalReport:
    """Assemble an EvalReport from {article_id: [scores...]} and expected bands."""
    stats = [
        ArticleScoreStats(
            article_id=aid,
            expected_band=expected_bands.get(aid, "?"),
            scores=scores,
        )
        for aid, scores in results.items()
    ]
    return EvalReport(stats=stats)

"""Pure, dependency-free metrics for evaluating filtering-score behavior.

Used by ``scripts/eval_filtering.py`` to quantify, against a labeled eval set:
  - **determinism**: does the same article get the same score across repeats?
  - **grid adherence**: do scores land on the intended 0.05 anchor grid?
  - **band alignment**: does the score fall in the expected quality band?

Kept free of boto3/LangChain so the analysis is unit-testable offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev

# The discrete anchor grid the tuned filtering rubric is meant to emit
# (see app/src/prompts/prompts.py). A one-step ±0.05 adjustment is allowed,
# so any 0.05 multiple in [0, 1] is "on grid".
GRID_STEP = 0.05


def is_on_grid(score: float, tol: float = 0.001) -> bool:
    """Whether a score sits on the 0.05 grid (within floating tolerance)."""
    if score < 0.0 or score > 1.0:
        return False
    nearest = round(score / GRID_STEP) * GRID_STEP
    return abs(score - nearest) <= tol


def band_of_score(score: float) -> str:
    """Coarse quality band label for a score (for alignment reporting)."""
    if score >= 0.75:
        return "high"  # included-topic / exceptional
    if score >= 0.60:
        return "strong"
    if score >= 0.45:
        return "moderate"
    if score >= 0.30:
        return "weak"
    return "reject"


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
    def all_on_grid(self) -> bool:
        return all(is_on_grid(s) for s in self.scores)

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
    def grid_rate(self) -> float:
        return self._rate(lambda s: s.all_on_grid)

    @property
    def band_match_rate(self) -> float:
        return self._rate(lambda s: s.band_matches)

    @property
    def mean_stdev(self) -> float:
        return mean(s.stdev for s in self.stats) if self.stats else 0.0

    def _rate(self, predicate) -> float:
        if not self.stats:
            return 0.0
        return sum(1 for s in self.stats if predicate(s)) / len(self.stats)

    def format_table(self) -> str:
        lines = [
            f"{'article':<28} {'expect':<9} {'mean':>5} {'σ':>5} "
            f"{'spread':>6} {'grid':>5} {'band✓':>6}",
            "-" * 70,
        ]
        for s in self.stats:
            lines.append(
                f"{s.article_id:<28} {s.expected_band:<9} {s.mean_score:>5.2f} "
                f"{s.stdev:>5.2f} {s.spread:>6.2f} "
                f"{'yes' if s.all_on_grid else 'NO':>5} "
                f"{'yes' if s.band_matches else 'NO':>6}"
            )
        lines.append("-" * 70)
        lines.append(
            f"determinism={self.determinism_rate:.0%}  "
            f"on-grid={self.grid_rate:.0%}  "
            f"band-match={self.band_match_rate:.0%}  "
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

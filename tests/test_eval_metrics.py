"""Tests for the offline filtering-score eval metrics (determinism, grid
adherence, band alignment). These validate the analysis the live eval relies on."""

from __future__ import annotations

import pytest

from app.src.eval_metrics import (
    ArticleScoreStats,
    band_of_score,
    build_report,
    is_on_grid,
)


class TestIsOnGrid:
    def test_exact_anchors_on_grid(self):
        for s in (0.0, 0.05, 0.35, 0.7, 0.8, 0.85, 1.0):
            assert is_on_grid(s), s

    def test_off_grid_value(self):
        assert not is_on_grid(0.83)
        assert not is_on_grid(0.67)

    def test_out_of_range(self):
        assert not is_on_grid(-0.1)
        assert not is_on_grid(1.5)


class TestBandOfScore:
    def test_bands(self):
        assert band_of_score(0.85) == "high"
        assert band_of_score(0.70) == "strong"
        assert band_of_score(0.50) == "moderate"
        assert band_of_score(0.35) == "weak"
        assert band_of_score(0.10) == "reject"

    def test_every_rubric_anchor_lands_in_its_named_band(self):
        """The band cutoffs and the rubric's anchor labels are two views of one
        scale; when they were maintained separately they drifted (0.60 is "Good"
        in the rubric but the cutoffs called it "strong"), which reported a
        correctly-scored article as a band mismatch."""
        from app.src.eval_metrics import ANCHOR_BANDS

        mismatched = {
            anchor: (band_of_score(anchor), expected)
            for anchor, expected in ANCHOR_BANDS.items()
            if band_of_score(anchor) != expected
        }
        assert not mismatched, f"anchor -> band drift: {mismatched}"

    def test_one_step_adjustment_never_skips_a_band(self):
        """The rubric permits a single +-0.05 step off an anchor.

        Such a step can land exactly midway between two anchors (0.65 is both
        0.60+0.05 and 0.70-0.05), and a cutoff has to fall somewhere, so the
        adjusted score may legitimately read as either neighbour's band. What
        must NOT happen is landing two bands away — that would mean the cutoffs
        are placed on the anchors themselves rather than between them.
        """
        from app.src.eval_metrics import ANCHOR_BANDS

        anchors = sorted(ANCHOR_BANDS)
        for i, anchor in enumerate(anchors):
            neighbours = {ANCHOR_BANDS[anchor]}
            if i > 0:
                neighbours.add(ANCHOR_BANDS[anchors[i - 1]])
            if i < len(anchors) - 1:
                neighbours.add(ANCHOR_BANDS[anchors[i + 1]])
            for adjusted in (round(anchor - 0.05, 2), round(anchor + 0.05, 2)):
                if not 0.0 <= adjusted <= 1.0:
                    continue
                assert band_of_score(adjusted) in neighbours, (
                    f"{anchor} adjusted to {adjusted} -> "
                    f"{band_of_score(adjusted)}, not in {neighbours}"
                )

    def test_spread_and_band_stability_reported(self):
        """Reproducibility is measured as band stability plus bounded spread,
        because models from Sonnet 5 on expose no sampling controls and so cannot
        be asked for byte-identical repeats."""
        from app.src.eval_metrics import build_report

        report = build_report(
            # a: varies but never leaves its band. b: varies across bands.
            {"a": [0.50, 0.60], "b": [0.60, 0.80]},
            {"a": "moderate", "b": "moderate"},
        )
        assert report.max_spread == pytest.approx(0.20)
        assert report.band_stability_rate == pytest.approx(0.5)
        assert report.determinism_rate == 0.0

    def test_band_stability_is_all_repeats_not_just_the_mean(self):
        from app.src.eval_metrics import build_report

        # Mean 0.70 lands in "strong", but one repeat did not — band_match would
        # call this a pass while band_stability correctly does not.
        report = build_report({"a": [0.60, 0.80]}, {"a": "strong"})
        assert report.band_match_rate == 1.0
        assert report.band_stability_rate == 0.0

    def test_values_between_anchors(self):
        """A non-anchor score takes its nearest anchor's band, ties going DOWN.

        This used to assert 0.65 -> "strong" and 0.25 -> "weak", which was the old
        ``_BAND_CUTOFFS`` table's behaviour: both are exact midpoints between two
        bands' anchors, and that table resolved them upward. The rubric's own
        instruction for an ambiguous case is the opposite ("when torn between two
        anchors, always choose the LOWER one"), so they now read as the lower band.
        """
        assert band_of_score(0.75) == "high"  # an anchor
        assert band_of_score(0.65) == "moderate"  # tie: 0.70 strong / 0.60 moderate
        assert band_of_score(0.45) == "moderate"  # nearest 0.50
        assert band_of_score(0.25) == "reject"  # tie: 0.35 weak / 0.15 reject


class TestArticleScoreStats:
    def test_deterministic_identical_scores(self):
        s = ArticleScoreStats("a", "high", [0.85, 0.85, 0.85])
        assert s.is_deterministic
        assert s.spread == 0.0
        assert s.stdev == 0.0
        assert s.all_on_grid
        assert s.band_matches

    def test_non_deterministic_spread(self):
        s = ArticleScoreStats("a", "high", [0.85, 0.80, 0.75])
        assert not s.is_deterministic
        assert abs(s.spread - 0.10) < 1e-9
        assert s.stdev > 0

    def test_band_mismatch_detected(self):
        # Expected high but mean lands in 'strong'.
        s = ArticleScoreStats("a", "high", [0.70, 0.70])
        assert not s.band_matches

    def test_off_grid_detected(self):
        s = ArticleScoreStats("a", "strong", [0.67, 0.70])
        assert not s.all_on_grid


class TestBuildReport:
    def test_aggregate_rates(self):
        results = {
            "a": [0.85, 0.85],  # deterministic, on-grid, band high (matches)
            "b": [0.70, 0.65],  # non-det, on-grid, mean 0.675 -> strong (matches)
            "c": [0.83, 0.83],  # deterministic but OFF grid, band high
        }
        expected = {"a": "high", "b": "strong", "c": "high"}
        report = build_report(results, expected)
        # a and c are deterministic; b is not -> 2/3.
        assert abs(report.determinism_rate - 2 / 3) < 1e-9
        # a and b on grid; c not -> 2/3.
        assert abs(report.grid_rate - 2 / 3) < 1e-9
        # all three bands match their expected -> 1.0.
        assert report.band_match_rate == 1.0

    def test_empty_report_safe(self):
        report = build_report({}, {})
        assert report.determinism_rate == 0.0
        assert report.format_table()  # does not crash

    def test_format_table_contains_summary(self):
        report = build_report({"a": [0.85, 0.85]}, {"a": "high"})
        table = report.format_table()
        assert "determinism=" in table and "band-match=" in table


class TestBandsHaveNoSecondScale:
    """``band_of_score`` must be derived from ANCHOR_BANDS, not a parallel table.

    There used to be a ``_BAND_CUTOFFS`` tuple documented as sitting "midway
    between the adjacent anchors so a one-step ±0.05 adjustment stays inside its
    anchor's band". Two of its four values were not the midpoint, and the property
    is unachievable where two bands' anchors are one step apart — measured, three
    of nine anchors left their own band on a legal adjustment, including 0.70,
    which is the default ``min_score``.
    """

    def test_no_cutoff_table_remains(self):
        import app.src.eval_metrics as em

        assert not hasattr(em, "_BAND_CUTOFFS")

    def test_ties_resolve_to_the_lower_band(self):
        """The rubric says "when torn between two anchors, choose the LOWER one",
        so a score exactly between two anchors must read as the lower one's band."""
        assert band_of_score(0.65) == "moderate"  # between 0.70 strong / 0.60 moderate
        assert band_of_score(0.25) == "reject"  # between 0.35 weak / 0.15 reject

    def test_a_score_reads_as_its_nearest_anchor(self):
        from app.src.eval_metrics import ANCHOR_BANDS

        for anchor, band in ANCHOR_BANDS.items():
            assert band_of_score(anchor) == band, anchor
            # A hair off an anchor is still that anchor's band.
            assert band_of_score(round(anchor + 0.01, 2)) == band, anchor

    def test_out_of_grid_extremes_still_classify(self):
        assert band_of_score(1.0) == "high"
        assert band_of_score(0.0) == "reject"

"""Tests for the offline filtering-score eval metrics (determinism, grid
adherence, band alignment). These validate the analysis the live eval relies on."""

from __future__ import annotations

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

    def test_band_boundaries_inclusive(self):
        assert band_of_score(0.75) == "high"
        assert band_of_score(0.60) == "strong"
        assert band_of_score(0.45) == "moderate"
        assert band_of_score(0.30) == "weak"


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

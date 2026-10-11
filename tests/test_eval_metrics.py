"""Tests for the offline filtering-score eval metrics (determinism, anchor
adherence, band alignment). These validate the analysis the live eval relies on."""

from __future__ import annotations

import pytest

from app.src.eval_metrics import (
    ArticleScoreStats,
    band_of_score,
    build_report,
    is_anchor,
)


class TestIsAnchor:
    def test_every_anchor_is_one(self):
        from app.src.eval_metrics import ANCHOR_BANDS

        for s in ANCHOR_BANDS:
            assert is_anchor(s), s

    @pytest.mark.parametrize("score", [0.83, 0.67, 0.65, 0.55, 0.0, 1.0])
    def test_a_grid_value_off_the_anchors_is_not(self, score):
        """The rubric no longer allows a ±0.05 step, so 0.65 is a defect."""
        assert not is_anchor(score)

    def test_out_of_range(self):
        assert not is_anchor(-0.1)
        assert not is_anchor(1.5)


class TestBandOfScore:
    def test_bands(self):
        assert band_of_score(0.85) == "included"
        assert band_of_score(0.80) == "high"
        assert band_of_score(0.70) == "strong"
        assert band_of_score(0.50) == "moderate"
        assert band_of_score(0.35) == "weak"
        assert band_of_score(0.10) == "reject"

    @pytest.mark.parametrize("criteria", ["all", "amazon"])
    def test_the_prompt_states_exactly_these_anchors(self, criteria):
        """The rubric text and ``ANCHOR_BANDS`` are one scale written twice."""
        import re

        from app.src.constants import FilteringCriteria
        from app.src.eval_metrics import ANCHOR_BANDS
        from app.src.prompts import FilteringPrompt

        template = FilteringPrompt._human_prompt_template[FilteringCriteria(criteria)]
        stated = {float(m) for m in re.findall(r"- \*\*(\d\.\d\d)\*\* —", template)}
        assert stated == set(ANCHOR_BANDS)

    @pytest.mark.parametrize("criteria", ["all", "amazon"])
    def test_included_topic_anchors_outrank_every_other_anchor(self, criteria):
        """ "Included topics take priority" has to be true of the numbers, not
        only of the wording: the lowest included-topic anchor must beat the
        highest anchor for other content."""
        import re

        from app.src.constants import FilteringCriteria
        from app.src.prompts import FilteringPrompt

        template = FilteringPrompt._human_prompt_template[FilteringCriteria(criteria)]
        included, other = template.split("**OTHER ", 1)
        included = included.split("**INCLUDED-TOPIC CONTENT**", 1)[1]
        anchors = re.compile(r"- \*\*(\d\.\d\d)\*\* —")
        assert min(map(float, anchors.findall(included))) > max(
            map(float, anchors.findall(other))
        )

    def test_excluded_topics_override_included_ones(self):
        """Without a stated precedence the same article flipped between 0.85
        (included topic) and 0.05 (excluded topic) on identical input."""
        from app.src.prompts import FilteringPrompt

        for template in FilteringPrompt._human_prompt_template.values():
            assert "An excluded topic overrides an included one" in template

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
        s = ArticleScoreStats("a", "high", [0.80, 0.80, 0.80])
        assert s.is_deterministic
        assert s.spread == 0.0
        assert s.stdev == 0.0
        assert s.all_on_anchor
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
        assert not s.all_on_anchor


class TestBuildReport:
    def test_aggregate_rates(self):
        results = {
            "a": [0.80, 0.80],  # deterministic, on-anchor, band high (matches)
            "b": [0.70, 0.60],  # non-det, on-anchor, mean 0.65 -> moderate
            "c": [0.78, 0.78],  # deterministic but OFF grid, band high
        }
        expected = {"a": "high", "b": "moderate", "c": "high"}
        report = build_report(results, expected)
        # a and c are deterministic; b is not -> 2/3.
        assert abs(report.determinism_rate - 2 / 3) < 1e-9
        # a and b on an anchor; c not -> 2/3.
        assert abs(report.anchor_rate - 2 / 3) < 1e-9
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
    """``band_of_score`` reads a score as its nearest anchor's band."""

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

    def test_an_included_post_scored_as_other_content_is_a_band_miss(self):
        # The eval could not tell these apart while both were "high", so a model
        # ignoring the included topics still passed.
        assert not ArticleScoreStats("a", "included", [0.80, 0.80]).band_matches

    def test_out_of_grid_extremes_still_classify(self):
        assert band_of_score(1.0) == "included"
        assert band_of_score(0.0) == "reject"


class TestEvalSets:
    """The live eval is run by hand, so a typo in a set (a band name that does not
    exist, a criteria the prompt has no rubric for) would only show up as a
    confusing failure after paying for the calls."""

    @pytest.fixture(params=["all", "amazon"])
    def eval_set(self, request) -> dict:
        import json
        from pathlib import Path

        name = (
            "filtering_eval_set.json"
            if request.param == "all"
            else f"filtering_eval_set_{request.param}.json"
        )
        path = Path(__file__).parent / "eval_data" / name
        return json.loads(path.read_text(encoding="utf-8"))

    def test_labels_are_bands_and_ids_unique(self, eval_set):
        from app.src.eval_metrics import ANCHOR_BANDS

        ids = [a["id"] for a in eval_set["articles"]]
        assert len(ids) == len(set(ids))
        assert {a["expected_band"] for a in eval_set["articles"]} <= set(
            ANCHOR_BANDS.values()
        )

    def test_criteria_has_a_rubric(self, eval_set):
        from app.src.constants import FilteringCriteria

        FilteringCriteria(eval_set["criteria"])

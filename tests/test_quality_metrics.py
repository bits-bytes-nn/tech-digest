"""Tests for the output-quality rubric.

These matter more than most: the rubric is what prompt tuning is steered by, so a
wrong measurement sends the prompt in the wrong direction. That already happened
once — bare numbers were counted as measurements, so version strings like
"Gemini 3.6" both inflated ``specificity`` and produced phantom "📊 repeats a
figure" findings.
"""

from __future__ import annotations

import pytest

from app.src.quality_metrics import (
    KO_LENGTH_BAND,
    QualityReport,
    evaluate_summary,
    han_da_sentences,
    quantities,
    split_sections,
    visible_text,
)


def _summary(body: str) -> str:
    return f"<h3>📌 왜 주목해야 하나요?</h3><p>{body}</p><h3>🛠️ 기술적 심층 분석</h3><p>{body}</p>"


class TestQuantities:
    def test_units_are_required(self):
        """A number with a unit is a measurement; a bare number is not."""
        assert quantities("지연이 22% 줄었습니다") == {"22%"}
        assert quantities("8,000토큰을 전송합니다") == {"8,000토큰"}
        assert quantities("3.2배 빨라졌습니다") == {"3.2배"}

    def test_version_strings_are_not_measurements(self):
        """The bug this metric was rewritten to fix."""
        assert quantities("Gemini 3.6 대비 GPT-5.5는") == set()

    def test_mixed_text_keeps_only_the_measurements(self):
        found = quantities(
            "Gemini 3.7 Flash는 지연을 22% 낮추고 8,000토큰을 처리합니다"
        )
        assert found == {"22%", "8,000토큰"}


class TestRegister:
    def test_han_da_sentences_are_flagged(self):
        offenders = han_da_sentences("이 방식은 지연을 줄인다. 그리고 비용을 낮춘다.")
        assert len(offenders) == 2

    def test_hamnida_register_is_clean(self):
        assert han_da_sentences("이 방식은 지연을 줄입니다. 비용도 낮춥니다.") == []

    def test_code_identifiers_do_not_trip_the_register_check(self):
        """Code is stripped before the prose checks, so an identifier ending in
        "다" inside a snippet is not a register violation."""
        html = "<p>정상 문장입니다.</p><pre><code>x = 계산한다</code></pre>"
        assert han_da_sentences(visible_text(html)) == []


class TestSectionSplitting:
    def test_sections_keyed_by_marker(self):
        sections = split_sections(
            "<h3>📌 개요</h3><p>첫째</p><h3>📊 성과</h3><p>둘째</p>"
        )
        assert sections["📌"] == "첫째"
        assert sections["📊"] == "둘째"

    def test_missing_sections_reported(self):
        result = evaluate_summary("a", "<h3>📌 개요</h3><p>본문입니다.</p>")
        assert "🛠️" in result.missing_sections


class TestDistinctiveness:
    def test_results_section_repeating_an_earlier_figure_is_penalised(self):
        html = (
            "<h3>📌 개요</h3><p>지연이 22% 줄었습니다.</p>"
            "<h3>📊 성과</h3><p>다시 말해 22% 개선입니다.</p>"
        )
        result = evaluate_summary("a", html)
        assert "22%" in result.repeated_quantities
        assert result.distinctiveness_score < 1.0

    def test_results_section_adding_implication_is_clean(self):
        html = (
            "<h3>📌 개요</h3><p>지연이 22% 줄었습니다.</p>"
            "<h3>📊 성과</h3><p>스트리밍 서비스의 SLA 관리가 쉬워집니다.</p>"
        )
        assert evaluate_summary("a", html).distinctiveness_score == 1.0

    def test_shared_version_string_is_not_a_repeat(self):
        html = (
            "<h3>📌 개요</h3><p>Gemini 3.7 Flash입니다.</p>"
            "<h3>📊 성과</h3><p>Gemini 3.7 Flash가 채택되었습니다.</p>"
        )
        assert evaluate_summary("a", html).repeated_quantities == ()


class TestLength:
    def test_in_band_scores_full(self):
        low, _ = KO_LENGTH_BAND
        result = evaluate_summary("a", _summary("가" * (low // 2 + 100)))
        assert result.length_score == 1.0

    def test_gutted_summary_scores_low(self):
        """The regression that prompted this rubric: a summary cut to ~2,500
        characters lost technical depth the clip limit never required."""
        result = evaluate_summary("a", _summary("가" * 600))
        assert result.length_score < 0.7


class TestLede:
    def test_single_concrete_sentence_in_band_scores_full(self):
        lede = "prefill과 decode를 분리해 동시성이 높을 때 토큰당 지연을 22~66% 줄였습니다."
        result = evaluate_summary("a", _summary("가" * 2200), one_liner=lede)
        assert result.lede_score == 1.0

    def test_meta_framing_is_penalised(self):
        lede = "이 글은 prefill과 decode 분리 아키텍처를 다루며 성능 개선을 설명합니다."
        assert evaluate_summary("a", _summary("가"), one_liner=lede).lede_score < 1.0

    def test_multiple_sentences_penalised(self):
        lede = "지연을 22% 줄였습니다. 비용도 절감됩니다. 그리고 확장성이 좋아집니다."
        assert evaluate_summary("a", _summary("가"), one_liner=lede).lede_score < 1.0

    def test_missing_lede_scores_zero(self):
        assert evaluate_summary("a", _summary("가"), one_liner="").lede_score == 0.0


class TestCaptions:
    def test_uncaptioned_image_penalised(self):
        html = _summary('<img src="https://x/y.png">')
        assert evaluate_summary("a", html).caption_score == 0.0

    def test_captioned_image_scores_full(self):
        html = _summary(
            '<figure><img src="https://x/y.png"><figcaption>구조도</figcaption></figure>'
        )
        assert evaluate_summary("a", html).caption_score == 1.0

    def test_no_images_is_not_a_defect(self):
        assert evaluate_summary("a", _summary("본문입니다.")).caption_score == 1.0


class TestStructurePenalties:
    def test_absence_filler_penalised(self):
        html = _summary("원문에 관련 정보가 없습니다.")
        assert evaluate_summary("a", html).structure_score < 1.0

    def test_subsection_headers_penalised(self):
        html = _summary("본문") + "<h4>소제목</h4>"
        assert evaluate_summary("a", html).subsection_headers == 1


class TestReport:
    def test_weakest_dimensions_are_surfaced_for_tuning(self):
        """The report's job is to say WHICH prompt rule to work on next."""
        bad = evaluate_summary("bad", _summary("짧다."), one_liner="")
        report = QualityReport(results=[bad])
        weakest = dict(report.weakest(3))
        assert "lede" in weakest
        assert report.overall < 1.0

    def test_empty_report_is_safe(self):
        report = QualityReport(results=[])
        assert report.overall == 0.0
        assert report.mean("length") == 0.0


class TestRubricMatchesThePrompt:
    """The rubric's targets and the prompt's stated numbers must not drift."""

    def test_length_band_matches_the_korean_prompt(self):
        from app.src.constants import Language
        from app.src.prompts import SummarizationPrompt

        template = SummarizationPrompt._human_prompt_template[Language.KO]
        low, high = KO_LENGTH_BAND
        assert f"{low:,}~{high:,}자" in template

    def test_lede_band_matches_the_korean_prompt(self):
        from app.src.constants import Language
        from app.src.prompts import SummarizationPrompt
        from app.src.quality_metrics import KO_LEDE_BAND

        template = SummarizationPrompt._human_prompt_template[Language.KO]
        low, high = KO_LEDE_BAND
        assert f"{low}~{high}자" in template

    @pytest.mark.parametrize("phrase", ["측정값", "표", "코드"])
    def test_specificity_rules_are_actually_in_the_prompt(self, phrase):
        from app.src.constants import Language
        from app.src.prompts import SummarizationPrompt

        assert phrase in SummarizationPrompt._human_prompt_template[Language.KO]

"""Unit tests for summarizer output normalization: tags/urls parsing and the
markdown-to-HTML conversion in SummaryOutput. No Bedrock access."""

from __future__ import annotations

from app.src.summarizer import (
    SummaryOutput,
    _normalize_tags,
    _normalize_urls,
    _postprocess_summary,
)


class TestPostprocessSummary:
    def test_terminal_colon_before_markup_becomes_period(self):
        # A sentence-ending syllable + colon at a clause boundary (colon followed
        # by a closing tag) is a mistake and is corrected to a period.
        assert (
            _postprocess_summary("<p>이것은 동작합니다:</p>")
            == "<p>이것은 동작합니다.</p>"
        )

    def test_terminal_colon_at_end_of_string_becomes_period(self):
        assert _postprocess_summary("문장이 끝납니다:") == "문장이 끝납니다."

    def test_introducing_colon_is_preserved(self):
        # Colon followed by text introduces a list/example and must NOT be
        # rewritten (the bug a blind "다:"->"다." replace used to cause).
        text = "<p>결과는 다음과 같습니다: 첫째, 둘째</p>"
        assert _postprocess_summary(text) == text

    def test_ratio_colon_is_preserved(self):
        assert _postprocess_summary("비율은 3:1입니다.") == "비율은 3:1입니다."

    def test_block_list_introducing_colon_preserved(self):
        # After markdown, a sentence introducing a BULLETED list renders as
        # "<p>...습니다:</p>\n<ul>...". The colon is at a tag boundary yet still
        # legitimately introduces the list, so it must NOT become a period.
        text = "<p>주요 구성 요소는 다음과 같습니다:</p>\n<ul>\n<li>A</li>\n</ul>"
        assert _postprocess_summary(text) == text

    def test_br_list_introducing_colon_preserved(self):
        # Same under nl2br: "습니다:<br>\n<ul>".
        text = "같습니다:<br>\n<ul><li>x</li></ul>"
        assert _postprocess_summary(text) == text

    def test_terminal_colon_before_paragraph_still_converted(self):
        # A genuine terminal colon before the NEXT paragraph (not a list) is
        # still corrected to a period.
        assert (
            _postprocess_summary("<p>동작합니다:</p>\n<p>다음</p>")
            == "<p>동작합니다.</p>\n<p>다음</p>"
        )

    def test_url_host_normalized(self):
        assert (
            _postprocess_summary("see magazine.sebastianraschka.com")
            == "see sebastianraschka.com"
        )

    def test_no_change_when_clean(self):
        assert _postprocess_summary("<p>clean</p>") == "<p>clean</p>"


class TestNormalizeTags:
    def test_comma_separated_string(self):
        assert _normalize_tags("RAG, Vector DB, Embeddings") == [
            "RAG",
            "Vector DB",
            "Embeddings",
        ]

    def test_list_of_strings(self):
        assert _normalize_tags(["a", "b"]) == ["a", "b"]

    def test_list_with_embedded_commas(self):
        assert _normalize_tags(["a, b", "c"]) == ["a", "b", "c"]

    def test_non_string_ignored(self):
        assert _normalize_tags(123) == []


class TestNormalizeUrls:
    def test_markdown_links_converted_to_anchors(self):
        out = _normalize_urls("[Paper](https://example.com/p), [Repo](https://x.com)")
        assert '<a href="https://example.com/p">Paper</a>' in out
        assert '<a href="https://x.com">Repo</a>' in out

    def test_plain_comma_separated(self):
        # Bare URLs are wrapped in sanitized anchors (consistent with the
        # markdown-link path) rather than rendered as non-clickable text.
        out = _normalize_urls("https://a.com, https://b.com")
        assert out == [
            '<a href="https://a.com">https://a.com</a>',
            '<a href="https://b.com">https://b.com</a>',
        ]

    def test_list_input(self):
        assert _normalize_urls(["https://a.com"]) == [
            '<a href="https://a.com">https://a.com</a>'
        ]

    def test_javascript_scheme_dropped(self):
        # XSS guard: non-allow-listed schemes never reach the rendered output.
        assert _normalize_urls("javascript:alert(1)") == []
        assert _normalize_urls("[Click](javascript:alert(1))") == []
        assert _normalize_urls(['<a href="javascript:alert(1)">x</a>']) == []

    def test_data_and_relative_dropped(self):
        assert _normalize_urls("data:text/html;base64,abcd") == []
        assert _normalize_urls("/relative/path") == []
        assert _normalize_urls("//scheme-relative.com") == []

    def test_existing_anchor_rescheme_checked_and_escaped(self):
        out = _normalize_urls(['<a href="https://ok.com">Safe</a>'])
        assert out == ['<a href="https://ok.com">Safe</a>']


class TestSummaryOutput:
    def test_markdown_summary_converted_to_html(self):
        out = SummaryOutput.model_validate(
            {"summary": "# Heading\n\nSome **bold** text.", "tags": [], "urls": []}
        )
        assert "<h1" in out.summary or "<strong>" in out.summary

    def test_tags_default_when_empty(self):
        out = SummaryOutput.model_validate({"summary": "x", "tags": [], "urls": []})
        assert out.tags == ["uncategorized"]

    def test_tags_deduped_and_capped(self):
        out = SummaryOutput.model_validate(
            {
                "summary": "x",
                "tags": "a, a, b, c, d, e, f, g",
                "urls": [],
            }
        )
        assert len(out.tags) <= 5
        # First-seen order preserved (model lists tags most-relevant-first),
        # duplicates removed, capped at MAX_TAGS.
        assert out.tags == ["a", "b", "c", "d", "e"]

    def test_html_replacements_applied(self):
        out = SummaryOutput.model_validate(
            {"summary": "```python\nprint(1)\n```", "tags": [], "urls": []}
        )
        # codehilite/fenced_code wraps code; the <code> replacement adds a class.
        assert "highlight" in out.summary

    def test_script_tag_stripped_from_summary(self):
        out = SummaryOutput.model_validate(
            {
                "summary": "<p>ok</p><script>alert(1)</script>",
                "tags": [],
                "urls": [],
            }
        )
        assert "<script" not in out.summary.lower()
        assert "alert(1)" not in out.summary
        assert "ok" in out.summary

    def test_event_handler_attr_stripped(self):
        out = SummaryOutput.model_validate(
            {"summary": '<p onclick="evil()">text</p>', "tags": [], "urls": []}
        )
        assert "onclick" not in out.summary.lower()
        assert "text" in out.summary

    def test_markdown_table_gets_bordered_class(self):
        # A markdown table (via the `tables` extension) must render with the
        # email-safe presentation class so it isn't borderless in mail clients.
        md = "| Metric | Value |\n| --- | --- |\n| speedup | 5x |"
        out = SummaryOutput.model_validate({"summary": md, "tags": [], "urls": []})
        assert '<table class="table table-bordered">' in out.summary

    def test_attributed_table_normalized_to_bordered_class(self):
        # Raw HTML tables the model emits with attributes (border/cellpadding)
        # used to slip past the old literal "<table>" string replace and render
        # unstyled. The sanitizer now forces the class on every table and strips
        # the stray presentational attributes.
        html = '<table border="0" cellpadding="0"><tr><td>a</td></tr></table>'
        out = SummaryOutput.model_validate({"summary": html, "tags": [], "urls": []})
        assert '<table class="table table-bordered">' in out.summary
        assert "border=" not in out.summary
        assert "cellpadding" not in out.summary

    def test_javascript_href_stripped_in_summary(self):
        out = SummaryOutput.model_validate(
            {
                "summary": '<p>see <a href="javascript:alert(1)">link</a></p>',
                "tags": [],
                "urls": [],
            }
        )
        assert "javascript:" not in out.summary.lower()
        # The text is preserved; only the unsafe href is removed.
        assert "link" in out.summary

    def test_safe_structural_tags_preserved(self):
        out = SummaryOutput.model_validate(
            {
                "summary": "<h3>Title</h3><p><strong>bold</strong> and <em>em</em></p>",
                "tags": [],
                "urls": [],
            }
        )
        assert "<h3>" in out.summary
        assert "<strong>" in out.summary
        assert "<em>" in out.summary

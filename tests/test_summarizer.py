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
    def test_korean_colon_to_period(self):
        assert (
            _postprocess_summary("이것은 중요합니다: 다음") == "이것은 중요합니다. 다음"
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
        out = _normalize_urls("https://a.com, https://b.com")
        assert out == ["https://a.com", "https://b.com"]

    def test_list_input(self):
        assert _normalize_urls(["https://a.com"]) == ["https://a.com"]


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
        assert out.tags == sorted(set(out.tags))

    def test_html_replacements_applied(self):
        out = SummaryOutput.model_validate(
            {"summary": "```python\nprint(1)\n```", "tags": [], "urls": []}
        )
        # codehilite/fenced_code wraps code; the <code> replacement adds a class.
        assert "highlight" in out.summary

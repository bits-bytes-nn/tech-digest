"""Unit tests for summarizer output normalization: tags/urls parsing and the
markdown-to-HTML conversion in SummaryOutput. No Bedrock access."""

from __future__ import annotations

from bs4 import BeautifulSoup

from app.domain.summary import (
    SummaryOutput,
    _normalize_tags,
    _normalize_urls,
    postprocess_summary,
)


class TestPostprocessSummary:
    def test_terminal_colon_before_markup_becomes_period(self):
        # A sentence-ending syllable + colon at a clause boundary (colon followed
        # by a closing tag) is a mistake and is corrected to a period.
        assert (
            postprocess_summary("<p>이것은 동작합니다:</p>")
            == "<p>이것은 동작합니다.</p>"
        )

    def test_terminal_colon_at_end_of_string_becomes_period(self):
        assert postprocess_summary("문장이 끝납니다:") == "문장이 끝납니다."

    def test_introducing_colon_is_preserved(self):
        # Colon followed by text introduces a list/example and must NOT be
        # rewritten (the bug a blind "다:"->"다." replace used to cause).
        text = "<p>결과는 다음과 같습니다: 첫째, 둘째</p>"
        assert postprocess_summary(text) == text

    def test_ratio_colon_is_preserved(self):
        assert postprocess_summary("비율은 3:1입니다.") == "비율은 3:1입니다."

    def test_block_list_introducing_colon_preserved(self):
        # After markdown, a sentence introducing a BULLETED list renders as
        # "<p>...습니다:</p>\n<ul>...". The colon is at a tag boundary yet still
        # legitimately introduces the list, so it must NOT become a period.
        text = "<p>주요 구성 요소는 다음과 같습니다:</p>\n<ul>\n<li>A</li>\n</ul>"
        assert postprocess_summary(text) == text

    def test_br_list_introducing_colon_preserved(self):
        # Same under nl2br: "습니다:<br>\n<ul>".
        text = "같습니다:<br>\n<ul><li>x</li></ul>"
        assert postprocess_summary(text) == text

    def test_terminal_colon_before_paragraph_still_converted(self):
        # A genuine terminal colon before the NEXT paragraph (not a list) is
        # still corrected to a period.
        assert (
            postprocess_summary("<p>동작합니다:</p>\n<p>다음</p>")
            == "<p>동작합니다.</p>\n<p>다음</p>"
        )

    def test_host_is_not_rewritten(self):
        """Host "normalization" was removed, so a link is passed through intact.

        It rewrote magazine.sebastianraschka -> sebastianraschka across the whole
        document, which turns a working Substack URL into a 404 on the author's own
        site (the two use different path layouts) and also fired on visible prose.
        """
        text = "see https://magazine.sebastianraschka.com/p/x for details"
        assert postprocess_summary(text) == text

    def test_no_change_when_clean(self):
        assert postprocess_summary("<p>clean</p>") == "<p>clean</p>"


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
        assert "<h1" in out.summary
        assert "<strong>bold</strong>" in out.summary

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

    def test_fenced_code_renders_as_plain_pre_code(self):
        # No class: nothing styles one (pre/code are styled by element), and
        # codehilite's markup changed with whether Pygments was installed.
        out = SummaryOutput.model_validate(
            {"summary": "```python\nprint(1)\n```", "tags": [], "urls": []}
        )
        assert out.summary == "<pre><code>print(1)\n</code></pre>"

    def test_model_classes_on_code_are_dropped(self):
        html = '<pre class="highlight"><code class="language-py">x = 1</code></pre>'
        out = SummaryOutput.model_validate({"summary": html, "tags": [], "urls": []})
        assert out.summary == "<pre><code>x = 1</code></pre>"

    def test_footnote_syntax_leaves_no_dead_links(self):
        # footnotes emitted "#fn:1" links, which fail the scheme check and
        # reached the reader as <a> tags without an href.
        out = SummaryOutput.model_validate(
            {"summary": "Text[^1].\n\n[^1]: The note.", "tags": [], "urls": []}
        )
        assert "<a" not in out.summary

    def test_definition_list_syntax_keeps_its_line_break(self):
        # def_list emitted <dl>, which the sanitizer unwraps: the term and its
        # definition then ran together on one line of bare text.
        out = SummaryOutput.model_validate(
            {"summary": "Term\n: Definition", "tags": [], "urls": []}
        )
        assert out.summary.startswith("<p>Term<br/>")

    def test_summary_image_is_sized_for_the_column(self):
        # Outlook for Windows ignores max-width: the width attribute is what
        # keeps a 2,000px chart inside the 800px column.
        html = (
            '<figure><img src="https://cdn.example.com/chart.png" alt="chart" '
            'width="2369" style="position:fixed"></figure>'
        )
        out = SummaryOutput.model_validate({"summary": html, "tags": [], "urls": []})
        img = BeautifulSoup(out.summary, "html.parser").find("img")
        assert img["width"] == "712"
        assert img["style"] == "width:100%;max-width:712px;height:auto"
        assert img["src"] == "https://cdn.example.com/chart.png"
        assert img["alt"] == "chart"

    def test_an_image_whose_src_was_dropped_is_removed(self):
        html = '<p>x</p><img src="javascript:alert(1)" alt="bad">'
        out = SummaryOutput.model_validate({"summary": html, "tags": [], "urls": []})
        assert "<img" not in out.summary

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
        assert '<table class="table-bordered">' in out.summary

    def test_attributed_table_normalized_to_bordered_class(self):
        # Raw HTML tables the model emits with attributes (border/cellpadding)
        # used to slip past the old literal "<table>" string replace and render
        # unstyled. The sanitizer now forces the class on every table and strips
        # the stray presentational attributes.
        html = '<table border="0" cellpadding="0"><tr><td>a</td></tr></table>'
        out = SummaryOutput.model_validate({"summary": html, "tags": [], "urls": []})
        assert '<table class="table-bordered">' in out.summary
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


class TestImageCaptions:
    """Images used to be dropped into the card with no context. The summary now
    wraps each one in a <figure> with a <figcaption>, so the sanitizer has to let
    those two tags through and the provenance guard has to keep working inside
    them."""

    def test_figure_and_figcaption_survive_sanitizing(self):
        from app.domain.summary import SummaryOutput

        out = SummaryOutput.model_validate(
            {
                "summary": (
                    '<figure><img src="https://cdn.example.com/a.png" alt="arch">'
                    "<figcaption>Prefill and decode run on separate pools."
                    "</figcaption></figure>"
                ),
                "tags": [],
                "urls": [],
            }
        )
        assert "<figure>" in out.summary
        assert "Prefill and decode run on separate pools." in out.summary

    def test_an_image_the_parser_resolved_is_kept(self):
        """The source may carry a relative src that only ``Post.images`` holds
        in its absolute form; that is still the article's own image."""
        from app.domain.summary import strip_unsourced_images

        url = "https://blog.example.com/img/arch.png"
        cleaned, dropped = strip_unsourced_images(
            f'<p>Body</p><img src="{url}">',
            source_html='<p>Body</p><img src="/img/arch.png">',
            known_images=[url],
        )
        assert dropped == [] and url in cleaned

    def test_an_image_without_a_src_is_dropped(self):
        from app.domain.summary import strip_unsourced_images

        cleaned, dropped = strip_unsourced_images(
            "<p>Body</p><img alt='x'>", source_html="<p>Body</p>", known_images=[]
        )
        assert dropped == ["(no src)"] and "<img" not in cleaned

    def test_unsourced_image_is_dropped_but_its_caption_context_remains(self):
        """The provenance guard removes the <img>; the surrounding figure/caption
        must not take the rest of the summary down with it."""
        from app.domain.summary import strip_unsourced_images

        cleaned, dropped = strip_unsourced_images(
            '<p>Body</p><figure><img src="https://cdn.example.com/invented.png">'
            "<figcaption>Caption</figcaption></figure>",
            source_html="<p>Body</p>",
            known_images=[],
        )
        assert dropped and "invented.png" not in cleaned
        assert "<p>Body</p>" in cleaned

    def test_prompts_ask_for_a_caption_in_both_languages(self):
        from app.application.prompts.prompts import SummarizationPrompt
        from app.domain.constants import Language

        for language in (Language.EN, Language.KO):
            template = SummarizationPrompt._human_prompt_template[language]
            assert "figcaption" in template, language

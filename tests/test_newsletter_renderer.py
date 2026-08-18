"""Unit tests for the newsletter renderer: date validation, pydantic models,
and end-to-end Jinja2 rendering against the real templates. No Selenium."""

from __future__ import annotations

import pytest

from app.src.constants import Language
from app.src.newsletter_renderer import (
    GMAIL_CLIP_BYTES,
    Article,
    BuildConfiguration,
    Footer,
    Header,
    NewsletterData,
    NewsletterRenderer,
    Section,
    collapse_html_whitespace,
    estimate_reading_minutes,
    validate_date,
)


class TestValidateDate:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("2026-05-30", "2026-05-30"),
            ("Fri, 30 May 2026 10:00:00 +0000", "2026-05-30"),
            ("2026-05-30T10:00:00", "2026-05-30"),
        ],
    )
    def test_known_formats(self, raw, expected):
        assert validate_date(raw) == expected

    def test_unparseable_falls_back_to_today(self):
        # Should not raise; returns *some* YYYY-MM-DD string.
        out = validate_date("garbage")
        assert len(out) == 10 and out[4] == "-"


class TestArticleModel:
    def test_valid(self, sample_article_data):
        article = Article.model_validate(sample_article_data)
        assert article.score == 0.84
        assert article.published_date == "2026-05-30"

    def test_score_bounds_enforced(self, sample_article_data):
        sample_article_data["score"] = 1.5
        with pytest.raises(ValueError):
            Article.model_validate(sample_article_data)

    def test_empty_title_rejected(self, sample_article_data):
        sample_article_data["title"] = ""
        with pytest.raises(ValueError):
            Article.model_validate(sample_article_data)

    def test_unsafe_link_scheme_blanked(self, sample_article_data):
        # A javascript:/data: link from scraped content must not be rendered as
        # a clickable href; it is blanked rather than raising.
        sample_article_data["link"] = "javascript:alert(1)"
        article = Article.model_validate(sample_article_data)
        assert article.link == ""

    def test_safe_link_scheme_preserved(self, sample_article_data):
        sample_article_data["link"] = "https://example.com/post"
        article = Article.model_validate(sample_article_data)
        assert article.link == "https://example.com/post"


class TestSectionGreetingEscaping:
    """The greeting is plain text rendered with Jinja `| safe`, so the Section
    model HTML-escapes it as a trust boundary (parallel to summary sanitizing)."""

    def test_script_injection_neutralized(self):
        section = Section(introduction="Hi <script>alert(1)</script> there")
        assert "<script>" not in section.introduction
        assert "&lt;script&gt;" in section.introduction

    def test_plain_text_without_metachars_is_unchanged(self):
        text = "Hello friends. This week the highlights are excellent."
        # No HTML metacharacters -> escaping leaves legitimate output intact.
        assert Section(introduction=text).introduction == text


class TestCollapseHtmlWhitespace:
    """Template indentation is ~10% of the message size, which counts against
    the mail-client clip budget — but collapsing it must never eat a meaningful
    inter-word space or touch preformatted code."""

    def test_indentation_collapsed_to_single_space(self):
        html = "<td>\n      <table>\n        <tr>\n"
        assert collapse_html_whitespace(html) == "<td> <table> <tr> "

    def test_inline_word_spacing_preserved(self):
        # A newline between two inline tags IS a word space in the rendered
        # prose; removing it would silently join two Korean words.
        html = "<p><em>분리</em>\n<em>구조</em></p>"
        assert collapse_html_whitespace(html) == "<p><em>분리</em> <em>구조</em></p>"

    def test_pre_block_left_byte_identical(self):
        html = '<div>\n  <pre class="x">def f():\n    return 1\n</pre>\n</div>'
        out = collapse_html_whitespace(html)
        assert "def f():\n    return 1\n" in out
        assert out.startswith("<div> <pre")

    def test_shrinks_real_template_output(self, templates_dir, sample_article_data):
        renderer = NewsletterRenderer(templates_dir)
        data = NewsletterData(
            header=Header(
                title="T", description="d", thumbnail="p.png", publish_date="2026-06-01"
            ),
            section=Section(introduction="hi"),
            articles=[Article.model_validate(sample_article_data)],
            footer=Footer(title="f"),
        )
        raw = renderer.newsletter_template.render(data=data.model_dump(mode="json"))
        assert len(collapse_html_whitespace(raw)) < len(raw)


class TestReadingTime:
    def test_korean_counted_in_characters(self):
        assert estimate_reading_minutes("<p>" + "가" * 900 + "</p>", Language.KO) == 2

    def test_english_counted_in_words(self):
        assert (
            estimate_reading_minutes("<p>" + "word " * 400 + "</p>", Language.EN) == 2
        )

    def test_never_zero(self):
        assert estimate_reading_minutes("", Language.KO) == 1
        assert estimate_reading_minutes("<p>짧다</p>", Language.KO) == 1


class TestNewsletterRendering:
    def _build_data(
        self, article_data, language: Language = Language.KO
    ) -> NewsletterData:
        return NewsletterData(
            header=Header(
                title="Weekly AI Tech Blog Digest",
                description="desc",
                thumbnail="peccy.png",
                publish_date="2026-06-01",
            ),
            section=Section(introduction="Hello friends!"),
            articles=[Article.model_validate(article_data)],
            footer=Footer(title="Thanks for reading."),
            language=language,
        )

    def test_language_attribute_follows_the_issue_language(
        self, templates_dir, sample_article_data
    ):
        """Regression: the template read ``data.language``, which NewsletterData
        never had, so an English issue still declared lang="ko"."""
        renderer = NewsletterRenderer(templates_dir)
        html = renderer.render_newsletter(
            self._build_data(sample_article_data, Language.EN)
        )
        assert 'lang="en"' in html
        assert "Language.EN" not in html  # the enum must serialize to its value

    def test_chrome_localized_for_korean(self, templates_dir, sample_article_data):
        renderer = NewsletterRenderer(templates_dir)
        html = renderer.render_newsletter(
            self._build_data(sample_article_data, Language.KO)
        )
        assert "원문 보기" in html
        assert "읽는 시간" in html
        assert "Published on" not in html
        assert "Additional resources for reference" not in html

    def test_chrome_localized_for_english(self, templates_dir, sample_article_data):
        renderer = NewsletterRenderer(templates_dir)
        html = renderer.render_newsletter(
            self._build_data(sample_article_data, Language.EN)
        )
        assert "Read the original" in html
        assert "min read" in html
        assert "원문 보기" not in html

    def test_one_liner_rendered_and_escaped(self, templates_dir, sample_article_data):
        sample_article_data["one_liner"] = "Cuts latency 40% <not-a-tag>"
        renderer = NewsletterRenderer(templates_dir)
        html = renderer.render_newsletter(self._build_data(sample_article_data))
        assert "Cuts latency 40%" in html
        # Rendered without |safe, so any stray markup stays inert text.
        assert "<not-a-tag>" not in html
        assert "&lt;not-a-tag&gt;" in html

    def test_absent_one_liner_renders_no_empty_lede(
        self, templates_dir, sample_article_data
    ):
        sample_article_data.pop("one_liner", None)
        renderer = NewsletterRenderer(templates_dir)
        html = renderer.render_newsletter(self._build_data(sample_article_data))
        assert 'class="lede"' not in html

    def test_renders_full_newsletter(self, templates_dir, sample_article_data):
        renderer = NewsletterRenderer(templates_dir)
        html = renderer.render_newsletter(self._build_data(sample_article_data))
        assert "Weekly AI Tech Blog Digest" in html
        assert sample_article_data["title"] in html
        assert "#Mixture Of Experts" in html or "Mixture Of Experts" in html
        assert html.strip().startswith("<!DOCTYPE")

    def test_summary_html_not_escaped(self, templates_dir, sample_article_data):
        renderer = NewsletterRenderer(templates_dir)
        html = renderer.render_newsletter(self._build_data(sample_article_data))
        # summary is marked |safe — the <h3> must survive as a real tag.
        assert "<h3>Why This Matters</h3>" in html

    def test_renders_single_article(self, templates_dir, sample_article_data):
        renderer = NewsletterRenderer(templates_dir)
        html = renderer.render_article(Article.model_validate(sample_article_data))
        assert sample_article_data["title"] in html

    def test_single_article_has_descriptive_alt(
        self, templates_dir, sample_article_data
    ):
        sample_article_data["source"] = "openai"
        renderer = NewsletterRenderer(templates_dir)
        html = renderer.render_article(Article.model_validate(sample_article_data))
        # Standalone-article thumbnail uses the source label, not a generic alt.
        assert 'alt="Openai logo"' in html
        assert 'alt="Thumbnail"' not in html

    def test_score_badge_rendered(self, templates_dir, sample_article_data):
        renderer = NewsletterRenderer(templates_dir)
        html = renderer.render_newsletter(self._build_data(sample_article_data))
        assert "★ 0.84" in html

    def test_dark_mode_and_accessibility(self, templates_dir, sample_article_data):
        renderer = NewsletterRenderer(templates_dir)
        html = renderer.render_newsletter(self._build_data(sample_article_data))
        assert "prefers-color-scheme: dark" in html
        assert 'lang="ko"' in html
        assert 'role="article"' in html

    def test_dark_mode_darkens_content_cells_and_table_headers(
        self, templates_dir, sample_article_data
    ):
        # Regression guard: the dark-mode block must darken the content-cell
        # backgrounds (.card-body/.td-padding) and the markdown table header, or
        # light text renders on light cells and is unreadable in dark mode.
        renderer = NewsletterRenderer(templates_dir)
        html = renderer.render_newsletter(self._build_data(sample_article_data))
        dark_block = html.split("prefers-color-scheme: dark", 1)[1]
        assert ".td-padding" in dark_block
        assert "table.table-bordered th" in dark_block
        # The article title must NOT keep an inline `!important` color that would
        # defeat the dark-mode rule.
        assert "color: #212121 !important" not in html

    def test_content_tables_use_auto_layout_and_wrap(
        self, templates_dir, sample_article_data
    ):
        renderer = NewsletterRenderer(templates_dir)
        html = renderer.render_newsletter(self._build_data(sample_article_data))
        assert "table-layout: auto" in html
        assert "overflow-wrap: anywhere" in html

    def test_source_alt_text(self, templates_dir, sample_article_data):
        sample_article_data["source"] = "openai"
        renderer = NewsletterRenderer(templates_dir)
        html = renderer.render_newsletter(self._build_data(sample_article_data))
        assert "Openai logo" in html


class TestClipBudget:
    """Eight of the eighteen previously published issues exceeded Gmail's clip
    threshold, which silently hid the trailing cards and the footer from readers.
    This ties the prompt's per-summary length budget to the template's markup
    weight, so a regression in either surfaces here instead of in an inbox."""

    # Upper bound of the Korean summary budget stated in SummarizationPrompt.
    KO_SUMMARY_CHARS = 2300

    def _issue(self, article_count: int) -> NewsletterData:
        body = (
            "<h3>📌 왜 이 아티클에 주목해야 하나요?</h3><p>"
            + "가" * self.KO_SUMMARY_CHARS
            + "</p>"
        )
        articles = [
            Article.model_validate(
                {
                    "title": f"기술 아티클 제목 {i}",
                    "link": f"https://example.com/{i}",
                    "published_date": "2026-06-01",
                    "thumbnail": "https://cdn.example.com/logo.png",
                    "summary": body,
                    "one_liner": "핵심을 한 문장으로 요약한 리드 문장입니다.",
                    "reading_minutes": 5,
                    "source": "aws",
                    "tags": ["Tag A", "Tag B", "Tag C", "Tag D", "Tag E"],
                    "urls": [f'<a href="https://example.com/ref{i}">Reference {i}</a>'],
                    "score": 0.85,
                }
            )
            for i in range(article_count)
        ]
        return NewsletterData(
            header=Header(
                title="Weekly AI Tech Blog Digest",
                description="d" * 200,
                thumbnail="https://cdn.example.com/peccy.png",
                publish_date="2026-06-01",
            ),
            section=Section(introduction="안녕 친구들! 난 Peccy야 😎 " + "글" * 120),
            articles=articles,
            footer=Footer(title="구독해 주셔서 감사합니다."),
            language=Language.KO,
        )

    @pytest.mark.parametrize("article_count", [3, 5])
    def test_issue_fits_within_clip_budget(self, templates_dir, article_count):
        renderer = NewsletterRenderer(templates_dir)
        html = renderer.render_newsletter(self._issue(article_count))
        size = len(html.encode("utf-8"))
        assert size <= GMAIL_CLIP_BYTES, (
            f"{article_count}-article issue is {size / 1024:.1f} KB, over the "
            f"{GMAIL_CLIP_BYTES / 1024:.0f} KB clip budget — Gmail will truncate it."
        )

    def test_oversized_issue_is_reported(
        self, templates_dir, caplog, propagating_logger
    ):
        renderer = NewsletterRenderer(templates_dir)
        data = self._issue(3)
        # Ten times the budgeted length: the build must say so rather than
        # emitting a message that will be silently cut in the reader's client.
        for article in data.articles:
            article.summary = "<p>" + "가" * (self.KO_SUMMARY_CHARS * 10) + "</p>"
        with caplog.at_level("WARNING"):
            renderer.render_newsletter(data)
        assert any("clip budget" in r.message for r in caplog.records)


class TestStandaloneArticlePage:
    """The standalone page used to carry its own divergent copy of the card and
    stylesheet, so dark mode and the accessibility work never reached it."""

    def test_shares_the_newsletter_styling(self, templates_dir, sample_article_data):
        renderer = NewsletterRenderer(templates_dir)
        html = renderer.render_article(
            Article.model_validate(sample_article_data), Language.KO
        )
        assert "prefers-color-scheme: dark" in html
        assert 'lang="ko"' in html
        assert 'role="article"' in html
        assert "table-layout: auto" in html

    def test_labels_follow_the_requested_language(
        self, templates_dir, sample_article_data
    ):
        renderer = NewsletterRenderer(templates_dir)
        assert "원문 보기" in renderer.render_article(
            Article.model_validate(sample_article_data), Language.KO
        )
        assert "Read the original" in renderer.render_article(
            Article.model_validate(sample_article_data), Language.EN
        )


class TestArticleSourceLabel:
    def test_known_source(self, sample_article_data):
        sample_article_data["source"] = "pinterest"
        article = Article.model_validate(sample_article_data)
        assert article.source_label == "Pinterest"

    def test_unknown_source_blank(self, sample_article_data):
        sample_article_data["source"] = "unknown"
        article = Article.model_validate(sample_article_data)
        assert article.source_label == ""


class TestBuildConfiguration:
    def test_defaults(self):
        config = BuildConfiguration()
        assert config.language == Language.KO
        assert config.stage == "dev"
        assert config.save_individual_articles is False

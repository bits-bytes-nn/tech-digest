"""Unit tests for the newsletter renderer: date validation, pydantic models,
and end-to-end Jinja2 rendering against the real templates. No Selenium."""

from __future__ import annotations

import pytest

from app.src.constants import Language
from app.src.newsletter_renderer import (
    Article,
    BuildConfiguration,
    Footer,
    Header,
    NewsletterData,
    NewsletterRenderer,
    Section,
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


class TestNewsletterRendering:
    def _build_data(self, article_data) -> NewsletterData:
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
        )

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

    def test_source_alt_text(self, templates_dir, sample_article_data):
        sample_article_data["source"] = "openai"
        renderer = NewsletterRenderer(templates_dir)
        html = renderer.render_newsletter(self._build_data(sample_article_data))
        assert "Openai logo" in html


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

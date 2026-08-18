"""NewsletterBuilder integration: how the digest's article JSON becomes an
ordered, localized issue. Exercises the real templates against real files.
"""

from __future__ import annotations

import json

import pytest

from app.src.constants import Language
from app.src.newsletter_renderer import BuildConfiguration, NewsletterBuilder

LOGOS = {
    "aws": "https://cdn.example.com/aws.png",
    "meta": "https://cdn.example.com/meta.png",
    "google": "https://cdn.example.com/google.png",
    "unknown": "https://cdn.example.com/unknown.png",
}


def _write_post(directory, *, title, source, score, published, summary=None):
    payload = {
        "title": title,
        "link": f"https://example.com/{title.replace(' ', '-')}",
        "published_date": published,
        "summary": summary or f"<h3>📌 개요</h3><p>{'내용 ' * 80}</p>",
        "one_liner": f"{title} 의 핵심 한 줄입니다.",
        "source": source,
        "tags": ["Tag"],
        "urls": [],
        "score": score,
    }
    path = directory / f"{source}-{title.replace(' ', '_')}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def digest_dirs(tmp_path):
    inputs = tmp_path / "inputs"
    (inputs / "2026-06-01").mkdir(parents=True)
    outputs = tmp_path / "outputs"
    return inputs, outputs


def _builder(inputs, outputs, templates_dir):
    return NewsletterBuilder(inputs, outputs, templates_dir, logos=LOGOS)


def _config(**overrides) -> BuildConfiguration:
    base = {
        "stage": "dev",
        "date_suffix": "2026-06-01",
        "language": Language.KO,
        "header_title": "Weekly AI Tech Blog Digest",
        "header_description": "desc",
        "header_thumbnail": "https://cdn.example.com/peccy.png",
        "first_section_intro": "안녕 친구들! 난 Peccy야 😎",
        "footer_title": "감사합니다.",
    }
    return BuildConfiguration(**(base | overrides))


class TestArticleOrdering:
    """The filtering stage pays to rank posts by relevance; presentation must not
    throw that ranking away. Sorting by publication date did exactly that — a
    weekly window puts almost every post on the same day or two, so ties fell
    back to glob (filename) order and routinely buried the top-scored piece."""

    def test_highest_score_leads_the_issue(self, digest_dirs, templates_dir):
        inputs, outputs = digest_dirs
        day = inputs / "2026-06-01"
        # Deliberately written so alphabetical filename order (aws < google <
        # meta) disagrees with relevance order.
        _write_post(
            day, title="Weak AWS Post", source="aws", score=0.70, published="2026-06-01"
        )
        _write_post(
            day,
            title="Mid Google Post",
            source="google",
            score=0.80,
            published="2026-06-01",
        )
        _write_post(
            day,
            title="Best Meta Post",
            source="meta",
            score=0.90,
            published="2026-06-01",
        )

        builder = _builder(inputs, outputs, templates_dir)
        articles = builder._load_articles("2026-06-01", Language.KO)
        assert [a.title for a in articles] == [
            "Best Meta Post",
            "Mid Google Post",
            "Weak AWS Post",
        ]

    def test_recency_breaks_score_ties(self, digest_dirs, templates_dir):
        inputs, outputs = digest_dirs
        day = inputs / "2026-06-01"
        _write_post(
            day, title="Older", source="aws", score=0.85, published="2026-05-28"
        )
        _write_post(
            day, title="Newer", source="aws", score=0.85, published="2026-06-01"
        )
        builder = _builder(inputs, outputs, templates_dir)
        articles = builder._load_articles("2026-06-01", Language.KO)
        assert [a.title for a in articles] == ["Newer", "Older"]

    def test_rendered_issue_shows_the_lead_article_first(
        self, digest_dirs, templates_dir
    ):
        inputs, outputs = digest_dirs
        day = inputs / "2026-06-01"
        _write_post(
            day, title="Weak AWS Post", source="aws", score=0.70, published="2026-06-01"
        )
        _write_post(
            day,
            title="Best Meta Post",
            source="meta",
            score=0.90,
            published="2026-06-01",
        )
        builder = _builder(inputs, outputs, templates_dir)
        path, _ = builder.build(_config())
        html = path.read_text(encoding="utf-8")
        assert html.index("Best Meta Post") < html.index("Weak AWS Post")


class TestBuildOutputs:
    def test_reading_time_injected_per_language(self, digest_dirs, templates_dir):
        inputs, outputs = digest_dirs
        day = inputs / "2026-06-01"
        _write_post(
            day,
            title="A Post",
            source="aws",
            score=0.85,
            published="2026-06-01",
            summary="<p>" + "가" * 900 + "</p>",
        )
        builder = _builder(inputs, outputs, templates_dir)
        ko = builder._load_articles("2026-06-01", Language.KO)
        assert ko[0].reading_minutes == 2

    def test_filename_carries_stage_date_and_language(self, digest_dirs, templates_dir):
        inputs, outputs = digest_dirs
        _write_post(
            inputs / "2026-06-01",
            title="A Post",
            source="aws",
            score=0.85,
            published="2026-06-01",
        )
        builder = _builder(inputs, outputs, templates_dir)
        path, _ = builder.build(_config(stage="prod", language=Language.EN))
        assert path.name == "newsletter-prod-2026-06-01-en.html"

    def test_missing_input_dir_yields_no_articles(self, digest_dirs, templates_dir):
        inputs, outputs = digest_dirs
        builder = _builder(inputs, outputs, templates_dir)
        assert builder._load_articles("2099-01-01", Language.KO) == []

    def test_unparseable_article_file_skipped_not_fatal(
        self, digest_dirs, templates_dir
    ):
        inputs, outputs = digest_dirs
        day = inputs / "2026-06-01"
        _write_post(day, title="Good", source="aws", score=0.85, published="2026-06-01")
        (day / "broken.json").write_text("{not json", encoding="utf-8")
        builder = _builder(inputs, outputs, templates_dir)
        articles = builder._load_articles("2026-06-01", Language.KO)
        assert [a.title for a in articles] == ["Good"]

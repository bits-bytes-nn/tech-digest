"""Golden parse tests for the per-site HTML scrapers.

These scrapers carry the most fragile heuristics in the codebase (CSS selectors,
title-length bounds, date regex, parent-walk depth). They previously had ZERO
unit coverage — only routing/source-detection was tested. Here we feed each
scraper a saved, representative HTML fixture and pin the parsed output, so a
site-layout change or a selector regression is caught offline.

Network is fully stubbed: ``_fetch_page`` returns the fixture soup and
``_make_robust_request`` is neutralized so the full-article scrape that
``Post.from_entry`` triggers never hits the wire.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from app.src import feed_parser
from app.src.feed_parser import (
    AnthropicBlogScraper,
    GoogleBlogScraper,
    LinkedInBlogScraper,
    MetaAIBlogScraper,
    QwenBlogScraper,
    XAIBlogScraper,
)

FIXTURES = Path(__file__).parent / "fixtures" / "scrapers"


def _load(name: str) -> BeautifulSoup:
    return BeautifulSoup((FIXTURES / name).read_text(encoding="utf-8"), "html.parser")


@pytest.fixture
def window() -> tuple[datetime, datetime]:
    """A window covering late-May 2026 (the recent fixture posts), excluding the
    deliberately-ancient entries in each fixture."""
    start = datetime(2026, 5, 20, tzinfo=UTC)
    end = datetime(2026, 5, 31, 23, 59, 59, tzinfo=UTC)
    return start, end


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    # The scrapers build Posts via Post.from_entry, which scrapes the full
    # article when the feed body is thin. Neutralize the HTTP layer so no
    # request is made and content stays empty.
    monkeypatch.setattr(feed_parser, "_make_robust_request", lambda url: None)


def _bind(scraper_cls, fixture: str, page_url: str, source: str, monkeypatch):
    scraper = scraper_cls(page_url=page_url, source=source)
    soup = _load(fixture)
    monkeypatch.setattr(scraper, "_fetch_page", lambda: soup)
    return scraper


class TestGoogleScraper:
    def test_parses_titles_links_dates(self, monkeypatch, window):
        scraper = _bind(
            GoogleBlogScraper,
            "google.html",
            "https://research.google/blog",
            "google",
            monkeypatch,
        )
        posts = scraper.fetch(*window)
        titles = {p.title for p in posts}
        assert "Scaling Transformers Efficiently" in titles
        assert "A Post With An Absolute URL" in titles
        # Out-of-window 2020 post excluded.
        assert "An Old Post Outside The Window" not in titles

    def test_relative_href_resolved_absolute(self, monkeypatch, window):
        scraper = _bind(
            GoogleBlogScraper,
            "google.html",
            "https://research.google/blog",
            "google",
            monkeypatch,
        )
        posts = scraper.fetch(*window)
        links = {p.link for p in posts}
        assert "https://research.google/blog/scaling-transformers-efficiently/" in links


class TestLinkedInScraper:
    def test_parses_items_and_window(self, monkeypatch, window):
        scraper = _bind(
            LinkedInBlogScraper,
            "linkedin.html",
            "https://www.linkedin.com/blog/engineering",
            "linkedin",
            monkeypatch,
        )
        posts = scraper.fetch(*window)
        titles = {p.title for p in posts}
        assert "Real-Time Ranking At Scale" in titles
        assert "Building A Feature Store" in titles
        assert "Ancient Post Outside Window" not in titles

    def test_relative_href_resolved_absolute(self, monkeypatch, window):
        # A root-relative LinkedIn href must be resolved against page_url (like
        # every other scraper) so the Post link is absolute — otherwise source
        # resolves to 'unknown' and the full-content scrape refuses the empty
        # host.
        html = (
            '<li class="post-list__item">'
            '<p class="grid-post__date">May 25, 2026</p>'
            '<a class="grid-post__link" href="/blog/engineering/rel-post">'
            "Relative Link Post</a></li>"
        )
        scraper = LinkedInBlogScraper(
            page_url="https://www.linkedin.com/blog/engineering", source="linkedin"
        )
        soup = BeautifulSoup(html, "html.parser")
        monkeypatch.setattr(scraper, "_fetch_page", lambda: soup)
        posts = scraper.fetch(*window)
        assert posts
        assert posts[0].link.startswith("https://www.linkedin.com/blog/engineering/")
        assert posts[0].source == "linkedin"


class TestQwenScraper:
    def test_parses_and_skips_nav(self, monkeypatch, window):
        scraper = _bind(
            QwenBlogScraper,
            "qwen.html",
            "https://qwenlm.github.io/blog/",
            "qwen",
            monkeypatch,
        )
        posts = scraper.fetch(*window)
        titles = {p.title for p in posts}
        assert "Qwen3 Release Notes" in titles
        # Nav links ("Blog Home", "Next Page") must not become posts.
        assert "Blog Home" not in titles
        assert not any("Next Page" in t for t in titles)


class TestQwenNavWordBoundary:
    """Nav-keyword filtering must match whole words, not substrings, so ML
    titles that merely CONTAIN a nav word ('Preventing' -> 'prev', 'PageRank'
    -> 'page', 'Continued' -> 'continue') are not silently dropped."""

    def _parse_title(self, title: str, monkeypatch):
        scraper = QwenBlogScraper(
            page_url="https://qwenlm.github.io/blog/", source="qwen"
        )
        # The anchor text is the title; the date lives as a sibling in the
        # parent (where _parse_item searches via str(item.parent)).
        html = (
            f'<div><a href="/blog/some-post/">{title}</a>'
            f"<span>May 25, 2026</span></div>"
        )
        item = BeautifulSoup(html, "html.parser").find("a")
        result = scraper._parse_item(item)
        return None if result is None else result.get("title")

    @pytest.mark.parametrize(
        "title",
        [
            "Preventing Reward Hacking in RLHF",
            "Continued Pretraining of Qwen3",
            "Improving PageRank with LLMs",
        ],
    )
    def test_ml_titles_with_nav_substrings_survive(self, title, monkeypatch):
        assert self._parse_title(title, monkeypatch) == title

    @pytest.mark.parametrize(
        "title", ["Next Page", "Previous", "Read more", "Continue reading"]
    )
    def test_real_nav_strings_still_dropped(self, title, monkeypatch):
        assert self._parse_title(title, monkeypatch) is None


class TestMetaScraper:
    def test_parses_blog_cards_and_window(self, monkeypatch, window):
        scraper = _bind(
            MetaAIBlogScraper,
            "meta.html",
            "https://ai.meta.com/blog",
            "meta",
            monkeypatch,
        )
        posts = scraper.fetch(*window)
        titles = {p.title for p in posts}
        assert "Llama 4 Multimodal Research" in titles
        assert "Segment Anything 3" in titles
        # Old post and nav links excluded.
        assert "Old Research Post" not in titles

    def test_links_resolved_against_page_url(self, monkeypatch, window):
        scraper = _bind(
            MetaAIBlogScraper,
            "meta.html",
            "https://ai.meta.com/blog",
            "meta",
            monkeypatch,
        )
        posts = scraper.fetch(*window)
        assert all(p.link.startswith("https://ai.meta.com/blog/") for p in posts)


class TestXAIScraper:
    def test_parses_news_and_window(self, monkeypatch, window):
        scraper = _bind(
            XAIBlogScraper,
            "xai.html",
            "https://x.ai/news",
            "xai",
            monkeypatch,
        )
        posts = scraper.fetch(*window)
        titles = {p.title for p in posts}
        assert "Grok 4 Launch" in titles
        assert "Colossus Cluster Scaling" in titles
        assert "Ancient Announcement" not in titles

    def test_links_resolved_against_page_url(self, monkeypatch, window):
        scraper = _bind(
            XAIBlogScraper,
            "xai.html",
            "https://x.ai/news",
            "xai",
            monkeypatch,
        )
        posts = scraper.fetch(*window)
        assert all(p.link.startswith("https://x.ai/news/") for p in posts)


class TestAnthropicScraper:
    def test_parses_engineering_index_and_window(self, monkeypatch, window):
        scraper = _bind(
            AnthropicBlogScraper,
            "anthropic.html",
            "https://www.anthropic.com/engineering",
            "anthropic",
            monkeypatch,
        )
        posts = scraper.fetch(*window)
        titles = {p.title for p in posts}
        assert "Building Effective Agents" in titles
        assert "Prompt Caching Internals" in titles
        assert "Legacy Systems Post" not in titles

    def test_links_resolved_absolute(self, monkeypatch, window):
        scraper = _bind(
            AnthropicBlogScraper,
            "anthropic.html",
            "https://www.anthropic.com/engineering",
            "anthropic",
            monkeypatch,
        )
        posts = scraper.fetch(*window)
        assert all(
            p.link.startswith("https://www.anthropic.com/engineering/") for p in posts
        )

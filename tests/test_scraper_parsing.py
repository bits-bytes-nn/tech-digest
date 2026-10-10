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
    LinkedInBlogScraper,
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


def _bind(scraper_cls, fixture: str, page_url: str, monkeypatch):
    scraper = scraper_cls(page_url=page_url)
    soup = _load(fixture)
    monkeypatch.setattr(scraper, "_fetch_page", lambda: soup)
    return scraper


class TestLinkedInScraper:
    def test_parses_items_and_window(self, monkeypatch, window):
        scraper = _bind(
            LinkedInBlogScraper,
            "linkedin.html",
            "https://www.linkedin.com/blog/engineering",
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
            page_url="https://www.linkedin.com/blog/engineering"
        )
        soup = BeautifulSoup(html, "html.parser")
        monkeypatch.setattr(scraper, "_fetch_page", lambda: soup)
        posts = scraper.fetch(*window)
        assert posts
        assert posts[0].link.startswith("https://www.linkedin.com/blog/engineering/")
        assert posts[0].source == "linkedin"

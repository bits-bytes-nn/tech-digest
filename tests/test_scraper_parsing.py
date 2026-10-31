"""Golden parse tests for the per-site HTML scrapers.

These scrapers carry the most fragile heuristics in the codebase (CSS selectors,
title-length bounds, date regex, parent-walk depth). They previously had ZERO
unit coverage — only routing/source-detection was tested. Here we feed each
scraper a saved, representative HTML fixture and pin the parsed output, so a
site-layout change or a selector regression is caught offline.

Network is fully stubbed: ``_fetch_page`` returns the fixture soup and
``_make_robust_request`` is neutralized so the full-article scrape that
``post_from_entry`` triggers never hits the wire.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from app.adapters import feeds
from app.adapters.feeds import (
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
    # The scrapers build Posts via post_from_entry, which scrapes the full
    # article when the feed body is thin. Neutralize the HTTP layer so no
    # request is made and content stays empty.
    monkeypatch.setattr(feeds, "_make_robust_request", lambda url: None)


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


class _Page:
    """Just enough of a ``requests.Response`` for ``parse_page``."""

    def __init__(self, html: str):
        self.content = html.encode("utf-8")
        self.text = html
        self.headers = {"content-type": "text/html; charset=utf-8"}
        self.encoding = "utf-8"


class TestLinkedInTruncatedTitles:
    """The index shows long titles cut to "..."; the post page has the full one."""

    _CARD = (
        '<li class="post-list__item">'
        '<p class="grid-post__date">May 25, 2026</p>'
        '<a class="grid-post__link" href="/blog/engineering/x">{title}</a></li>'
    )

    def _scraper(self, monkeypatch, title: str) -> LinkedInBlogScraper:
        scraper = LinkedInBlogScraper(
            page_url="https://www.linkedin.com/blog/engineering"
        )
        soup = BeautifulSoup(self._CARD.format(title=title), "html.parser")
        monkeypatch.setattr(scraper, "_fetch_page", lambda: soup)
        return scraper

    @pytest.mark.parametrize("marker", ["...", "\u2026"])
    def test_a_truncated_title_is_completed_from_the_post_page(
        self, monkeypatch, window, marker
    ):
        scraper = self._scraper(monkeypatch, f"Inside the data context la{marker}")
        page = (
            '<html><head><meta property="og:title" '
            'content="Inside the data context layer: trustworthy context">'
            "</head><body></body></html>"
        )
        monkeypatch.setattr(feeds, "_make_robust_request", lambda url: _Page(page))
        [post] = scraper.fetch(*window)
        assert post.title == "Inside the data context layer: trustworthy context"

    def test_a_page_without_og_title_falls_back_to_its_title_tag(
        self, monkeypatch, window
    ):
        scraper = self._scraper(monkeypatch, "Inside the data context la...")
        page = "<html><head><title>Inside the data context layer</title></head></html>"
        monkeypatch.setattr(feeds, "_make_robust_request", lambda url: _Page(page))
        [post] = scraper.fetch(*window)
        assert post.title == "Inside the data context layer"

    def test_an_unreachable_post_page_keeps_the_index_title(self, monkeypatch, window):
        scraper = self._scraper(monkeypatch, "Inside the data context la...")
        [post] = scraper.fetch(*window)
        assert post.title == "Inside the data context la..."

    def test_a_complete_title_costs_no_request(self, monkeypatch, window):
        calls: list[str] = []
        monkeypatch.setattr(
            feeds, "_make_robust_request", lambda url: calls.append(url)
        )
        scraper = self._scraper(monkeypatch, "Building A Feature Store")
        [post] = scraper.fetch(*window)
        assert post.title == "Building A Feature Store"
        # Only the article-body scrape asks for the page, not the title check.
        assert len(calls) == 1

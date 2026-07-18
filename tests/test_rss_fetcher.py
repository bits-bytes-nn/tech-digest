"""Tests for RssFetcher.fetch — the workhorse fetcher for most sources. Covers
the bozo-handling branch (a non-encoding parse error with no entries must raise
SourceFetchError so the source is marked FAILED, while a CharacterEncodingOverride
is tolerated) and the fail-closed date gate. feedparser.parse is stubbed, so no
network is touched."""

from __future__ import annotations

from datetime import UTC, datetime

import feedparser
import pytest
from feedparser.exceptions import CharacterEncodingOverride

from app.src import feed_parser
from app.src.feed_parser import RssFetcher, SourceFetchError


@pytest.fixture
def window() -> tuple[datetime, datetime]:
    start = datetime(2026, 5, 20, tzinfo=UTC)
    end = datetime(2026, 5, 31, 23, 59, 59, tzinfo=UTC)
    return start, end


class _FakeFeed:
    def __init__(self, entries, bozo=False, bozo_exception=None):
        self.entries = entries
        self.bozo = bozo
        self._bozo_exception = bozo_exception

    def get(self, key, default=None):
        if key == "bozo_exception":
            return self._bozo_exception
        return default


def _entry(published: str) -> feedparser.FeedParserDict:
    return feedparser.FeedParserDict(
        title="A Post",
        link="https://openai.com/news/x",
        published=published,
        summary="body",
    )


@pytest.fixture(autouse=True)
def no_scrape(monkeypatch):
    # Post.from_entry may scrape when feed content is thin; neutralize network.
    monkeypatch.setattr(feed_parser, "_make_robust_request", lambda url: None)


class TestRssFetcherBozo:
    def test_non_encoding_bozo_with_no_entries_raises(self, monkeypatch, window):
        feed = _FakeFeed(entries=[], bozo=True, bozo_exception=ValueError("bad XML"))
        monkeypatch.setattr(feed_parser.feedparser, "parse", lambda *a, **k: feed)
        fetcher = RssFetcher("https://openai.com/news/rss.xml")
        with pytest.raises(SourceFetchError):
            fetcher.fetch(*window)

    def test_encoding_override_with_entries_tolerated(self, monkeypatch, window):
        feed = _FakeFeed(
            entries=[_entry("2026-05-25T10:00:00Z")],
            bozo=True,
            bozo_exception=CharacterEncodingOverride("encoding mismatch"),
        )
        monkeypatch.setattr(feed_parser.feedparser, "parse", lambda *a, **k: feed)
        fetcher = RssFetcher("https://openai.com/news/rss.xml")
        posts = fetcher.fetch(*window)
        assert len(posts) == 1

    def test_unparseable_date_entry_dropped(self, monkeypatch, window):
        feed = _FakeFeed(entries=[_entry("not a date")], bozo=False)
        monkeypatch.setattr(feed_parser.feedparser, "parse", lambda *a, **k: feed)
        fetcher = RssFetcher("https://openai.com/news/rss.xml")
        # Fail-closed: undated entry excluded rather than dated to "now".
        assert fetcher.fetch(*window) == []

    def test_out_of_window_entry_excluded(self, monkeypatch, window):
        feed = _FakeFeed(entries=[_entry("2020-01-01T10:00:00Z")], bozo=False)
        monkeypatch.setattr(feed_parser.feedparser, "parse", lambda *a, **k: feed)
        fetcher = RssFetcher("https://openai.com/news/rss.xml")
        assert fetcher.fetch(*window) == []

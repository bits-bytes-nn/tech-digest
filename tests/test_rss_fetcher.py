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

from app.adapters import feeds
from app.adapters.feeds import RssFetcher, SourceFetchError, entry_date_text

FEED_URL = "https://aws.amazon.com/blogs/machine-learning/feed/"
POST_URL = "https://aws.amazon.com/blogs/machine-learning/"


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
        link=POST_URL + "x",
        published=published,
        summary="body",
    )


@pytest.fixture(autouse=True)
def offline_feed_download(monkeypatch):
    # RssFetcher downloads through the request layer before parsing; serve an
    # empty body so the stubbed feedparser.parse decides what the feed holds.
    from types import SimpleNamespace

    from app.adapters import feeds as _fp

    monkeypatch.setattr(
        _fp,
        "_try_request",
        lambda session, url, headers: SimpleNamespace(content=b"", headers={}, url=url),
    )


@pytest.fixture(autouse=True)
def no_scrape(monkeypatch):
    # post_from_entry may scrape when feed content is thin; neutralize network.
    monkeypatch.setattr(feeds, "_make_robust_request", lambda url: None)


class TestRssFetcherBozo:
    def test_non_encoding_bozo_with_no_entries_raises(self, monkeypatch, window):
        feed = _FakeFeed(entries=[], bozo=True, bozo_exception=ValueError("bad XML"))
        monkeypatch.setattr(feeds.feedparser, "parse", lambda *a, **k: feed)
        fetcher = RssFetcher(FEED_URL)
        with pytest.raises(SourceFetchError):
            fetcher.fetch(*window)

    def test_encoding_override_with_entries_tolerated(self, monkeypatch, window):
        feed = _FakeFeed(
            entries=[_entry("2026-05-25T10:00:00Z")],
            bozo=True,
            bozo_exception=CharacterEncodingOverride("encoding mismatch"),
        )
        monkeypatch.setattr(feeds.feedparser, "parse", lambda *a, **k: feed)
        fetcher = RssFetcher(FEED_URL)
        posts = fetcher.fetch(*window)
        assert len(posts) == 1

    def test_unparseable_date_entry_dropped(self, monkeypatch, window):
        feed = _FakeFeed(entries=[_entry("not a date")], bozo=False)
        monkeypatch.setattr(feeds.feedparser, "parse", lambda *a, **k: feed)
        fetcher = RssFetcher(FEED_URL)
        # Fail-closed: undated entry excluded rather than dated to "now".
        assert fetcher.fetch(*window) == []

    def test_out_of_window_entry_excluded(self, monkeypatch, window):
        feed = _FakeFeed(entries=[_entry("2020-01-01T10:00:00Z")], bozo=False)
        monkeypatch.setattr(feeds.feedparser, "parse", lambda *a, **k: feed)
        fetcher = RssFetcher(FEED_URL)
        assert fetcher.fetch(*window) == []


class TestAtomEntriesWithoutPublished:
    """A valid Atom feed can carry only ``<updated>``; every entry used to be lost.

    ``<published>`` is OPTIONAL in Atom while ``<updated>`` is REQUIRED, and
    feedparser maps them to distinct attributes. Reading only ``published`` meant
    the fail-closed date gate dropped every entry of such a feed — permanently and
    invisibly, because the source still reported candidates and so read as a merely
    quiet blog rather than a broken one.
    """

    ATOM_UPDATED_ONLY = (
        '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">'
        "<title>T</title><entry><title>Only Updated</title>"
        '<link href="https://updated.example/p"/>'
        "<updated>2026-08-05T10:00:00Z</updated><id>1</id></entry></feed>"
    )

    def test_feedparser_really_omits_published(self):
        """Pin the upstream behaviour this guards against, so the guard cannot be
        removed on the belief that feedparser normalizes the two."""
        entry = feedparser.parse(self.ATOM_UPDATED_ONLY).entries[0]
        assert not hasattr(entry, "published")
        assert entry.updated == "2026-08-05T10:00:00Z"

    def test_entry_date_text_falls_back_to_updated(self):
        entry = feedparser.parse(self.ATOM_UPDATED_ONLY).entries[0]
        assert entry_date_text(entry) == "2026-08-05T10:00:00Z"

    def test_published_wins_when_both_exist(self):
        """``updated`` is a modification time, so preferring it would let an old
        post edited this week re-enter the weekly window."""
        entry = feedparser.FeedParserDict(
            published="2026-08-01T00:00:00Z", updated="2026-08-07T00:00:00Z"
        )
        assert entry_date_text(entry) == "2026-08-01T00:00:00Z"

    def test_no_date_at_all_is_still_empty(self):
        assert entry_date_text(feedparser.FeedParserDict(title="t")) == ""

    def test_such_a_feed_now_yields_its_post(self, monkeypatch):
        parsed = feedparser.parse(self.ATOM_UPDATED_ONLY)
        monkeypatch.setattr(feeds.feedparser, "parse", lambda *a, **k: parsed)
        monkeypatch.setattr(feeds, "_make_robust_request", lambda url: None)
        fetcher = RssFetcher("https://updated.example/atom")
        posts = fetcher.fetch(
            datetime(2026, 8, 1, tzinfo=UTC), datetime(2026, 8, 8, tzinfo=UTC)
        )
        assert [p.title for p in posts] == ["Only Updated"]
        # And the Post carries the feed's date, not "now".
        assert posts[0].published_date.date().isoformat() == "2026-08-05"


class TestFeedDownload:
    """The feed is downloaded by the request layer, which carries the timeout."""

    def test_an_unreachable_feed_is_a_source_failure(self, monkeypatch, window):
        monkeypatch.setattr(feeds, "_try_request", lambda *a, **k: None)
        with pytest.raises(SourceFetchError):
            RssFetcher(FEED_URL).fetch(*window)

    def test_bytes_and_headers_reach_feedparser(self, monkeypatch, window):
        from types import SimpleNamespace

        seen = {}
        monkeypatch.setattr(
            feeds,
            "_try_request",
            lambda session, url, headers: SimpleNamespace(
                content=b"<rss/>",
                headers={"Content-Type": "application/rss+xml; charset=utf-8"},
                url=FEED_URL,
            ),
        )

        def parse(content, response_headers):
            seen["content"] = content
            seen["headers"] = response_headers
            return _FakeFeed(entries=[], bozo=False)

        monkeypatch.setattr(feeds.feedparser, "parse", parse)
        RssFetcher(FEED_URL).fetch(*window)
        assert seen["content"] == b"<rss/>"
        assert seen["headers"]["content-type"].startswith("application/rss+xml")
        assert seen["headers"]["content-location"] == FEED_URL


class TestUnreadableEntries:
    def _feed(self, links):
        return _FakeFeed(
            entries=[
                feedparser.FeedParserDict(
                    title=f"t{i}",
                    link=link,
                    published="2026-05-28T10:00:00Z",
                    summary="body " * 400,
                )
                for i, link in enumerate(links)
            ],
            bozo=False,
        )

    def test_one_bad_entry_costs_only_that_entry(self, monkeypatch, window):
        feed = self._feed(["http://[bad/x", POST_URL + "ok"])
        monkeypatch.setattr(feeds.feedparser, "parse", lambda *a, **k: feed)
        posts = RssFetcher(FEED_URL).fetch(*window)
        assert [p.link for p in posts] == [POST_URL + "ok"]

    def test_every_in_window_entry_failing_is_a_source_failure(
        self, monkeypatch, window
    ):
        feed = self._feed(["http://[bad/x", "http://[bad/y"])
        monkeypatch.setattr(feeds.feedparser, "parse", lambda *a, **k: feed)
        with pytest.raises(SourceFetchError):
            RssFetcher(FEED_URL).fetch(*window)

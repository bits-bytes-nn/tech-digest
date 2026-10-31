"""Tests for crawler health tracking: a fetch that errors is classified as
FAILED (not silently empty), and the report surfaces failing sources for the
SNS alert."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from bs4 import BeautifulSoup

import app.src.feed_parser as fp
from app.src.feed_parser import (
    CrawlReport,
    Post,
    PostCollector,
    SourceFetchError,
    SourceHealth,
    SourceStatus,
)


def _post(link: str) -> Post:
    return Post(title="t", link=link, published_date=datetime.now(UTC))


class _OkFetcher:
    def __init__(self, url, posts):
        self.source_url = url
        self._posts = posts

    def fetch(self, start, end):
        return self._posts


class _EmptyFetcher:
    def __init__(self, url):
        self.source_url = url

    def fetch(self, start, end):
        return []


class _FailingFetcher:
    def __init__(self, url):
        self.source_url = url

    def fetch(self, start, end):
        raise SourceFetchError("anti-bot block (403)")


class TestCrawlHealthClassification:
    def _collect(self, fetchers):
        collector = PostCollector(fetchers)
        now = datetime.now(UTC)
        posts = collector.collect_posts(now, now)
        return posts, collector.report

    def test_ok_source_classified(self):
        posts, report = self._collect(
            [_OkFetcher("https://a.com/feed", [_post("https://a.com/1")])]
        )
        assert len(posts) == 1
        assert report.ok and report.ok[0].status is SourceStatus.OK
        assert report.ok[0].post_count == 1

    def test_empty_source_classified(self):
        _, report = self._collect([_EmptyFetcher("https://b.com/feed")])
        assert report.empty and not report.failed
        assert report.empty[0].status is SourceStatus.EMPTY

    def test_failed_source_classified(self):
        _, report = self._collect([_FailingFetcher("https://meta.com/blog")])
        assert report.failed
        fail = report.failed[0]
        assert fail.status is SourceStatus.FAILED
        assert "403" in (fail.error or "")

    def test_one_failure_does_not_abort_others(self):
        posts, report = self._collect(
            [
                _FailingFetcher("https://meta.com/blog"),
                _OkFetcher("https://ok.com/feed", [_post("https://ok.com/1")]),
            ]
        )
        assert len(posts) == 1
        assert len(report.failed) == 1
        assert len(report.ok) == 1

    def test_summary_line(self):
        _, report = self._collect(
            [
                _OkFetcher("https://ok.com/feed", [_post("https://ok.com/1")]),
                _EmptyFetcher("https://b.com/feed"),
                _FailingFetcher("https://meta.com/blog"),
            ]
        )
        line = report.summary_line()
        assert "1 ok" in line and "1 empty" in line and "1 failed" in line


class TestCrawlReportAlert:
    def test_format_alert_lists_failures(self):
        report = CrawlReport(
            sources=[
                SourceHealth(
                    url="https://meta.com/blog",
                    fetcher="MetaAIBlogScraper",
                    status=SourceStatus.FAILED,
                    error="anti-bot block",
                ),
                SourceHealth(
                    url="https://ok.com/feed",
                    fetcher="RssFetcher",
                    status=SourceStatus.OK,
                    post_count=3,
                ),
            ]
        )
        alert = report.format_alert()
        assert "FAILED sources" in alert
        assert "https://meta.com/blog" in alert
        assert "anti-bot block" in alert
        assert "https://ok.com/feed" in alert


class TestBrokenVsQuiet:
    """EMPTY conflated two unrelated situations; ``candidates`` separates them.

    A blog that published nothing this week still offers its back catalogue and has
    the date window filter it out. A moved feed, or a scraper whose selector stopped
    matching, offers NOTHING to filter. Both used to render as one identical line
    among the ~12 empty sources of a normal run, so a broken crawler could sit there
    for weeks unnoticed. The discriminator needs no cross-run state: the counts come
    from the same fetch.
    """

    def _health(self, url: str, status: SourceStatus, candidates: int | None):
        return SourceHealth(
            url=url, fetcher="RssFetcher", status=status, candidates=candidates
        )

    def test_a_quiet_blog_is_not_flagged(self):
        report = CrawlReport(
            sources=[self._health("https://eugeneyan.com/rss/", SourceStatus.EMPTY, 14)]
        )
        assert report.empty
        assert report.likely_broken == []

    def test_a_source_offering_nothing_is_flagged(self):
        report = CrawlReport(
            sources=[self._health("https://ai.meta.com/blog/", SourceStatus.EMPTY, 0)]
        )
        assert [s.url for s in report.likely_broken] == ["https://ai.meta.com/blog/"]

    def test_unknown_count_is_not_flagged(self):
        """None means the fetcher reported nothing, which is not the same as zero.
        Treating it as zero would page on the unknown case."""
        report = CrawlReport(
            sources=[self._health("https://x.example/rss", SourceStatus.EMPTY, None)]
        )
        assert report.likely_broken == []

    def test_a_healthy_source_with_zero_candidates_is_impossible_but_not_flagged(self):
        """``likely_broken`` is derived from EMPTY only — an OK source produced
        posts by definition, so its count cannot mean the same thing."""
        report = CrawlReport(
            sources=[self._health("https://a.example/rss", SourceStatus.OK, 0)]
        )
        assert report.likely_broken == []

    def test_summary_line_names_the_broken_subset_only_when_present(self):
        quiet = CrawlReport(
            sources=[self._health("https://q/rss", SourceStatus.EMPTY, 9)]
        )
        assert "with nothing to filter" not in quiet.summary_line()
        broken = CrawlReport(
            sources=[self._health("https://b/rss", SourceStatus.EMPTY, 0)]
        )
        assert "1 with nothing to filter" in broken.summary_line()

    def test_report_body_explains_each_empty_source(self):
        report = CrawlReport(
            sources=[
                self._health("https://quiet/rss", SourceStatus.EMPTY, 14),
                self._health("https://broken/rss", SourceStatus.EMPTY, 0),
                self._health("https://unknown/rss", SourceStatus.EMPTY, None),
            ]
        )
        body = report.format_alert()
        assert "offered 14 item(s), none in the date window" in body
        assert "LIKELY BROKEN" in body
        assert "no count reported" in body


class TestCandidatesComeFromTheFetchers:
    """Each fetcher must report its own pre-filter count, or the discriminator is
    silently unavailable for that source (it reads as "unknown")."""

    @pytest.fixture(autouse=True)
    def offline_feed_download(self, monkeypatch):
        monkeypatch.setattr(
            fp,
            "_try_request",
            lambda session, url, headers: SimpleNamespace(
                content=b"", headers={}, url=url
            ),
        )

    WINDOW = (datetime(2026, 8, 1, tzinfo=UTC), datetime(2026, 8, 8, tzinfo=UTC))

    @staticmethod
    def _feed(entries: list) -> SimpleNamespace:
        return SimpleNamespace(bozo=0, entries=entries, get=lambda _k, _d=None: None)

    def test_rss_fetcher_reports_entry_count(self, monkeypatch):
        entry = fp.feedparser.FeedParserDict(
            title="old", link="https://a.example/1", published="2020-01-01"
        )
        monkeypatch.setattr(fp.feedparser, "parse", lambda *a, **k: self._feed([entry]))
        fetcher = fp.RssFetcher("https://a.example/rss")
        # The window excludes the 2020 entry, so the feed is EMPTY but not broken.
        assert fetcher.fetch(*self.WINDOW) == []
        assert fetcher.candidates == 1

    def test_rss_fetcher_reports_zero_for_an_empty_feed(self, monkeypatch):
        monkeypatch.setattr(fp.feedparser, "parse", lambda *a, **k: self._feed([]))
        fetcher = fp.RssFetcher("https://a.example/rss")
        fetcher.fetch(*self.WINDOW)
        assert fetcher.candidates == 0

    def test_a_stale_count_does_not_leak_into_the_next_fetch(self, monkeypatch):
        """A collector may be reused, and a count from a previous fetch would
        describe the wrong run."""

        def _boom(*a, **k):
            raise RuntimeError("network down")

        monkeypatch.setattr(fp.feedparser, "parse", _boom)
        fetcher = fp.RssFetcher("https://a.example/rss")
        fetcher.candidates = 99
        with pytest.raises(SourceFetchError):
            fetcher.fetch(*self.WINDOW)
        assert fetcher.candidates is None

    @staticmethod
    def _bound(monkeypatch, scraper, html):
        monkeypatch.setattr(
            scraper, "_fetch_page", lambda: BeautifulSoup(html, "html.parser")
        )
        monkeypatch.setattr(fp, "_make_robust_request", lambda url: None)
        return scraper

    @pytest.mark.parametrize(
        ("date_text", "expected"), [("sometime soon", 0), ("January 5, 2020", 1)]
    )
    def test_generic_scraper_counts_only_items_it_could_date(
        self, monkeypatch, date_text, expected
    ):
        # The Google Research failure exactly: the card and its date element are
        # there, the date text no longer parses.
        html = (
            '<ul><li class="post-list__item"><p class="grid-post__date">'
            f"{date_text}</p>"
            '<a class="grid-post__link" href="/blog/engineering/x">A Post</a>'
            "</li></ul>"
        )
        scraper = self._bound(
            monkeypatch,
            fp.LinkedInBlogScraper(
                page_url="https://www.linkedin.com/blog/engineering"
            ),
            html,
        )
        assert scraper.fetch(*self.WINDOW) == []
        assert scraper.candidates == expected

    def test_a_page_whose_cards_stopped_matching_offers_nothing(self, monkeypatch):
        scraper = self._bound(
            monkeypatch,
            fp.LinkedInBlogScraper(
                page_url="https://www.linkedin.com/blog/engineering"
            ),
            "<div>redesigned</div>",
        )
        assert scraper.fetch(*self.WINDOW) == []
        assert scraper.candidates == 0

    def test_rss_entries_with_no_readable_date_count_as_nothing_offered(
        self, monkeypatch
    ):
        entry = fp.feedparser.FeedParserDict(
            title="t", link="https://a.example/1", published="not a date"
        )
        monkeypatch.setattr(fp.feedparser, "parse", lambda *a, **k: self._feed([entry]))
        fetcher = fp.RssFetcher("https://a.example/rss")
        assert fetcher.fetch(*self.WINDOW) == []
        assert fetcher.candidates == 0

    def test_the_collector_records_what_the_fetcher_reported(self):
        class _Fetcher:
            source_url = "https://a.example/rss"
            candidates: int | None = None

            def fetch(self, start, end):
                self.candidates = 0
                return []

        collector = PostCollector([_Fetcher()])
        collector.collect_posts(*self.WINDOW)
        assert collector.report.sources[0].candidates == 0
        assert len(collector.report.likely_broken) == 1


class TestPagesUnreadable:
    """A feed that fetches while every article page is blocked (OpenAI behind a
    Cloudflare challenge) used to count as a healthy source."""

    def _ok(self, posts: int, unreadable: int) -> SourceHealth:
        return SourceHealth(
            url="https://openai.com/news/rss.xml",
            fetcher="RssFetcher",
            status=SourceStatus.OK,
            post_count=posts,
            unreadable_pages=unreadable,
        )

    def test_every_page_unreadable_is_flagged(self):
        report = CrawlReport(sources=[self._ok(15, 15)])
        assert report.pages_unreadable == report.sources
        assert "1 with unreadable article pages" in report.summary_line()
        assert "15 of 15 post(s)" in report.format_alert()

    @pytest.mark.parametrize(("posts", "unreadable"), [(15, 14), (3, 0), (0, 0)])
    def test_partial_or_no_failures_are_not(self, posts, unreadable):
        report = CrawlReport(sources=[self._ok(posts, unreadable)])
        assert report.pages_unreadable == []
        assert "unreadable" not in report.summary_line()

    def test_the_collector_counts_contributed_posts_only(self):
        blocked = _post("https://aws.amazon.com/blogs/machine-learning/a")
        blocked.page_unreadable = True
        duplicate = _post("https://aws.amazon.com/blogs/machine-learning/a")
        duplicate.page_unreadable = True
        collector = PostCollector(
            [_OkFetcher("https://aws.amazon.com/blogs/machine-learning/feed/", [blocked, duplicate])]
        )
        now = datetime.now(UTC)
        collector.collect_posts(now, now)
        source = collector.report.sources[0]
        assert (source.post_count, source.unreadable_pages) == (1, 1)
        assert collector.report.pages_unreadable == [source]


class TestTruncatedFeeds:
    """A feed that lists only its newest N entries can drop the start of the week
    before the run. Detected without stored state: the oldest dated entry offered
    is newer than the window start."""

    WINDOW = (datetime(2026, 10, 24, tzinfo=UTC), datetime(2026, 10, 31, tzinfo=UTC))

    @pytest.fixture(autouse=True)
    def offline(self, monkeypatch):
        monkeypatch.setattr(
            fp,
            "_try_request",
            lambda session, url, headers: SimpleNamespace(
                content=b"", headers={}, url=url
            ),
        )
        monkeypatch.setattr(fp, "_make_robust_request", lambda url: None)

    def _fetch(self, monkeypatch, dates: list[str]) -> fp.RssFetcher:
        entries = [
            fp.feedparser.FeedParserDict(
                title=f"t{i}",
                link=f"https://aws.amazon.com/blogs/machine-learning/{i}",
                published=date,
                content=[{"value": "<p>" + "word " * 800 + "</p>"}],
            )
            for i, date in enumerate(dates)
        ]
        feed = SimpleNamespace(bozo=0, entries=entries, get=lambda _k, _d=None: None)
        monkeypatch.setattr(fp.feedparser, "parse", lambda *a, **k: feed)
        fetcher = fp.RssFetcher("https://aws.amazon.com/blogs/machine-learning/feed/")
        fetcher.fetch(*self.WINDOW)
        return fetcher

    def test_a_feed_whose_oldest_entry_is_in_the_window_is_flagged(
        self, monkeypatch, caplog, propagating_logger
    ):
        with caplog.at_level("WARNING"):
            fetcher = self._fetch(
                monkeypatch, ["2026-10-30T10:00:00Z", "2026-10-26T17:15:00Z"]
            )
        assert fetcher.truncated
        warning = next(r for r in caplog.records if r.levelname == "WARNING")
        assert "machine-learning/feed/" in warning.getMessage()
        assert "2026-10-26T17:15:00+00:00" in warning.getMessage()

    def test_a_feed_reaching_before_the_window_is_not(self, monkeypatch):
        fetcher = self._fetch(
            monkeypatch, ["2026-10-30T10:00:00Z", "2026-10-20T00:00:00Z"]
        )
        assert not fetcher.truncated

    def test_an_entry_exactly_at_the_window_start_covers_it(self, monkeypatch):
        """The window is half-open, ``start <= date``: nothing before it is wanted."""
        assert not self._fetch(monkeypatch, ["2026-10-24T00:00:00Z"]).truncated

    def test_undated_entries_do_not_count_and_an_empty_feed_is_not_truncated(
        self, monkeypatch
    ):
        assert not self._fetch(monkeypatch, []).truncated
        assert not self._fetch(monkeypatch, ["not a date"]).truncated
        assert self._fetch(
            monkeypatch, ["not a date", "2026-10-28T00:00:00Z"]
        ).truncated

    def test_the_flag_does_not_leak_into_the_next_fetch(self, monkeypatch):
        fetcher = self._fetch(monkeypatch, ["2026-10-30T10:00:00Z"])
        assert fetcher.truncated
        monkeypatch.setattr(
            fp.feedparser,
            "parse",
            lambda *a, **k: SimpleNamespace(
                bozo=0, entries=[], get=lambda _k, _d=None: None
            ),
        )
        fetcher.fetch(*self.WINDOW)
        assert not fetcher.truncated

    def test_the_collector_records_it_and_it_stays_out_of_the_alert(self):
        class _Truncated(_OkFetcher):
            truncated = True

        collector = PostCollector(
            [
                _Truncated("https://a.example/feed", [_post("https://a.example/1")]),
                _OkFetcher("https://b.example/feed", [_post("https://b.example/1")]),
            ]
        )
        collector.collect_posts(*self.WINDOW)
        report = collector.report
        assert [s.url for s in report.truncated] == ["https://a.example/feed"]
        # Logged and counted, not alerted on: no SNS trigger reads it.
        assert not (report.failed or report.likely_broken or report.pages_unreadable)

    def test_run_summary_counts_truncated_sources(self, caplog, propagating_logger):
        import json

        from app import main
        from app.src.constants import Language

        report = CrawlReport(
            sources=[
                SourceHealth(
                    url="https://a.example/feed",
                    fetcher="RssFetcher",
                    status=SourceStatus.OK,
                    post_count=1,
                    truncated=True,
                )
            ]
        )
        with caplog.at_level("INFO"):
            main._log_run_summary(report, [], [], Language.EN)
        line = next(r.getMessage() for r in caplog.records if "RUN_SUMMARY" in r.getMessage())
        assert json.loads(line.removeprefix("RUN_SUMMARY "))["sources_truncated"] == 1

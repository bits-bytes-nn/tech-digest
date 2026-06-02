"""Tests for crawler health tracking: a fetch that errors is classified as
FAILED (not silently empty), and the report surfaces failing sources for the
SNS alert."""

from __future__ import annotations

from datetime import UTC, datetime

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

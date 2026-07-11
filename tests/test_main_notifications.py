"""Tests for main.py notification wiring: the crawl-health SNS alert and the
success notification's crawl summary. Uses a fake boto session — no AWS."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# main.py uses the container import layout (`from src import ...`,
# `from configs import ...`), so it must be imported with app/ on the path.
APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import main  # noqa: E402

from src.feed_parser import CrawlReport, SourceHealth, SourceStatus  # noqa: E402


class _FakeSNS:
    def __init__(self):
        self.published: list[dict] = []

    def publish(self, **kwargs):
        self.published.append(kwargs)
        return {"MessageId": "test"}


class _FakeSession:
    def __init__(self):
        self.sns = _FakeSNS()

    def client(self, service):
        assert service == "sns"
        return self.sns


@pytest.fixture
def failing_report() -> CrawlReport:
    return CrawlReport(
        sources=[
            SourceHealth(
                url="https://ai.meta.com/blog",
                fetcher="MetaAIBlogScraper",
                status=SourceStatus.FAILED,
                error="anti-bot block (403)",
            ),
            SourceHealth(
                url="https://openai.com/news/rss.xml",
                fetcher="RssFetcher",
                status=SourceStatus.OK,
                post_count=4,
            ),
        ]
    )


class TestCrawlHealthAlert:
    def test_alert_published_with_failures(self, failing_report):
        session = _FakeSession()
        main._send_crawl_health_alert(session, "arn:topic", failing_report)
        assert len(session.sns.published) == 1
        msg = session.sns.published[0]
        assert "Crawl Health" in msg["Subject"]
        assert "ALERT" in msg["Subject"]
        assert "ai.meta.com/blog" in msg["Message"]
        assert "anti-bot block" in msg["Message"]

    def test_report_failed_partition(self, failing_report):
        assert len(failing_report.failed) == 1
        assert len(failing_report.ok) == 1


class TestPartialDeliveryAlert:
    def test_alert_on_partial_failure_includes_crawl_summary(self, failing_report):
        session = _FakeSession()
        main._maybe_send_partial_delivery_alert(
            session,
            "arn:topic",
            success_count=2,
            total_recipients=3,
            failed_recipients=["bad@example.com"],
            crawl_report=failing_report,
        )
        assert len(session.sns.published) == 1
        published = session.sns.published[0]
        assert "ALERT" in published["Subject"]
        msg = published["Message"]
        assert "2/3" in msg
        assert "bad@example.com" in msg
        assert "Crawl health:" in msg
        assert "1 ok" in msg and "1 failed" in msg

    def test_no_alert_on_full_success(self, failing_report):
        session = _FakeSession()
        main._maybe_send_partial_delivery_alert(
            session,
            "arn:topic",
            success_count=3,
            total_recipients=3,
            failed_recipients=[],
            crawl_report=failing_report,
        )
        assert session.sns.published == []


class TestEmptyDigestAlert:
    def _post(self, title: str):
        from datetime import datetime

        from src.feed_parser import Post

        return Post(
            title=title, link="https://x.com", published_date=datetime(2026, 7, 11)
        )

    def _report(self, total: int) -> CrawlReport:
        return CrawlReport(
            sources=[
                SourceHealth(
                    url="https://x.com/rss",
                    fetcher="RssFetcher",
                    status=SourceStatus.OK,
                    post_count=total,
                )
            ]
        )

    def test_alert_when_all_posts_filtered_out(self):
        session = _FakeSession()
        filtered_out = [
            (self._post("A"), "Filtering failed (no model response)."),
            (self._post("B"), "Filtering failed (no model response)."),
            (self._post("C"), "Low relevance score."),
        ]
        main._maybe_send_empty_digest_alert(
            session, "arn:topic", self._report(3), filtered_out
        )
        assert len(session.sns.published) == 1
        msg = session.sns.published[0]["Message"]
        assert "Empty Digest" in session.sns.published[0]["Subject"]
        assert "ALERT" in session.sns.published[0]["Subject"]
        # Reason breakdown surfaces the dominant cause (mass model failure here).
        assert "2x Filtering failed (no model response)." in msg
        assert "1x Low relevance score." in msg

    def test_no_alert_when_nothing_collected(self):
        # An empty crawl is covered by crawl-health alerts, not this one.
        session = _FakeSession()
        main._maybe_send_empty_digest_alert(session, "arn:topic", self._report(0), [])
        assert session.sns.published == []

    def test_no_alert_when_no_filtered_out_posts(self):
        # Defensive: total>0 but nothing recorded as filtered out (shouldn't
        # normally happen) must not fire a misleading alert.
        session = _FakeSession()
        main._maybe_send_empty_digest_alert(session, "arn:topic", self._report(5), [])
        assert session.sns.published == []


class TestHandlerControlFlow:
    """Exercise handler's early-return and error paths without AWS/Bedrock by
    stubbing config, boto sessions, AWS-env detection, and the fetch step."""

    def _patch_common(self, monkeypatch):
        monkeypatch.setattr(main, "is_running_in_aws", lambda: False)

        class _Cfg:
            class resources:
                profile_name = None
                default_region_name = "ap-northeast-2"
                bedrock_region_name = "us-west-2"

        monkeypatch.setattr(main.Config, "load", staticmethod(lambda: _Cfg()))
        monkeypatch.setattr(main.boto3, "Session", lambda **k: object())

    def test_no_posts_returns_200(self, monkeypatch):
        self._patch_common(monkeypatch)
        from src.feed_parser import CrawlReport

        monkeypatch.setattr(
            main,
            "_fetch_and_filter_posts",
            lambda *a, **k: ([], "2026-06-01", [], CrawlReport()),
        )
        result = main.handler({}, None)
        assert result["statusCode"] == 200
        assert "No posts" in result["body"]

    def test_exception_returns_500(self, monkeypatch):
        self._patch_common(monkeypatch)

        def _boom(*a, **k):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(main, "_fetch_and_filter_posts", _boom)
        result = main.handler({}, None)
        assert result["statusCode"] == 500
        assert "kaboom" in result["body"]

    def test_exception_publishes_failure_notification(self, monkeypatch):
        # On AWS with a topic configured, an unhandled error must surface via SNS
        # (the "failures always surface" design promise), not just return 500.
        self._patch_common(monkeypatch)
        monkeypatch.setattr(main, "is_running_in_aws", lambda: True)
        monkeypatch.setenv(main.EnvVars.TOPIC_ARN.value, "arn:topic")
        # On AWS the handler uses the module-global session, not boto3.Session().
        session = _FakeSession()
        monkeypatch.setattr(main, "DEFAULT_BOTO_SESSION", session)
        monkeypatch.setattr(main, "BEDROCK_BOTO_SESSION", session)

        def _boom(*a, **k):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(main, "_fetch_and_filter_posts", _boom)
        monkeypatch.setattr(main, "_setup_aws_env", lambda *a, **k: None)

        result = main.handler({}, None)
        assert result["statusCode"] == 500
        assert len(session.sns.published) == 1
        assert "kaboom" in session.sns.published[0]["Message"]

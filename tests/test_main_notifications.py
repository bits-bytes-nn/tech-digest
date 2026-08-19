"""Tests for main.py notification wiring: the crawl-health SNS alert and the
success notification's crawl summary. Uses a fake boto session — no AWS."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest

# main.py uses the container import layout (`from src import ...`,
# `from configs import ...`), so it must be imported with app/ on the path.
APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import main  # noqa: E402

from src.feed_parser import CrawlReport, SourceHealth, SourceStatus  # noqa: E402

PROJECT = "tech-digest"


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
        main._send_crawl_health_alert(session, "arn:topic", PROJECT, failing_report, [])
        assert len(session.sns.published) == 1
        msg = session.sns.published[0]
        assert "Crawl Health" in msg["Subject"]
        assert "ALERT" in msg["Subject"]
        assert "ai.meta.com/blog" in msg["Message"]
        assert "anti-bot block" in msg["Message"]

    def test_expected_flaky_source_does_not_alert(self, failing_report):
        """A source that is known to reject AWS egress IPs every week must not
        page, or the alarm becomes noise that hides a genuinely new breakage."""
        session = _FakeSession()
        main._send_crawl_health_alert(
            session, "arn:topic", PROJECT, failing_report, ["ai.meta.com"]
        )
        assert session.sns.published == []

    def test_unexpected_failure_still_alerts_when_others_are_allowed(
        self, failing_report
    ):
        failing_report.sources.append(
            SourceHealth(
                url="https://research.google/blog",
                fetcher="GoogleBlogScraper",
                status=SourceStatus.FAILED,
                error="selector changed",
            )
        )
        session = _FakeSession()
        main._send_crawl_health_alert(
            session, "arn:topic", PROJECT, failing_report, ["ai.meta.com"]
        )
        assert len(session.sns.published) == 1
        assert "research.google/blog" in session.sns.published[0]["Message"]

    def test_report_failed_partition(self, failing_report):
        assert len(failing_report.failed) == 1
        assert len(failing_report.ok) == 1


class TestPartialDeliveryAlert:
    def test_alert_on_partial_failure_includes_crawl_summary(self, failing_report):
        session = _FakeSession()
        main._maybe_send_partial_delivery_alert(
            session,
            "arn:topic",
            PROJECT,
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
            PROJECT,
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
            session, "arn:topic", PROJECT, self._report(3), filtered_out
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
        main._maybe_send_empty_digest_alert(
            session, "arn:topic", PROJECT, self._report(0), []
        )
        assert session.sns.published == []

    def test_no_alert_when_no_filtered_out_posts(self):
        # Defensive: total>0 but nothing recorded as filtered out (shouldn't
        # normally happen) must not fire a misleading alert.
        session = _FakeSession()
        main._maybe_send_empty_digest_alert(
            session, "arn:topic", PROJECT, self._report(5), []
        )
        assert session.sns.published == []


class TestGeneratePostFilename:
    """Two distinct posts (different links) that sanitize to the same
    source+title must map to DIFFERENT filenames, or one silently overwrites
    the other on disk and vanishes from the digest."""

    def _post(self, title: str, link: str):
        from datetime import datetime

        from src.feed_parser import Post

        return Post(
            title=title,
            link=link,
            published_date=datetime(2026, 7, 11),
            source="aws",
        )

    def test_same_title_distinct_links_get_distinct_filenames(self):
        a = self._post("GPT-4 is here!", "https://aws.amazon.com/a")
        b = self._post("GPT-4 is here?", "https://aws.amazon.com/b")
        fa = main._generate_post_filename(a)
        fb = main._generate_post_filename(b)
        assert fa != fb
        assert fa.startswith("aws-gpt-4_is_here-") and fa.endswith(".json")

    def test_filename_is_stable_for_same_link(self):
        p = self._post("Title", "https://aws.amazon.com/x")
        assert main._generate_post_filename(p) == main._generate_post_filename(p)


class TestEmailOrchestration:
    """_process_newsletter_emails counts successes/failures correctly so the
    partial-delivery alert fires on an all-failed (silent) delivery, and skips
    recipients a previous attempt at the same issue already reached."""

    class _Cfg:
        class resources:
            stage = "dev"
            s3_bucket_name = "bucket"
            s3_prefix = "tech-digest"

        class newsletter:
            header_title = "Digest"
            sender = "sender@example.com"

    def test_all_sends_fail_reports_all_as_failed(self, monkeypatch, tmp_path):
        newsletter = tmp_path / "n.html"
        newsletter.write_text("<html>body</html>", encoding="utf-8")
        monkeypatch.setattr(main, "send_email", lambda *a, **k: False)
        monkeypatch.setattr(main.time, "sleep", lambda *_: None)
        outcome = main._process_newsletter_emails(
            newsletter,
            "2026-07-11",
            self._Cfg(),
            object(),
            main.Language.KO,
            recipients=["a@example.com", "b@example.com"],
        )
        assert outcome.succeeded == 0
        assert outcome.attempted == 2
        assert set(outcome.failed) == {"a@example.com", "b@example.com"}

    def test_unreadable_newsletter_reports_all_failed(self, monkeypatch, tmp_path):
        missing = tmp_path / "does-not-exist.html"
        monkeypatch.setattr(main, "send_email", lambda *a, **k: True)
        outcome = main._process_newsletter_emails(
            missing,
            "2026-07-11",
            self._Cfg(),
            object(),
            main.Language.KO,
            recipients=["a@example.com"],
        )
        assert outcome.succeeded == 0
        assert outcome.attempted == 1
        assert outcome.failed == ["a@example.com"]

    def test_partial_success_counts_split(self, monkeypatch, tmp_path):
        newsletter = tmp_path / "n.html"
        newsletter.write_text("<html>body</html>", encoding="utf-8")
        # First recipient succeeds, second fails.
        outcomes = iter([True, False])
        monkeypatch.setattr(main, "send_email", lambda *a, **k: next(outcomes))
        monkeypatch.setattr(main.time, "sleep", lambda *_: None)
        outcome = main._process_newsletter_emails(
            newsletter,
            "2026-07-11",
            self._Cfg(),
            object(),
            main.Language.KO,
            recipients=["ok@example.com", "bad@example.com"],
        )
        assert outcome.succeeded == 1
        assert outcome.failed == ["bad@example.com"]
        assert outcome.attempted == 2

    def test_explicit_recipients_bypass_the_ledger(self, monkeypatch, tmp_path):
        """A --recipients run is the operator addressing one mailbox on purpose.

        It must neither read nor write the ledger, or a test send would make the
        scheduled run skip that reader for the rest of the issue's life.
        """
        newsletter = tmp_path / "n.html"
        newsletter.write_text("<html>body</html>", encoding="utf-8")
        monkeypatch.setattr(main, "send_email", lambda *a, **k: True)
        monkeypatch.setattr(main.time, "sleep", lambda *_: None)
        touched = []
        monkeypatch.setattr(
            main, "read_s3_json", lambda *a, **k: touched.append("read") or None
        )
        monkeypatch.setattr(
            main, "write_s3_json", lambda *a, **k: touched.append("write") or True
        )
        outcome = main._process_newsletter_emails(
            newsletter,
            "2026-07-11",
            self._Cfg(),
            object(),
            main.Language.KO,
            recipients=["me@example.com"],
        )
        assert outcome.succeeded == 1
        assert touched == []


class TestHandlerControlFlow:
    """Exercise handler's early-return and error paths without AWS/Bedrock by
    stubbing config, boto sessions, AWS-env detection, and the fetch step."""

    def _patch_common(self, monkeypatch):
        monkeypatch.setattr(main, "is_running_in_aws", lambda: False)

        class _Cfg:
            class resources:
                # project_name is read for the alarm subject line, so a stub
                # without it makes the failure paths raise inside the handler's
                # own error handling.
                project_name = PROJECT
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

    def test_empty_digest_after_filtering_publishes_alert(self, monkeypatch):
        # On AWS: posts were collected but none survived filtering. The handler
        # must return 200 AND fire exactly one 'Empty Digest' SNS alert (the
        # silent-failure guard). Uses an all-OK crawl report so the crawl-health
        # alert does not also fire.
        self._patch_common(monkeypatch)
        monkeypatch.setattr(main, "is_running_in_aws", lambda: True)
        monkeypatch.setenv(main.EnvVars.TOPIC_ARN.value, "arn:topic")
        session = _FakeSession()
        monkeypatch.setattr(main, "DEFAULT_BOTO_SESSION", session)
        monkeypatch.setattr(main, "BEDROCK_BOTO_SESSION", session)
        monkeypatch.setattr(main, "_setup_aws_env", lambda *a, **k: None)

        from datetime import datetime

        from src.feed_parser import Post

        report = CrawlReport(
            sources=[
                SourceHealth(
                    url="https://x.com/rss",
                    fetcher="RssFetcher",
                    status=SourceStatus.OK,
                    post_count=2,
                )
            ]
        )
        filtered_out = [
            (
                Post(
                    title="A",
                    link="https://x.com/a",
                    published_date=datetime(2026, 7, 11),
                ),
                "Summarization failed (no model response).",
            ),
            (
                Post(
                    title="B",
                    link="https://x.com/b",
                    published_date=datetime(2026, 7, 11),
                ),
                "Summarization failed (no model response).",
            ),
        ]
        monkeypatch.setattr(
            main,
            "_fetch_and_filter_posts",
            lambda *a, **k: ([], "2026-07-11", filtered_out, report),
        )

        result = main.handler({}, None)
        assert result["statusCode"] == 200
        assert len(session.sns.published) == 1
        assert "Empty Digest" in session.sns.published[0]["Subject"]

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


class TestNoRecipientsAlert:
    """The pipeline's worst silent failure: everything succeeds and nobody gets
    the issue. It is invisible to the existing alerts — partial-delivery keys on
    FAILED recipients, and a run with no recipients has none — so a missing or
    empty recipients object in S3 produced a 200 and no page."""

    def _report(self) -> CrawlReport:
        return CrawlReport(
            sources=[
                SourceHealth(
                    url="https://x.com/rss",
                    fetcher="RssFetcher",
                    status=SourceStatus.OK,
                    post_count=3,
                )
            ]
        )

    def test_alert_names_the_object_to_go_look_at(self, monkeypatch):
        from configs import Config

        # The committed CI config, not the developer's: config-dev.yaml is
        # gitignored, so loading it passes locally and fails in CI.
        monkeypatch.setenv("CONFIG_FILE_SUFFIX", "ci")
        session = _FakeSession()
        config = Config.load()
        main._send_no_recipients_alert(session, "arn:topic", config, self._report())
        assert len(session.sns.published) == 1
        published = session.sns.published[0]
        assert "No Recipients" in published["Subject"]
        assert "ALERT" in published["Subject"]
        # The cause is always outside the code, so the alarm has to point at it.
        assert "s3://" in published["Message"]
        assert "recipients" in published["Message"].lower()
        assert "0/0" in published["Message"]

    def test_partial_delivery_alert_cannot_cover_this_case(self, failing_report):
        """Pins the reason a separate alarm is needed rather than reusing that one."""
        session = _FakeSession()
        main._maybe_send_partial_delivery_alert(
            session, "arn:topic", PROJECT, 0, 0, [], failing_report
        )
        assert session.sns.published == []


class TestClippedNewsletterAlert:
    """Mail clients truncate an over-large message, so the tail of the issue is
    replaced by a "View entire message" link. Nothing about the send fails, and
    the renderer's log line is seen by nobody on a weekly unattended job. A real
    prod issue (2026-06-02) shipped at 101.4% of the limit."""

    def test_no_alert_under_the_limit(self, tmp_path):
        from src import GMAIL_CLIP_BYTES

        path = tmp_path / "n.html"
        path.write_bytes(b"x" * (GMAIL_CLIP_BYTES - 1))
        session = _FakeSession()
        main._maybe_send_clipped_newsletter_alert(
            session, "arn:topic", PROJECT, path, 3
        )
        assert session.sns.published == []

    def test_alert_over_the_limit_reports_size_and_remedy(self, tmp_path):
        from src import GMAIL_CLIP_BYTES

        path = tmp_path / "n.html"
        path.write_bytes(b"x" * (GMAIL_CLIP_BYTES + 2048))
        session = _FakeSession()
        main._maybe_send_clipped_newsletter_alert(
            session, "arn:topic", PROJECT, path, 5
        )
        assert len(session.sns.published) == 1
        published = session.sns.published[0]
        assert "Newsletter Clipped" in published["Subject"]
        assert "max_posts" in published["Message"]
        assert "5" in published["Message"]

    def test_missing_file_does_not_raise(self, tmp_path):
        """A size check must never be the thing that fails a successful run."""
        session = _FakeSession()
        main._maybe_send_clipped_newsletter_alert(
            session, "arn:topic", PROJECT, tmp_path / "absent.html", 3
        )
        assert session.sns.published == []


class TestDuplicateSendGuard:
    """The ledger makes the send loop idempotent per recipient.

    Both retry paths (the Batch job definition's ``retry_attempts=2`` and the
    EventBridge target's) re-run main.py from the start, so a job that died after
    the send loop used to deliver the whole issue a second time — and a Spot
    reclaim is the likeliest cause now that Spot is the preferred compute
    environment.
    """

    class _Cfg:
        class resources:
            stage = "prod"
            s3_bucket_name = "bucket"
            s3_prefix = "tech-digest"

        class newsletter:
            header_title = "Digest"
            sender = "sender@example.com"

    def _run(self, monkeypatch, tmp_path, ledger, send_results=None):
        """Run the send phase against a fake ledger; returns (outcome, writes)."""
        newsletter = tmp_path / "n.html"
        newsletter.write_text("<html>body</html>", encoding="utf-8")
        sent: list[str] = []
        writes: list[dict] = []
        results = iter(send_results) if send_results is not None else None

        def _send(_session, _subject, _sender, to, _content):
            sent.append(to[0])
            return next(results) if results is not None else True

        monkeypatch.setattr(main, "send_email", _send)
        monkeypatch.setattr(main.time, "sleep", lambda *_: None)
        monkeypatch.setattr(main, "read_s3_json", lambda *a, **k: ledger)
        monkeypatch.setattr(
            main,
            "write_s3_json",
            # Deep-copy the payload: the real helper serializes it immediately,
            # while this fake would otherwise keep a reference that the loop keeps
            # mutating, making every recorded write look identical.
            lambda _s, _b, _k, payload: writes.append(deepcopy(payload)) or True,
        )
        outcome = main._process_newsletter_emails(
            newsletter,
            "2026-08-22",
            self._Cfg(),
            object(),
            main.Language.KO,
            recipients=None,
        )
        return outcome, sent, writes

    def _patch_recipients(self, monkeypatch, recipients):
        monkeypatch.setattr(main, "_get_recipients", lambda *a, **k: list(recipients))

    def test_a_retry_of_a_completed_send_sends_nothing(self, monkeypatch, tmp_path):
        self._patch_recipients(monkeypatch, ["a@example.com", "b@example.com"])
        outcome, sent, _ = self._run(
            monkeypatch,
            tmp_path,
            ledger={"delivered": ["a@example.com", "b@example.com"]},
        )
        assert sent == []
        assert outcome.skipped == 2
        assert outcome.attempted == 0
        assert outcome.resolved == 2

    def test_a_retry_after_a_mid_loop_crash_serves_only_the_remainder(
        self, monkeypatch, tmp_path
    ):
        """Strictly better than before, not merely not-worse.

        An issue-level marker would have claimed the whole issue was done and left
        the unserved readers with nothing; no marker at all re-sent to everyone.
        """
        self._patch_recipients(
            monkeypatch, ["a@example.com", "b@example.com", "c@example.com"]
        )
        outcome, sent, _ = self._run(
            monkeypatch, tmp_path, ledger={"delivered": ["a@example.com"]}
        )
        assert sent == ["b@example.com", "c@example.com"]
        assert outcome.skipped == 1
        assert outcome.succeeded == 2

    def test_the_ledger_is_written_after_every_success_not_once_at_the_end(
        self, monkeypatch, tmp_path
    ):
        """A ledger written only at the end is absent exactly when it matters."""
        self._patch_recipients(monkeypatch, ["a@example.com", "b@example.com"])
        _outcome, _sent, writes = self._run(monkeypatch, tmp_path, ledger=None)
        assert len(writes) == 2
        assert writes[0]["delivered"] == ["a@example.com"]
        assert writes[1]["delivered"] == ["a@example.com", "b@example.com"]

    def test_a_failed_send_is_not_recorded_as_delivered(self, monkeypatch, tmp_path):
        """Otherwise the retry would skip the reader who never got the issue."""
        self._patch_recipients(monkeypatch, ["ok@example.com", "bad@example.com"])
        outcome, _sent, writes = self._run(
            monkeypatch, tmp_path, ledger=None, send_results=[True, False]
        )
        assert outcome.failed == ["bad@example.com"]
        assert writes[-1]["delivered"] == ["ok@example.com"]

    def test_comparison_is_case_insensitive(self, monkeypatch, tmp_path):
        self._patch_recipients(monkeypatch, ["Reader@Example.COM"])
        outcome, sent, _ = self._run(
            monkeypatch, tmp_path, ledger={"delivered": ["reader@example.com"]}
        )
        assert sent == []
        assert outcome.skipped == 1

    def test_an_unreadable_ledger_fails_open(self, monkeypatch, tmp_path):
        """A duplicate issue is an annoyance; an issue nobody receives is the
        failure this pipeline exists to avoid. So an absent or malformed ledger
        must send, not withhold."""
        self._patch_recipients(monkeypatch, ["a@example.com"])
        for ledger in (None, {}, {"delivered": "not-a-list"}, {"other": 1}):
            _outcome, sent, _ = self._run(monkeypatch, tmp_path, ledger=ledger)
            assert sent == ["a@example.com"], ledger

    def test_ledger_key_identifies_stage_date_and_language(self):
        key = main._delivery_ledger_key(self._Cfg(), "2026-08-22", main.Language.EN)
        assert key == "tech-digest/deliveries/delivered-prod-2026-08-22-en.json"

    def test_ledger_key_without_an_s3_prefix_has_no_leading_slash(self):
        class _NoPrefix(self._Cfg):
            class resources:
                stage = "dev"
                s3_bucket_name = "bucket"
                s3_prefix = None

        key = main._delivery_ledger_key(_NoPrefix(), "2026-08-22", main.Language.KO)
        assert key == "deliveries/delivered-dev-2026-08-22-ko.json"

    def test_a_different_issue_gets_a_different_ledger(self):
        cfg = self._Cfg()
        keys = {
            main._delivery_ledger_key(cfg, "2026-08-22", main.Language.KO),
            main._delivery_ledger_key(cfg, "2026-08-22", main.Language.EN),
            main._delivery_ledger_key(cfg, "2026-08-29", main.Language.KO),
        }
        assert len(keys) == 3

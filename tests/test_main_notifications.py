"""Tests for the alerts, the send phase and the handler that wires them: the
crawl-health alert, the delivery ledger and the alarms the handler raises. Uses
fake ports and a fake boto session — no AWS."""

from __future__ import annotations

import re
from copy import deepcopy

import pytest

from app import main
from app.application import alerts, delivery, pipeline
from app.application.delivery import DeliveryOutcome
from app.domain.constants import Language
from app.domain.crawl import CrawlReport, SourceHealth, SourceStatus

PROJECT = "tech-digest"


class _FakePublisher:
    """An AlarmPublisher that records each alarm, keyed like an SNS publish."""

    def __init__(self):
        self.published: list[dict] = []

    def publish(self, subject, message):
        self.published.append({"Subject": subject, "Message": message})


class _FakeMailer:
    """A Mailer whose sends succeed unless ``results`` says otherwise."""

    def __init__(self, results=None, suppressed=()):
        self.calls: list[tuple] = []
        self._results = iter(results) if results is not None else None
        self._suppressed = set(suppressed)

    def send(self, subject, sender, recipients, html_content):
        self.calls.append((subject, sender, recipients, html_content))
        return next(self._results) if self._results is not None else True

    def suppressed_destinations(self):
        return set(self._suppressed)

    @property
    def sent(self) -> list[str]:
        return [recipients[0] for _s, _f, recipients, _h in self.calls]


class _FakeStore:
    """An ObjectStore holding at most a delivery ledger, recording every access."""

    def __init__(self, ledger=None):
        self.ledger = ledger
        self.reads: list[str] = []
        self.writes: list[dict] = []

    def read_json(self, key):
        self.reads.append(key)
        return self.ledger

    def write_json(self, key, payload):
        # Deep-copy the payload: the real store serializes it immediately, while
        # a kept reference would show the loop's later appends in every write.
        self.writes.append(deepcopy(payload))
        return True

    def download(self, key, local_path):
        raise AssertionError("recipients are stubbed in these tests")

    def upload(self, local_path, prefix=""):
        raise AssertionError("nothing is archived in these tests")


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
        publisher = _FakePublisher()
        alerts.send_crawl_health_alert(publisher, PROJECT, failing_report, [])
        assert len(publisher.published) == 1
        msg = publisher.published[0]
        assert "Crawl Health" in msg["Subject"]
        assert "ALERT" in msg["Subject"]
        assert "ai.meta.com/blog" in msg["Message"]
        assert "anti-bot block" in msg["Message"]

    def test_expected_flaky_source_does_not_alert(self, failing_report):
        """A source that is known to reject AWS egress IPs every week must not
        page, or the alarm becomes noise that hides a genuinely new breakage."""
        publisher = _FakePublisher()
        alerts.send_crawl_health_alert(
            publisher, PROJECT, failing_report, ["ai.meta.com"]
        )
        assert publisher.published == []

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
        publisher = _FakePublisher()
        alerts.send_crawl_health_alert(
            publisher, PROJECT, failing_report, ["ai.meta.com"]
        )
        assert len(publisher.published) == 1
        assert "research.google/blog" in publisher.published[0]["Message"]

    def test_report_failed_partition(self, failing_report):
        assert len(failing_report.failed) == 1
        assert len(failing_report.ok) == 1

    def _empty(self, url: str, candidates: int | None) -> CrawlReport:
        return CrawlReport(
            sources=[
                SourceHealth(
                    url=url,
                    fetcher="RssFetcher",
                    status=SourceStatus.EMPTY,
                    candidates=candidates,
                ),
                SourceHealth(
                    url="https://openai.com/news/rss.xml",
                    fetcher="RssFetcher",
                    status=SourceStatus.OK,
                    post_count=4,
                    candidates=9,
                ),
            ]
        )

    def test_a_source_offering_nothing_alerts_even_though_it_did_not_fail(self):
        """The whole point of the candidate count.

        This source fetched successfully, so it is not FAILED, and it produced no
        posts, so it used to be one more line among the ~12 EMPTY sources a normal
        run has. Nothing paged, and a moved feed could sit there for weeks.
        """
        publisher = _FakePublisher()
        alerts.send_crawl_health_alert(
            publisher, PROJECT, self._empty("https://moved/rss", 0), []
        )
        assert len(publisher.published) == 1
        msg = publisher.published[0]["Message"]
        assert "Sources offering nothing: 1" in msg
        assert "https://moved/rss" in msg

    def test_a_quiet_source_does_not_alert(self):
        """A blog that published nothing must never page, or the alarm fires on the
        ~12 normally-empty sources and becomes noise nobody reads — the same failure
        expected_flaky_urls exists to prevent."""
        publisher = _FakePublisher()
        alerts.send_crawl_health_alert(
            publisher, PROJECT, self._empty("https://quiet/rss", 14), []
        )
        assert publisher.published == []

    def test_an_unknown_count_does_not_alert(self):
        publisher = _FakePublisher()
        alerts.send_crawl_health_alert(
            publisher, PROJECT, self._empty("https://unknown/rss", None), []
        )
        assert publisher.published == []

    def _unreadable(self) -> CrawlReport:
        return CrawlReport(
            sources=[
                SourceHealth(
                    url="https://openai.com/news/rss.xml",
                    fetcher="RssFetcher",
                    status=SourceStatus.OK,
                    post_count=15,
                    unreadable_pages=15,
                )
            ]
        )

    def test_a_source_with_unreadable_pages_alerts(self):
        publisher = _FakePublisher()
        alerts.send_crawl_health_alert(publisher, PROJECT, self._unreadable(), [])
        assert len(publisher.published) == 1
        msg = publisher.published[0]["Message"]
        assert re.search(r"Unreadable article pages:\s+1", msg)
        assert re.search(r"Check these:\s+https://openai.com/news/rss.xml", msg)

    def test_expected_flaky_suppresses_the_unreadable_pages_trigger(self):
        publisher = _FakePublisher()
        alerts.send_crawl_health_alert(
            publisher, PROJECT, self._unreadable(), ["openai.com"]
        )
        assert publisher.published == []

    def test_expected_flaky_suppresses_the_offering_nothing_trigger_too(self):
        """x.ai is blocked from AWS egress IPs every week; whether that shows up as
        FAILED or as zero candidates, it must not page."""
        publisher = _FakePublisher()
        alerts.send_crawl_health_alert(
            publisher,
            PROJECT,
            self._empty("https://x.ai/news", 0),
            ["x.ai/news"],
        )
        assert publisher.published == []


class TestPartialDeliveryAlert:
    def test_alert_on_partial_failure_includes_crawl_summary(self, failing_report):
        publisher = _FakePublisher()
        alerts.maybe_send_partial_delivery_alert(
            publisher,
            PROJECT,
            success_count=2,
            total_recipients=3,
            failed_recipients=["bad@example.com"],
            crawl_report=failing_report,
        )
        assert len(publisher.published) == 1
        published = publisher.published[0]
        assert "ALERT" in published["Subject"]
        msg = published["Message"]
        assert "2/3" in msg
        assert "bad@example.com" in msg
        assert "Crawl health:" in msg
        assert "1 ok" in msg and "1 failed" in msg

    def test_no_alert_on_full_success(self, failing_report):
        publisher = _FakePublisher()
        alerts.maybe_send_partial_delivery_alert(
            publisher,
            PROJECT,
            success_count=3,
            total_recipients=3,
            failed_recipients=[],
            crawl_report=failing_report,
        )
        assert publisher.published == []


class TestEmptyDigestAlert:
    def _post(self, title: str):
        from datetime import datetime

        from app.domain.post import Post

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
        publisher = _FakePublisher()
        filtered_out = [
            (self._post("A"), "Filtering failed (no model response)."),
            (self._post("B"), "Filtering failed (no model response)."),
            (self._post("C"), "Low relevance score."),
        ]
        alerts.maybe_send_empty_digest_alert(
            publisher, PROJECT, self._report(3), filtered_out
        )
        assert len(publisher.published) == 1
        msg = publisher.published[0]["Message"]
        assert "Empty Digest" in publisher.published[0]["Subject"]
        assert "ALERT" in publisher.published[0]["Subject"]
        # Reason breakdown surfaces the dominant cause (mass model failure here).
        assert "2x Filtering failed (no model response)." in msg
        assert "1x Low relevance score." in msg

    def test_no_alert_when_nothing_collected(self):
        # An empty crawl is covered by crawl-health alerts, not this one — even
        # if something was recorded as filtered out.
        publisher = _FakePublisher()
        filtered_out = [(self._post("A"), "Low relevance score.")]
        alerts.maybe_send_empty_digest_alert(
            publisher, PROJECT, self._report(0), filtered_out
        )
        assert publisher.published == []

    def test_no_alert_when_no_filtered_out_posts(self):
        # Defensive: total>0 but nothing recorded as filtered out (shouldn't
        # normally happen) must not fire a misleading alert.
        publisher = _FakePublisher()
        alerts.maybe_send_empty_digest_alert(publisher, PROJECT, self._report(5), [])
        assert publisher.published == []


class TestGeneratePostFilename:
    """Two distinct posts (different links) that sanitize to the same
    source+title must map to DIFFERENT filenames, or one silently overwrites
    the other on disk and vanishes from the digest."""

    def _post(self, title: str, link: str):
        from datetime import datetime

        from app.domain.post import Post

        return Post(
            title=title,
            link=link,
            published_date=datetime(2026, 7, 11),
            source="aws",
        )

    def test_same_title_distinct_links_get_distinct_filenames(self):
        a = self._post("GPT-4 is here!", "https://aws.amazon.com/a")
        b = self._post("GPT-4 is here?", "https://aws.amazon.com/b")
        fa = pipeline._generate_post_filename(a)
        fb = pipeline._generate_post_filename(b)
        assert fa != fb
        assert fa.startswith("aws-gpt-4_is_here-") and fa.endswith(".json")

    def test_filename_is_stable_for_same_link(self):
        p = self._post("Title", "https://aws.amazon.com/x")
        assert pipeline._generate_post_filename(p) == pipeline._generate_post_filename(
            p
        )


class TestEmailOrchestration:
    """process_newsletter_emails counts successes/failures correctly so the
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
        monkeypatch.setattr(delivery.time, "sleep", lambda *_: None)
        outcome = delivery.process_newsletter_emails(
            newsletter,
            "2026-07-11",
            self._Cfg(),
            _FakeStore(),
            _FakeMailer(results=[False, False]),
            Language.KO,
            tmp_path,
            recipients=["a@example.com", "b@example.com"],
        )
        assert outcome.succeeded == 0
        assert outcome.attempted == 2
        assert set(outcome.failed) == {"a@example.com", "b@example.com"}

    def test_unreadable_newsletter_reports_all_failed(self, monkeypatch, tmp_path):
        missing = tmp_path / "does-not-exist.html"
        outcome = delivery.process_newsletter_emails(
            missing,
            "2026-07-11",
            self._Cfg(),
            _FakeStore(),
            _FakeMailer(),
            Language.KO,
            tmp_path,
            recipients=["a@example.com"],
        )
        assert outcome.succeeded == 0
        assert outcome.attempted == 1
        assert outcome.failed == ["a@example.com"]

    def test_partial_success_counts_split(self, monkeypatch, tmp_path):
        newsletter = tmp_path / "n.html"
        newsletter.write_text("<html>body</html>", encoding="utf-8")
        monkeypatch.setattr(delivery.time, "sleep", lambda *_: None)
        outcome = delivery.process_newsletter_emails(
            newsletter,
            "2026-07-11",
            self._Cfg(),
            _FakeStore(),
            # First recipient succeeds, second fails.
            _FakeMailer(results=[True, False]),
            Language.KO,
            tmp_path,
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
        monkeypatch.setattr(delivery.time, "sleep", lambda *_: None)
        store = _FakeStore()
        outcome = delivery.process_newsletter_emails(
            newsletter,
            "2026-07-11",
            self._Cfg(),
            store,
            _FakeMailer(),
            Language.KO,
            tmp_path,
            recipients=["me@example.com"],
        )
        assert outcome.succeeded == 1
        assert store.reads == [] and store.writes == []

    @pytest.mark.parametrize(
        ("header_title", "expected"),
        [
            ("Digest", "[2026-07-11] Digest"),
            # Unset in the stage config: the subject names the title the rendered
            # header falls back to, not "None".
            (None, "[2026-07-11] Weekly AI Tech Blog Digest"),
        ],
    )
    def test_subject_uses_the_rendered_title(
        self, monkeypatch, tmp_path, header_title, expected
    ):
        newsletter = tmp_path / "n.html"
        newsletter.write_text("<html>body</html>", encoding="utf-8")
        mailer = _FakeMailer()
        monkeypatch.setattr(delivery.time, "sleep", lambda *_: None)

        title = header_title

        class _Cfg(self._Cfg):
            class newsletter:
                header_title = title
                sender = "sender@example.com"

        delivery.process_newsletter_emails(
            newsletter,
            "2026-07-11",
            _Cfg(),
            _FakeStore(),
            mailer,
            Language.KO,
            tmp_path,
            recipients=["a@example.com"],
        )
        assert [subject for subject, *_ in mailer.calls] == [expected]


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
        from app.domain.crawl import CrawlReport

        monkeypatch.setattr(
            main,
            "fetch_and_filter_posts",
            lambda *a, **k: ([], "2026-06-01", [], CrawlReport()),
        )
        result = main.handler({}, None)
        assert result["statusCode"] == 200
        assert "No posts" in result["body"]

    def test_exception_returns_500(self, monkeypatch):
        self._patch_common(monkeypatch)

        def _boom(*a, **k):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(main, "fetch_and_filter_posts", _boom)
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
        monkeypatch.setattr(main, "_boto_sessions", lambda _cfg: (session, session))
        monkeypatch.setattr(main, "_setup_aws_env", lambda *a, **k: None)

        from datetime import datetime

        from app.domain.post import Post

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
            "fetch_and_filter_posts",
            lambda *a, **k: ([], "2026-07-11", filtered_out, report),
        )

        result = main.handler({}, None)
        assert result["statusCode"] == 200
        assert len(session.sns.published) == 1
        assert "Empty Digest" in session.sns.published[0]["Subject"]

    @pytest.mark.parametrize(
        "source",
        [
            # Fetched fine but offered nothing (candidates == 0): dead, not quiet.
            SourceHealth(
                url="https://moved.example/feed",
                fetcher="RssFetcher",
                status=SourceStatus.EMPTY,
                candidates=0,
            ),
            # Fetched fine, but no post's article page could be read.
            SourceHealth(
                url="https://blocked.example/feed",
                fetcher="RssFetcher",
                status=SourceStatus.OK,
                post_count=3,
                unreadable_pages=3,
            ),
        ],
        ids=["likely_broken", "pages_unreadable"],
    )
    def test_a_source_that_did_not_fail_alone_raises_the_crawl_health_alert(
        self, monkeypatch, source
    ):
        # No source FAILED; the handler must still page on either signal.
        self._patch_common(monkeypatch)

        class _Cfg:
            class resources:
                project_name = PROJECT
                profile_name = None
                default_region_name = "ap-northeast-2"
                bedrock_region_name = "us-west-2"

            class scraping:
                expected_flaky_urls = ()

        monkeypatch.setattr(main.Config, "load", staticmethod(lambda: _Cfg()))
        monkeypatch.setattr(main, "is_running_in_aws", lambda: True)
        monkeypatch.setenv(main.EnvVars.TOPIC_ARN.value, "arn:topic")
        session = _FakeSession()
        monkeypatch.setattr(main, "_boto_sessions", lambda _cfg: (session, session))
        monkeypatch.setattr(main, "_setup_aws_env", lambda *a, **k: None)
        report = CrawlReport(sources=[source])
        monkeypatch.setattr(
            main,
            "fetch_and_filter_posts",
            lambda *a, **k: ([], "2026-07-11", [], report),
        )
        main.handler({}, None)
        assert [m["Subject"] for m in session.sns.published] == [
            f"[{PROJECT}] Crawl Health — ALERT"
        ]

    def test_exception_publishes_failure_notification(self, monkeypatch):
        # On AWS with a topic configured, an unhandled error must surface via SNS
        # (the "failures always surface" design promise), not just return 500.
        self._patch_common(monkeypatch)
        monkeypatch.setattr(main, "is_running_in_aws", lambda: True)
        monkeypatch.setenv(main.EnvVars.TOPIC_ARN.value, "arn:topic")
        # On AWS the handler uses the module-global session, not boto3.Session().
        session = _FakeSession()
        monkeypatch.setattr(main, "_boto_sessions", lambda _cfg: (session, session))

        def _boom(*a, **k):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(main, "fetch_and_filter_posts", _boom)
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
        from app.configs import Config

        # The committed CI config, not the developer's: config-dev.yaml is
        # gitignored, so loading it passes locally and fails in CI.
        monkeypatch.setenv("CONFIG_FILE_SUFFIX", "ci-batch")
        publisher = _FakePublisher()
        config = Config.load()
        alerts.send_no_recipients_alert(
            publisher, config, self._report(), DeliveryOutcome()
        )
        assert len(publisher.published) == 1
        published = publisher.published[0]
        assert "No Recipients" in published["Subject"]
        assert "ALERT" in published["Subject"]
        # The cause is always outside the code, so the alarm has to point at it.
        assert "s3://" in published["Message"]
        assert "recipients" in published["Message"].lower()
        assert "0/0" in published["Message"]

    def test_partial_delivery_alert_cannot_cover_this_case(self, failing_report):
        """Pins the reason a separate alarm is needed rather than reusing that one."""
        publisher = _FakePublisher()
        alerts.maybe_send_partial_delivery_alert(
            publisher, PROJECT, 0, 0, [], failing_report
        )
        assert publisher.published == []


class TestClippedNewsletterAlert:
    """Mail clients truncate an over-large message, so the tail of the issue is
    replaced by a "View entire message" link. Nothing about the send fails, and
    the renderer's log line is seen by nobody on a weekly unattended job. A real
    prod issue (2026-06-02) shipped at 101.4% of the limit."""

    def test_no_alert_under_the_limit(self, tmp_path):
        from app.domain.newsletter import GMAIL_CLIP_BYTES

        path = tmp_path / "n.html"
        path.write_bytes(b"x" * (GMAIL_CLIP_BYTES - 1))
        publisher = _FakePublisher()
        alerts.maybe_send_clipped_newsletter_alert(publisher, PROJECT, path, 3)
        assert publisher.published == []

    def test_alert_over_the_limit_reports_size_and_remedy(self, tmp_path):
        from app.domain.newsletter import GMAIL_CLIP_BYTES

        path = tmp_path / "n.html"
        path.write_bytes(b"x" * (GMAIL_CLIP_BYTES + 2048))
        publisher = _FakePublisher()
        alerts.maybe_send_clipped_newsletter_alert(publisher, PROJECT, path, 5)
        assert len(publisher.published) == 1
        published = publisher.published[0]
        assert "Newsletter Clipped" in published["Subject"]
        assert "max_posts" in published["Message"]
        assert "5" in published["Message"]

    def test_missing_file_does_not_raise(self, tmp_path):
        """A size check must never be the thing that fails a successful run."""
        publisher = _FakePublisher()
        alerts.maybe_send_clipped_newsletter_alert(
            publisher, PROJECT, tmp_path / "absent.html", 3
        )
        assert publisher.published == []


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

    def _run(self, monkeypatch, tmp_path, ledger, send_results=None, suppressed=()):
        """Run the send phase against a fake ledger; returns (outcome, sent, writes)."""
        newsletter = tmp_path / "n.html"
        newsletter.write_text("<html>body</html>", encoding="utf-8")
        mailer = _FakeMailer(send_results, suppressed)
        store = _FakeStore(ledger)
        monkeypatch.setattr(delivery.time, "sleep", lambda *_: None)
        outcome = delivery.process_newsletter_emails(
            newsletter,
            "2026-08-22",
            self._Cfg(),
            store,
            mailer,
            Language.KO,
            tmp_path,
            recipients=None,
        )
        return outcome, mailer.sent, store.writes

    def _patch_recipients(self, monkeypatch, recipients):
        monkeypatch.setattr(
            delivery, "_get_recipients", lambda *a, **k: list(recipients)
        )

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
        outcome, sent, writes = self._run(
            monkeypatch, tmp_path, ledger={"delivered": ["a@example.com"]}
        )
        assert sent == ["b@example.com", "c@example.com"]
        assert outcome.skipped == 1
        assert outcome.succeeded == 2
        # The ledger is overwritten, not merged, so the reader the crashed run
        # already served must be carried forward — or a second retry re-sends.
        assert writes[-1]["delivered"] == [
            "a@example.com",
            "b@example.com",
            "c@example.com",
        ]

    def test_the_ledger_is_written_after_every_success_not_once_at_the_end(
        self, monkeypatch, tmp_path
    ):
        """A ledger written only at the end is absent exactly when it matters."""
        self._patch_recipients(monkeypatch, ["a@example.com", "b@example.com"])
        _outcome, _sent, writes = self._run(monkeypatch, tmp_path, ledger=None)
        assert len(writes) == 2
        assert writes[0]["delivered"] == ["a@example.com"]
        assert writes[1]["delivered"] == ["a@example.com", "b@example.com"]

    def test_a_suppressed_address_is_skipped_not_sent(self, monkeypatch, tmp_path):
        self._patch_recipients(monkeypatch, ["ok@example.com", "Gone@example.com"])
        outcome, sent, writes = self._run(
            monkeypatch, tmp_path, ledger=None, suppressed={"gone@example.com"}
        )
        assert sent == ["ok@example.com"]
        assert outcome.suppressed == 1
        assert outcome.attempted == 1
        assert outcome.failed == []
        assert writes[-1]["delivered"] == ["ok@example.com"]

    def test_a_failed_send_is_not_recorded_as_delivered(self, monkeypatch, tmp_path):
        """Otherwise the retry would skip the reader who never got the issue."""
        self._patch_recipients(monkeypatch, ["ok@example.com", "bad@example.com"])
        outcome, _sent, writes = self._run(
            monkeypatch, tmp_path, ledger=None, send_results=[True, False]
        )
        assert outcome.failed == ["bad@example.com"]
        assert writes[-1]["delivered"] == ["ok@example.com"]

    def test_a_failure_before_a_success_is_not_carried_into_the_ledger(
        self, monkeypatch, tmp_path
    ):
        # Failure first: the write that follows the later success must not list
        # the reader who never got the issue, or the retry skips them for good.
        self._patch_recipients(monkeypatch, ["bad@example.com", "ok@example.com"])
        outcome, _sent, writes = self._run(
            monkeypatch, tmp_path, ledger=None, send_results=[False, True]
        )
        assert outcome.failed == ["bad@example.com"]
        assert writes[-1]["delivered"] == ["ok@example.com"]

    @pytest.mark.parametrize(
        ("recipient", "ledger_entry"),
        [
            ("reader@example.com", "READER@Example.com"),
            ("Reader@Example.COM", "reader@example.com"),
        ],
        ids=["ledger-cased", "recipient-cased"],
    )
    def test_ledger_entries_are_compared_case_insensitively(
        self, monkeypatch, tmp_path, recipient, ledger_entry
    ):
        self._patch_recipients(monkeypatch, [recipient])
        outcome, sent, _ = self._run(
            monkeypatch, tmp_path, ledger={"delivered": [ledger_entry]}
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
        key = delivery._delivery_ledger_key(self._Cfg(), "2026-08-22", Language.EN)
        assert key == "tech-digest/deliveries/delivered-prod-2026-08-22-en.json"

    def test_ledger_key_without_an_s3_prefix_has_no_leading_slash(self):
        class _NoPrefix(self._Cfg):
            class resources:
                stage = "dev"
                s3_bucket_name = "bucket"
                s3_prefix = None

        key = delivery._delivery_ledger_key(_NoPrefix(), "2026-08-22", Language.KO)
        assert key == "deliveries/delivered-dev-2026-08-22-ko.json"

    def test_a_different_issue_gets_a_different_ledger(self):
        cfg = self._Cfg()
        keys = {
            delivery._delivery_ledger_key(cfg, "2026-08-22", Language.KO),
            delivery._delivery_ledger_key(cfg, "2026-08-22", Language.EN),
            delivery._delivery_ledger_key(cfg, "2026-08-29", Language.KO),
        }
        assert len(keys) == 3


class TestDeliveryAlertsFromTheHandler:
    """The handler, not just the alert helpers, decides which delivery alarm fires.

    Each case stubs the send phase with the outcome it would report and checks
    what reaches SNS.
    """

    class _Cfg:
        class resources:
            project_name = PROJECT
            profile_name = None
            default_region_name = "ap-northeast-2"
            bedrock_region_name = "us-west-2"
            stage = "prod"
            s3_bucket_name = "bucket"
            s3_prefix = "tech-digest"

        class scraping:
            expected_flaky_urls = ()

        class newsletter:
            send_emails = True

    def _alerts(self, monkeypatch, tmp_path, outcome) -> list[str]:
        from app.domain.post import Post

        session = _FakeSession()
        post = Post(title="t", link="https://x.example/p", published_date="2026-10-09")
        newsletter = tmp_path / "n.html"
        newsletter.write_text("<html></html>", encoding="utf-8")
        monkeypatch.setattr(main, "is_running_in_aws", lambda: True)
        monkeypatch.setenv(main.EnvVars.TOPIC_ARN.value, "arn:topic")
        monkeypatch.setattr(main, "_boto_sessions", lambda _cfg: (session, session))
        monkeypatch.setattr(main.Config, "load", classmethod(lambda cls: self._Cfg()))
        monkeypatch.setattr(main, "_setup_aws_env", lambda *a: None)
        monkeypatch.setattr(main, "archive_suffix", lambda *a: "")
        monkeypatch.setattr(
            main,
            "fetch_and_filter_posts",
            lambda *a, **k: ([post], "2026-10-10", [], CrawlReport()),
        )
        monkeypatch.setattr(
            main, "process_posts_and_create_newsletter", lambda *a, **k: newsletter
        )
        monkeypatch.setattr(main, "process_newsletter_emails", lambda *a: outcome)
        assert main.handler({}, None)["statusCode"] == 200
        return [m["Subject"] for m in session.sns.published]

    def test_no_recipients_alerts_once(self, monkeypatch, tmp_path):
        assert self._alerts(monkeypatch, tmp_path, DeliveryOutcome()) == [
            f"[{PROJECT}] No Recipients — ALERT"
        ]

    def test_a_failed_recipient_alerts_partial_delivery(self, monkeypatch, tmp_path):
        outcome = DeliveryOutcome(
            resolved=2, attempted=2, succeeded=1, failed=["b@example.com"]
        )
        assert self._alerts(monkeypatch, tmp_path, outcome) == [
            f"[{PROJECT}] Newsletter Delivery — ALERT"
        ]

    def test_a_retry_that_skipped_everyone_is_silent(self, monkeypatch, tmp_path):
        outcome = DeliveryOutcome(resolved=2, skipped=2)
        assert self._alerts(monkeypatch, tmp_path, outcome) == []

    def test_a_retry_that_served_everyone_but_the_suppressed_is_silent(
        self, monkeypatch, tmp_path
    ):
        outcome = DeliveryOutcome(resolved=3, skipped=2, suppressed=1)
        assert self._alerts(monkeypatch, tmp_path, outcome) == []

    def test_a_fully_suppressed_list_alerts(self, monkeypatch, tmp_path):
        # Resolved, but every address is on the SES suppression list: nobody gets
        # the issue, and there is no failed send for partial-delivery to report.
        outcome = DeliveryOutcome(resolved=2, suppressed=2)
        assert self._alerts(monkeypatch, tmp_path, outcome) == [
            f"[{PROJECT}] No Recipients — ALERT"
        ]

    def test_the_suppressed_alert_says_why(self, monkeypatch):
        from app.configs import Config

        monkeypatch.setenv("CONFIG_FILE_SUFFIX", "ci-batch")
        publisher = _FakePublisher()
        alerts.send_no_recipients_alert(
            publisher,
            Config.load(),
            CrawlReport(),
            DeliveryOutcome(resolved=3, skipped=1, suppressed=2),
        )
        [published] = publisher.published
        assert "0/2" in published["Message"]
        assert "Suppressed by SES" in published["Message"]

    def test_the_send_phase_reports_a_fully_suppressed_list(
        self, monkeypatch, tmp_path
    ):
        newsletter = tmp_path / "n.html"
        newsletter.write_text("<html>body</html>", encoding="utf-8")
        recipients = ["a@example.com", "b@example.com"]
        mailer = _FakeMailer(suppressed=recipients)
        monkeypatch.setattr(
            delivery, "_get_recipients", lambda *a, **k: list(recipients)
        )

        outcome = delivery.process_newsletter_emails(
            newsletter,
            "2026-10-10",
            TestDuplicateSendGuard._Cfg(),
            _FakeStore(),
            mailer,
            Language.KO,
            tmp_path,
        )
        assert mailer.calls == []
        assert outcome == DeliveryOutcome(resolved=2, suppressed=2)

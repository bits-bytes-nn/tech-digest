"""Guards on the paths between a built issue and a reader's inbox.

Each test pins a failure that used to be silent or to reach the wrong people: a
test send falling back to the full list, a failed run exiting 0, a duplicate
address, an alert error aborting delivery, a greeting error losing the issue,
and test runs writing into the real log file. Fake sessions only — no AWS.
"""

from __future__ import annotations

import logging
import runpy
import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import main  # noqa: E402

from src.constants import Language  # noqa: E402


@pytest.fixture(autouse=True)
def no_suppression_list(monkeypatch):
    """The send loop reads the SES suppression list; these tests use fake sessions."""
    monkeypatch.setattr(main, "suppressed_destinations", lambda _session: set())


class TestRunBatchRecipients:
    def test_only_invalid_addresses_refuse_to_fall_back(self, monkeypatch):
        # "--recipients" with nothing valid used to become "null", which means
        # "everyone on the S3 list". It must stop before anything is submitted.
        from configs import Config

        class _Submitted(BaseException):
            """Not an Exception, so run_batch's own handler cannot swallow it."""

        def _reached(*args, **kwargs):
            raise _Submitted

        monkeypatch.setattr(Config, "load", classmethod(_reached))
        monkeypatch.setattr(
            sys,
            "argv",
            ["run_batch.py", "--recipients", "me@exmaple", "not-an-address"],
        )
        with pytest.raises(SystemExit) as exc:
            runpy.run_path(str(APP_DIR / "run_batch.py"), run_name="__main__")
        assert exc.value.code == 1


class TestCliExitCode:
    @pytest.mark.parametrize(("status", "code"), [(200, 0), (500, 1)])
    def test_exit_code_follows_the_handler(self, monkeypatch, status, code):
        events = []

        def handler(event, context):
            events.append(event)
            return {"statusCode": status, "body": ""}

        monkeypatch.setattr(main, "handler", handler)
        assert main.cli(["--recipients", "me@example.com", "you@example.com"]) == code
        assert events[0]["RECIPIENTS"] == "me@example.com,you@example.com"

    def test_null_sentinels_mean_no_override(self, monkeypatch):
        events = []
        monkeypatch.setattr(
            main, "handler", lambda e, c: events.append(e) or {"statusCode": 200}
        )
        main.cli(["--end-date", "null", "--language", "null", "--recipients", "null"])
        assert events[0] == {"END_DATE": None, "LANGUAGE": None, "RECIPIENTS": None}


class TestUniqueRecipients:
    def test_case_insensitive_and_first_spelling_kept(self):
        assert main._unique(["A@x.com", "b@x.com", "a@x.com", " b@x.com"]) == [
            "A@x.com",
            "b@x.com",
        ]


class TestAlertsNeverAbortDelivery:
    def test_publish_error_is_logged_not_raised(self, caplog):
        class _BrokenSNS:
            def publish(self, **kwargs):
                raise RuntimeError("throttled")

        class _Session:
            def client(self, service):
                return _BrokenSNS()

        with caplog.at_level(logging.ERROR, logger="app"):
            main._publish_alarm(
                _Session(),
                "arn:aws:sns:us-east-1:0:t",
                project="p",
                event="Crawl Health",
                status="ALERT",
                fields={"a": "b"},
            )
        assert "Could not publish" in caplog.text


class TestGreetingFailureKeepsTheIssue:
    @pytest.mark.parametrize("language", [Language.KO, Language.EN])
    def test_fixed_intro_in_the_issues_language(self, monkeypatch, language):
        class _FailingGreeter:
            def __init__(self, *args, **kwargs):
                pass

            def greet(self, context=None):
                raise RuntimeError("Bedrock unavailable")

        monkeypatch.setattr(main, "Greeter", _FailingGreeter)
        monkeypatch.setattr(
            main, "_save_greeting_to_file", lambda *a: pytest.fail("nothing to save")
        )

        class _Config:
            class summarization:
                greeting_model_id = "unused"

        intro = main._generate_greeting([], "2026-10-10", _Config(), None, language)
        assert intro == main.FALLBACK_INTRO[language]


class TestTitleCaseFailureKeepsTheTitles:
    def test_an_unavailable_caser_leaves_titles_alone(self, monkeypatch):
        from src.feed_parser import Post

        def _broken(*args, **kwargs):
            raise ValueError("Unsupported model ID")

        monkeypatch.setattr(main, "TitleCaser", _broken)

        class _Config:
            class summarization:
                greeting_model_id = "unused"

        post = Post(
            title="a guide", link="https://x.example/p", published_date="2026-10-09"
        )
        main._title_case_titles([post], _Config(), None)
        assert post.title == "a guide"


class TestTestRunsKeepOutOfTheLogFile:
    def test_no_file_handler_under_pytest(self):
        from src.logger import configure_logger

        configured = configure_logger("tech-digest-probe")
        assert not any(isinstance(h, logging.FileHandler) for h in configured.handlers)


class TestRecipientsOverrideInTheContainer:
    """The container entrypoint's own guard, independent of run_batch."""

    class _Cfg:
        class resources:
            stage = "dev"
            s3_bucket_name = "bucket"
            s3_prefix = "tech-digest"
            profile_name = None
            default_region_name = "ap-northeast-2"
            bedrock_region_name = "us-west-2"
            project_name = "tech-digest"

        class newsletter:
            send_emails = True

    def test_an_all_invalid_override_never_falls_back_to_the_list(self, monkeypatch):
        downloads = []
        monkeypatch.setattr(
            main,
            "_download_recipients_file",
            lambda *a, **k: downloads.append(a) or True,
        )
        assert main._get_recipients(self._Cfg(), object(), ["not-an-address"]) == []
        assert downloads == []

    def test_the_override_reaches_the_send_phase_and_archives_as_a_test(
        self, monkeypatch, tmp_path
    ):
        from src.feed_parser import CrawlReport, Post

        seen: dict = {}
        post = Post(title="t", link="https://x.example/p", published_date="2026-10-09")
        monkeypatch.setattr(main.Config, "load", classmethod(lambda cls: self._Cfg()))
        monkeypatch.setattr(main, "_setup_aws_env", lambda *a: None)
        monkeypatch.setattr(main, "_log_run_summary", lambda *a: None)
        monkeypatch.setattr(
            main,
            "_fetch_and_filter_posts",
            lambda *a, **k: ([post], "2026-10-10", [], CrawlReport()),
        )

        def build(*a, **k):
            seen["archive_suffix"] = k["archive_suffix"]
            return tmp_path / "n.html"

        def send(_path, _date, _cfg, _session, _language, recipients):
            seen["recipients"] = recipients
            return main.DeliveryOutcome(resolved=1, attempted=1, succeeded=1)

        monkeypatch.setattr(main, "_process_posts_and_create_newsletter", build)
        monkeypatch.setattr(main, "_process_newsletter_emails", send)
        result = main.handler({"RECIPIENTS": "me@example.com"}, None)
        assert result["statusCode"] == 200
        assert seen == {"archive_suffix": "tests", "recipients": ["me@example.com"]}


class TestArchiveKeys:
    """Only the run that delivers an issue may write its canonical archive key:
    the bucket keeps no versions, and a rebuild can carry different posts."""

    _Cfg = TestRecipientsOverrideInTheContainer._Cfg

    def _suffix(self, monkeypatch, delivered, recipients=None, attempt=None):
        monkeypatch.setattr(main, "_already_delivered", lambda *a: set(delivered))
        if attempt:
            monkeypatch.setenv("AWS_BATCH_JOB_ATTEMPT", attempt)
        else:
            monkeypatch.delenv("AWS_BATCH_JOB_ATTEMPT", raising=False)
        return main._archive_suffix(
            self._Cfg(), object(), "2026-10-10", Language.KO, recipients
        )

    def test_the_scheduled_first_attempt_writes_the_canonical_key(self, monkeypatch):
        assert self._suffix(monkeypatch, delivered=[]) == ""

    def test_a_test_send_archives_under_tests(self, monkeypatch):
        assert self._suffix(
            monkeypatch, delivered=["a@x.com"], recipients="me@x.com"
        ) == ("tests")

    def test_a_retry_after_sending_does_not_replace_the_delivered_copy(
        self, monkeypatch
    ):
        assert self._suffix(monkeypatch, delivered=["a@x.com"], attempt="2") == (
            "retries/2"
        )

    @pytest.mark.parametrize(
        ("suffix", "expected"),
        [
            ("", ["newsletters", "articles"]),
            ("tests", ["newsletters/tests", "articles/tests"]),
        ],
    )
    def test_uploads_use_the_suffix(self, monkeypatch, tmp_path, suffix, expected):
        prefixes = []
        newsletter = tmp_path / "n.html"
        article = tmp_path / "articles" / "a.html"
        monkeypatch.setattr(main, "_prepare_and_save_posts", lambda *a: tmp_path)
        monkeypatch.setattr(main, "_generate_greeting", lambda *a: "hi")
        monkeypatch.setattr(main, "ROOT_DIR", tmp_path)
        monkeypatch.setattr(
            main, "_build_newsletter", lambda *a: (newsletter, [article.name])
        )
        monkeypatch.setattr(
            main,
            "_upload_files_to_s3",
            lambda _paths, prefix, *a: prefixes.append(prefix),
        )
        main._process_posts_and_create_newsletter(
            [], "2026-10-10", self._Cfg(), None, None, archive_suffix=suffix
        )
        assert prefixes == expected


class TestModelFailureAlert:
    class _SNS:
        def __init__(self):
            self.published: list[dict] = []

        def publish(self, **kwargs):
            self.published.append(kwargs)

    class _Session:
        def __init__(self, sns):
            self.sns = sns

        def client(self, name):
            return self.sns

    def _post(self, title):
        from src.feed_parser import Post

        return Post(
            title=title, link=f"https://x.example/{title}", published_date="2026-10-09"
        )

    def test_a_shipped_issue_with_failed_calls_alerts(self):
        sns = self._SNS()
        main._maybe_send_model_failure_alert(
            self._Session(sns),
            "arn:topic",
            "tech-digest",
            [self._post("kept")],
            [
                (self._post("lost"), "Summarization failed (no model response)."),
                (self._post("low"), "Off topic."),
            ],
        )
        [message] = sns.published
        assert "Model Failures" in message["Subject"]
        assert "lost" in message["Message"] and "low" not in message["Message"]

    def test_verdicts_alone_do_not_alert(self):
        sns = self._SNS()
        main._maybe_send_model_failure_alert(
            self._Session(sns),
            "arn:topic",
            "tech-digest",
            [self._post("kept")],
            [(self._post("low"), "Off topic.")],
        )
        assert sns.published == []


class TestModelFailureAlertFromTheHandler:
    def test_a_shipped_issue_that_lost_a_post_pages(self, monkeypatch, tmp_path):
        from src.feed_parser import CrawlReport, Post

        sns = TestModelFailureAlert._SNS()
        session = TestModelFailureAlert._Session(sns)

        class _Cfg:
            class resources:
                project_name = "tech-digest"
                profile_name = None
                default_region_name = "ap-northeast-2"
                bedrock_region_name = "us-west-2"

            class scraping:
                expected_flaky_urls = ()

            class newsletter:
                send_emails = False

        kept = Post(
            title="kept", link="https://x.example/k", published_date="2026-10-09"
        )
        lost = Post(
            title="lost", link="https://x.example/l", published_date="2026-10-09"
        )
        newsletter = tmp_path / "n.html"
        newsletter.write_text("<html></html>", encoding="utf-8")
        monkeypatch.setattr(main, "is_running_in_aws", lambda: True)
        monkeypatch.setenv(main.EnvVars.TOPIC_ARN.value, "arn:topic")
        monkeypatch.setattr(main, "_boto_sessions", lambda _cfg: (session, session))
        monkeypatch.setattr(main.Config, "load", classmethod(lambda cls: _Cfg()))
        monkeypatch.setattr(main, "_setup_aws_env", lambda *a: None)
        monkeypatch.setattr(main, "_log_run_summary", lambda *a: None)
        monkeypatch.setattr(main, "_archive_suffix", lambda *a: "")
        monkeypatch.setattr(
            main,
            "_fetch_and_filter_posts",
            lambda *a, **k: (
                [kept],
                "2026-10-10",
                [(lost, "Summarization failed (no model response).")],
                CrawlReport(),
            ),
        )
        monkeypatch.setattr(
            main, "_process_posts_and_create_newsletter", lambda *a, **k: newsletter
        )
        assert main.handler({}, None)["statusCode"] == 200
        assert [m["Subject"] for m in sns.published] == [
            "[tech-digest] Model Failures — WARN"
        ]

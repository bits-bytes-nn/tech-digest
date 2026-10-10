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


class TestTestRunsKeepOutOfTheLogFile:
    def test_no_file_handler_under_pytest(self):
        from src.logger import configure_logger

        configured = configure_logger("tech-digest-probe")
        assert not any(isinstance(h, logging.FileHandler) for h in configured.handlers)

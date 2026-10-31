"""Tests for run_batch: the "null"-sentinel parameter encoding (round-tripped
through main.cli's decode), date validation, and the submit-and-wait flow."""

from __future__ import annotations

import runpy
import sys

import boto3
import pytest

import app.logger as app_logger
from app import main as app_main
from app import run_batch
from app.adapters import aws
from app.configs import Config


class TestSanitizeParameters:
    def test_none_becomes_null_sentinel(self):
        out = run_batch.sanitize_parameters({"end_date": None})
        assert out["end_date"] == "null"

    def test_comma_joined_recipients_pass_through(self):
        out = run_batch.sanitize_parameters({"recipients": "a@x.com,b@y.com"})
        assert out["recipients"] == "a@x.com,b@y.com"

    def test_plain_value_stringified(self):
        out = run_batch.sanitize_parameters({"language": "ko"})
        assert out["language"] == "ko"


class TestValidateDate:
    def test_valid_date(self):
        assert run_batch.is_valid_date("2026-06-01") is True

    def test_null_sentinel_is_valid(self):
        assert run_batch.is_valid_date("null") is True

    def test_empty_is_valid(self):
        assert run_batch.is_valid_date("") is True

    def test_bad_format_invalid(self):
        assert run_batch.is_valid_date("06/01/2026") is False

    def test_garbage_invalid(self):
        assert run_batch.is_valid_date("not-a-date") is False


def _container_argv(params: dict[str, str]) -> list[str]:
    """What Batch hands main.py after substituting the job command's ``Ref::``s
    (see the job definition in scripts/deploy_infra.py)."""
    return [
        "--end-date",
        params["end_date"],
        "--language",
        params["language"],
        "--recipients",
        params["recipients"],
    ]


class TestParametersRoundTrip:
    """What run_batch submits must decode in main.cli to what the operator asked."""

    def _decoded(self, monkeypatch, **kwargs) -> dict:
        events = []
        monkeypatch.setattr(
            app_main, "handler", lambda e, c: events.append(e) or {"statusCode": 200}
        )
        app_main.cli(_container_argv(run_batch.sanitize_parameters(kwargs)))
        return events[0]

    def test_recipients_reach_the_handler(self, monkeypatch):
        event = self._decoded(
            monkeypatch,
            end_date="2026-10-31",
            language="en",
            recipients="a@x.com,b@y.com",
        )
        assert event == {
            "END_DATE": "2026-10-31",
            "LANGUAGE": "en",
            "RECIPIENTS": "a@x.com,b@y.com",
        }

    def test_unset_parameters_decode_to_none(self, monkeypatch):
        event = self._decoded(
            monkeypatch, end_date=None, language=None, recipients=None
        )
        assert event == {"END_DATE": None, "LANGUAGE": None, "RECIPIENTS": None}


class _Cfg:
    class resources:
        project_name = "tech-digest"
        stage = "dev"
        profile_name = None
        default_region_name = "ap-northeast-2"


@pytest.fixture
def batch_calls(monkeypatch) -> dict:
    """Stub config, session, SSM and Batch both on ``run_batch`` and in the
    modules that define them, which a ``__main__`` re-execution imports from."""
    calls: dict = {"submitted": [], "waited": []}
    monkeypatch.setattr(Config, "load", classmethod(lambda cls: _Cfg()))
    monkeypatch.setattr(boto3, "Session", lambda **k: object())
    ssm = {
        "/tech-digest/dev/batch-job-queue": "queue",
        "/tech-digest/dev/batch-job-definition": "definition",
    }
    calls["ssm"] = ssm

    def get_param(_session, name):
        return ssm[name]

    def submit(_session, name, queue, definition, parameters):
        calls["submitted"].append(
            {"name": name, "queue": queue, "def": definition, "params": parameters}
        )
        return calls.get("job_id", "job-1")

    def wait(_session, job_id):
        calls["waited"].append(job_id)
        return calls.get("wait_result", True)

    for module in (aws, run_batch):
        monkeypatch.setattr(module, "get_ssm_param_value", get_param)
        monkeypatch.setattr(module, "submit_batch_job", submit)
        monkeypatch.setattr(module, "wait_for_batch_job_completion", wait)
    for module in (app_logger, run_batch):
        monkeypatch.setattr(module, "is_running_in_aws", lambda: False)
    return calls


class TestMain:
    def test_submits_and_waits(self, batch_calls):
        run_batch.main("newsletter", end_date=None, language="ko", recipients=None)
        [submitted] = batch_calls["submitted"]
        assert submitted["queue"] == "queue"
        assert submitted["def"] == "definition"
        assert submitted["name"].startswith("tech-digest-dev-newsletter-")
        assert submitted["params"] == {
            "end_date": "null",
            "language": "ko",
            "recipients": "null",
        }
        assert batch_calls["waited"] == ["job-1"]

    def test_an_ssm_lookup_failure_raises_value_error(self, batch_calls):
        batch_calls["ssm"].clear()
        with pytest.raises(ValueError, match="SSM"):
            run_batch.main("newsletter")
        assert batch_calls["submitted"] == []

    def test_a_failed_submit_raises(self, batch_calls):
        batch_calls["job_id"] = None
        with pytest.raises(RuntimeError, match="submit"):
            run_batch.main("newsletter")
        assert batch_calls["waited"] == []

    def test_a_failed_job_raises(self, batch_calls):
        batch_calls["wait_result"] = False
        with pytest.raises(RuntimeError, match="job-1"):
            run_batch.main("newsletter")


class TestEntryPoint:
    def _run(self, monkeypatch, *argv: str) -> None:
        monkeypatch.setattr(sys, "argv", ["run_batch.py", *argv])
        runpy.run_path(run_batch.__file__, run_name="__main__")

    def test_valid_recipients_are_submitted_comma_joined(
        self, monkeypatch, batch_calls
    ):
        self._run(
            monkeypatch,
            "--end-date",
            "2026-10-31",
            "--recipients",
            "a@example.com",
            "not-an-address",
            "b@example.com",
        )
        [submitted] = batch_calls["submitted"]
        assert submitted["params"] == {
            "end_date": "2026-10-31",
            "language": "null",
            "recipients": "a@example.com,b@example.com",
        }

    def test_a_bad_date_exits_before_submitting(self, monkeypatch, batch_calls):
        with pytest.raises(SystemExit) as exc:
            self._run(monkeypatch, "--end-date", "31/10/2026")
        assert exc.value.code == 1
        assert batch_calls["submitted"] == []

    def test_a_failed_job_exits_1(self, monkeypatch, batch_calls):
        batch_calls["wait_result"] = False
        with pytest.raises(SystemExit) as exc:
            self._run(monkeypatch)
        assert exc.value.code == 1

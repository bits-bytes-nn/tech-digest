"""Tests for aws_helpers — pure logic around boto3 (SES MIME building, S3
key/prefix handling, 404-vs-error branches, batch polling state machine).
Uses fake boto clients; no real AWS."""

from __future__ import annotations

from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from app.src.aws_helpers import (
    check_and_download_from_s3,
    get_ssm_param_value,
    send_email,
    submit_batch_job,
    upload_to_s3,
    wait_for_batch_job_completion,
)


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, "Op")


class _FakeSession:
    def __init__(self, client):
        self._client = client

    def client(self, service):
        return self._client


# --------------------------------------------------------------------------- #
# send_email
# --------------------------------------------------------------------------- #
class _FakeSES:
    def __init__(self):
        self.sent: list[dict] = []

    def send_raw_email(self, **kwargs):
        self.sent.append(kwargs)
        return {"MessageId": "msg-123"}


class TestSendEmail:
    def test_no_recipients_returns_false(self):
        ses = _FakeSES()
        ok = send_email(_FakeSession(ses), "subj", "a@b.com", [], "<p>x</p>")
        assert ok is False
        assert ses.sent == []

    def test_sends_html_multipart(self):
        ses = _FakeSES()
        ok = send_email(
            _FakeSession(ses),
            "subj",
            "from@example.com",
            ["to@example.com"],
            "<p>hi</p>",
        )
        assert ok is True
        raw = ses.sent[0]["RawMessage"]["Data"]
        assert "text/html" in raw
        assert ses.sent[0]["Source"] == "from@example.com"
        assert ses.sent[0]["Destinations"] == ["to@example.com"]

    def test_message_id_when_sender_lacks_at(self):
        ses = _FakeSES()
        # Sender without '@' must still build a valid Message-ID (no crash).
        ok = send_email(
            _FakeSession(ses), "s", "no-at-sign", ["to@example.com"], "<p>x</p>"
        )
        assert ok is True
        assert "Message-ID" in ses.sent[0]["RawMessage"]["Data"]

    def test_client_error_returns_false(self):
        class _Boom:
            def send_raw_email(self, **kwargs):
                raise _client_error("MessageRejected")

        ok = send_email(_FakeSession(_Boom()), "s", "a@b.com", ["c@d.com"], "<p>x</p>")
        assert ok is False


# --------------------------------------------------------------------------- #
# upload_to_s3
# --------------------------------------------------------------------------- #
class _FakeS3Upload:
    def __init__(self):
        self.calls: list[tuple] = []

    def upload_file(self, path, bucket, key):
        self.calls.append((path, bucket, key))


class TestUploadToS3:
    def test_missing_file_returns_false(self, tmp_path):
        s3 = _FakeS3Upload()
        ok = upload_to_s3(_FakeSession(s3), tmp_path / "nope.txt", "bucket")
        assert ok is False and s3.calls == []

    def test_empty_bucket_returns_false(self, tmp_path):
        f = tmp_path / "a.html"
        f.write_text("x")
        s3 = _FakeS3Upload()
        ok = upload_to_s3(_FakeSession(s3), f, "")
        assert ok is False

    def test_prefix_joined_and_stripped(self, tmp_path):
        f = tmp_path / "a.html"
        f.write_text("x")
        s3 = _FakeS3Upload()
        ok = upload_to_s3(_FakeSession(s3), f, "bucket", s3_prefix="/news/")
        assert ok is True
        assert s3.calls[0][2] == "news/a.html"

    def test_no_prefix_uses_bare_name(self, tmp_path):
        f = tmp_path / "a.html"
        f.write_text("x")
        s3 = _FakeS3Upload()
        upload_to_s3(_FakeSession(s3), f, "bucket")
        assert s3.calls[0][2] == "a.html"


# --------------------------------------------------------------------------- #
# check_and_download_from_s3
# --------------------------------------------------------------------------- #
class TestCheckAndDownload:
    def test_missing_args_returns_false(self, tmp_path):
        assert (
            check_and_download_from_s3(_FakeSession(object()), "", "k", tmp_path / "x")
            is False
        )

    def test_404_returns_false(self, tmp_path):
        class _S3:
            def head_object(self, **kwargs):
                raise _client_error("404")

        ok = check_and_download_from_s3(
            _FakeSession(_S3()), "bucket", "key", tmp_path / "out.txt"
        )
        assert ok is False

    def test_success_downloads(self, tmp_path):
        out = tmp_path / "sub" / "out.txt"

        class _S3:
            def head_object(self, **kwargs):
                return {}

            def download_file(self, bucket, key, dest):
                Path(dest).write_text("downloaded")

        ok = check_and_download_from_s3(_FakeSession(_S3()), "bucket", "key", out)
        assert ok is True
        assert out.read_text() == "downloaded"


# --------------------------------------------------------------------------- #
# get_ssm_param_value
# --------------------------------------------------------------------------- #
class TestGetSsmParam:
    def test_returns_value(self):
        class _SSM:
            def get_parameter(self, **kwargs):
                return {"Parameter": {"Value": "secret"}}

        assert get_ssm_param_value(_FakeSession(_SSM()), "/p/x") == "secret"

    def test_raises_on_client_error(self):
        class _SSM:
            def get_parameter(self, **kwargs):
                raise _client_error("ParameterNotFound")

        with pytest.raises(ClientError):
            get_ssm_param_value(_FakeSession(_SSM()), "/p/x")


# --------------------------------------------------------------------------- #
# submit_batch_job + wait_for_batch_job_completion
# --------------------------------------------------------------------------- #
class TestSubmitBatchJob:
    def test_returns_job_id(self):
        class _Batch:
            def submit_job(self, **kwargs):
                return {"jobId": "job-1"}

        assert submit_batch_job(_FakeSession(_Batch()), "n", "q", "d") == "job-1"

    def test_returns_none_on_error(self):
        class _Batch:
            def submit_job(self, **kwargs):
                raise _client_error("ClientException")

        assert submit_batch_job(_FakeSession(_Batch()), "n", "q", "d") is None


class TestWaitForBatchJob:
    def test_succeeded(self):
        class _Batch:
            def describe_jobs(self, jobs):
                return {"jobs": [{"status": "SUCCEEDED"}]}

        assert wait_for_batch_job_completion(_FakeSession(_Batch()), "job-1") is True

    def test_failed(self):
        class _Batch:
            def describe_jobs(self, jobs):
                return {"jobs": [{"status": "FAILED", "statusReason": "OOM"}]}

        assert wait_for_batch_job_completion(_FakeSession(_Batch()), "job-1") is False

    def test_missing_status_aborts(self):
        class _Batch:
            def describe_jobs(self, jobs):
                return {"jobs": [{}]}

        assert wait_for_batch_job_completion(_FakeSession(_Batch()), "job-1") is False

    def test_client_error_returns_false(self):
        class _Batch:
            def describe_jobs(self, jobs):
                raise _client_error("ServerException")

        assert wait_for_batch_job_completion(_FakeSession(_Batch()), "job-1") is False

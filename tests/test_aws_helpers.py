"""Tests for aws_helpers — pure logic around boto3 (SES MIME building, S3
key/prefix handling, 404-vs-error branches, batch polling state machine).
Uses fake boto clients; no real AWS."""

from __future__ import annotations

import email
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from app.src.aws_helpers import (
    check_and_download_from_s3,
    get_ssm_param_value,
    html_to_text,
    read_s3_json,
    send_email,
    submit_batch_job,
    upload_to_s3,
    wait_for_batch_job_completion,
    write_s3_json,
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

    def test_connection_error_returns_false(self):
        # Not a ClientError: once botocore's own retries are spent, a dropped
        # connection surfaces as BotoCoreError, which used to escape and abort
        # the send loop for every recipient after this one.
        from botocore.exceptions import EndpointConnectionError

        class _Down:
            def send_raw_email(self, **kwargs):
                raise EndpointConnectionError(endpoint_url="https://email")

        ok = send_email(_FakeSession(_Down()), "s", "a@b.com", ["c@d.com"], "<p>x</p>")
        assert ok is False

    def test_list_unsubscribe_header_present(self):
        """Bulk-sender reputation systems expect an opt-out header on newsletter
        mail; without one the reader's only lever is the spam button, which is
        what actually damages the sending identity."""
        ses = _FakeSES()
        send_email(
            _FakeSession(ses),
            "subj",
            "digest@example.com",
            ["to@example.com"],
            "<p>hi</p>",
        )
        raw = ses.sent[0]["RawMessage"]["Data"]
        assert (
            "List-Unsubscribe: <mailto:digest@example.com?subject=unsubscribe>" in raw
        )

    def test_plain_text_alternative_precedes_html(self):
        """An HTML-only message reads as bulk mail to spam filters and shows
        nothing in a text-only client. The last alternative is the preferred
        one, so the plain part goes first."""
        ses = _FakeSES()
        html = (
            "<html><head><title>Issue</title><style>p {color: red}</style></head>"
            '<body><div style="display: none; max-height: 0">preheader</div>'
            '<h1>Weekly Digest</h1><h2 class="card-title">First Title</h2>'
            "<p>Body <strong>bold</strong> text.</p>"
            '<a href="https://example.com/a">Read the original</a>'
            "<h2>Second Title</h2><p>More.</p></body></html>"
        )
        send_email(_FakeSession(ses), "s", "a@b.com", ["c@d.com"], html)
        message = email.message_from_string(ses.sent[0]["RawMessage"]["Data"])
        assert message.get_content_type() == "multipart/alternative"
        parts = [p for p in message.walk() if not p.is_multipart()]
        assert [p.get_content_type() for p in parts] == ["text/plain", "text/html"]
        text = parts[0].get_payload(decode=True).decode("utf-8")
        assert parts[1].get_payload(decode=True).decode("utf-8") == html
        assert "First Title" in text
        assert "Second Title" in text
        assert "Body bold text." in text
        assert "Read the original (https://example.com/a)" in text
        for gone in ("preheader", "color: red", "<", "Issue"):
            assert gone not in text


class TestHtmlToText:
    def test_blocks_become_lines_and_blank_runs_collapse(self):
        html = "<div><p>One</p>\n\n\n<p>Two<br>Three</p></div><ul><li>a</li><li>b</li></ul>"
        assert html_to_text(html) == "One\n\nTwo\nThree\n\na\n\nb\n"

    def test_link_whose_text_is_its_url_is_not_repeated(self):
        html = '<a href="https://example.com/x">https://example.com/x</a>'
        assert html_to_text(html) == "https://example.com/x\n"

    def test_no_unsubscribe_header_without_a_usable_address(self):
        ses = _FakeSES()
        send_email(_FakeSession(ses), "s", "no-at-sign", ["to@example.com"], "<p>x</p>")
        assert "List-Unsubscribe" not in ses.sent[0]["RawMessage"]["Data"]


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

    def test_a_job_not_yet_visible_is_polled_again_not_failed(self):
        # Right after submit describe_jobs can return no jobs. Reporting that as
        # a failure made run_batch exit 1 while the job went on to send.
        responses = iter(
            [{"jobs": []}, {"jobs": [{}]}, {"jobs": [{"status": "SUCCEEDED"}]}]
        )

        class _Batch:
            def describe_jobs(self, jobs):
                return next(responses)

        assert (
            wait_for_batch_job_completion(
                _FakeSession(_Batch()), "job-1", poll_interval_seconds=0
            )
            is True
        )

    def test_client_error_returns_false(self):
        class _Batch:
            def describe_jobs(self, jobs):
                raise _client_error("ServerException")

        assert wait_for_batch_job_completion(_FakeSession(_Batch()), "job-1") is False

    def test_waiter_outlasts_the_longest_possible_job(self):
        """The waiter must not give up before the job it waits on can, or a long
        but healthy run is reported as failed and exits non-zero."""
        from app.src.aws_helpers import DEFAULT_BATCH_TIMEOUT
        from app.src.constants import (
            BATCH_ATTEMPT_TIMEOUT_MINUTES,
            BATCH_ATTEMPTS,
            BATCH_RUNNABLE_LIMIT_MINUTES,
        )

        longest_job_seconds = 60 * (
            BATCH_RUNNABLE_LIMIT_MINUTES
            + BATCH_ATTEMPTS * BATCH_ATTEMPT_TIMEOUT_MINUTES
        )
        assert DEFAULT_BATCH_TIMEOUT > longest_job_seconds


# --------------------------------------------------------------------------- #
# read_s3_json / write_s3_json
#
# These back the delivery ledger, whose contract is FAIL OPEN: every way of not
# getting an answer must look the same to the caller (None), because sending a
# duplicate issue is an annoyance while withholding one is the failure the
# pipeline exists to avoid.
# --------------------------------------------------------------------------- #
class _FakeS3Objects:
    def __init__(self, body: bytes | None = None, error: ClientError | None = None):
        self._body = body
        self._error = error
        self.puts: list[dict] = []

    def get_object(self, **kwargs):
        if self._error:
            raise self._error
        return {"Body": _FakeBody(self._body or b"")}

    def put_object(self, **kwargs):
        if self._error:
            raise self._error
        self.puts.append(kwargs)
        return {}


class _FakeBody:
    def __init__(self, data: bytes):
        self._data = data

    def read(self):
        return self._data


class TestReadS3Json:
    def test_reads_a_json_object(self):
        client = _FakeS3Objects(body=b'{"delivered": ["a@example.com"]}')
        out = read_s3_json(_FakeSession(client), "bucket", "key.json")
        assert out == {"delivered": ["a@example.com"]}

    def test_missing_object_is_none(self):
        client = _FakeS3Objects(error=_client_error("NoSuchKey"))
        assert read_s3_json(_FakeSession(client), "bucket", "key.json") is None

    def test_other_client_error_is_also_none(self):
        """Absent and unreadable are the same answer to every caller so far; the
        log line is what distinguishes them."""
        client = _FakeS3Objects(error=_client_error("AccessDenied"))
        assert read_s3_json(_FakeSession(client), "bucket", "key.json") is None

    def test_malformed_json_is_none(self):
        client = _FakeS3Objects(body=b"{not json")
        assert read_s3_json(_FakeSession(client), "bucket", "key.json") is None

    def test_json_that_is_not_an_object_is_none(self):
        client = _FakeS3Objects(body=b'["a@example.com"]')
        assert read_s3_json(_FakeSession(client), "bucket", "key.json") is None

    def test_undecodable_bytes_are_none(self):
        client = _FakeS3Objects(body=b"\xff\xfe not utf-8")
        assert read_s3_json(_FakeSession(client), "bucket", "key.json") is None

    @pytest.mark.parametrize("bucket,key", [("", "k"), ("b", ""), ("", "")])
    def test_missing_bucket_or_key_is_none(self, bucket, key):
        client = _FakeS3Objects(body=b"{}")
        assert read_s3_json(_FakeSession(client), bucket, key) is None


class TestWriteS3Json:
    def test_writes_utf8_json(self):
        client = _FakeS3Objects()
        assert write_s3_json(
            _FakeSession(client),
            "bucket",
            "key.json",
            {"delivered": ["가@example.com"]},
        )
        put = client.puts[0]
        assert put["Bucket"] == "bucket"
        assert put["Key"] == "key.json"
        assert put["ContentType"] == "application/json"
        # ensure_ascii=False, so non-ASCII survives as itself.
        assert "가@example.com" in put["Body"].decode("utf-8")

    def test_client_error_returns_false(self):
        client = _FakeS3Objects(error=_client_error("AccessDenied"))
        assert not write_s3_json(_FakeSession(client), "bucket", "key.json", {"a": 1})

    @pytest.mark.parametrize("bucket,key", [("", "k"), ("b", "")])
    def test_missing_bucket_or_key_returns_false(self, bucket, key):
        client = _FakeS3Objects()
        assert not write_s3_json(_FakeSession(client), bucket, key, {"a": 1})


class TestSuppressedDestinations:
    class _Client:
        def __init__(self, pages=None, error=None):
            self.pages, self.error, self.calls = list(pages or []), error, []

        def list_suppressed_destinations(self, **kwargs):
            self.calls.append(kwargs)
            if self.error:
                raise self.error
            return self.pages.pop(0)

    class _Session:
        def __init__(self, client):
            self._client = client

        def client(self, name):
            assert name == "sesv2"
            return self._client

    def test_follows_pages_and_lowercases(self):
        from app.src.aws_helpers import suppressed_destinations

        client = self._Client(
            [
                {
                    "SuppressedDestinationSummaries": [{"EmailAddress": "A@x.com"}],
                    "NextToken": "t1",
                },
                {"SuppressedDestinationSummaries": [{"EmailAddress": "b@x.com"}]},
            ]
        )
        assert suppressed_destinations(self._Session(client)) == {"a@x.com", "b@x.com"}
        assert client.calls == [{}, {"NextToken": "t1"}]

    def test_a_read_failure_suppresses_nobody(self):
        from botocore.exceptions import ClientError

        from app.src.aws_helpers import suppressed_destinations

        error = ClientError({"Error": {"Code": "AccessDenied"}}, "List")
        assert (
            suppressed_destinations(self._Session(self._Client(error=error))) == set()
        )

"""Unit tests for utils: email validation, date ranges, the HTML tag output
parser, and BatchProcessor fallback semantics. No network access."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError, ReadTimeoutError
from langchain_core.exceptions import OutputParserException

from app.domain.alarms import format_alarm, validate_email, validate_emails
from app.domain.post import get_date_range
from app.src.utils import (
    BatchProcessor,
    HTMLTagOutputParser,
    is_retryable,
    retry_with_backoff,
)


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, "Converse")


class TestEmailValidation:
    @pytest.mark.parametrize(
        "email,valid",
        [
            ("user@example.com", True),
            ("first.last+tag@sub.domain.co", True),
            ("  spaced@example.com  ", True),
            ("no-at-sign", False),
            ("missing@tld", False),
            ("@example.com", False),
            ("", False),
        ],
    )
    def test_validate_email(self, email, valid):
        assert validate_email(email) is valid

    def test_validate_emails_filters_invalid(self):
        out = validate_emails(["a@example.com", "bad", "b@example.com"])
        assert out == ["a@example.com", "b@example.com"]


class TestFormatAlarm:
    _TS = datetime(2026, 6, 10, 4, 12, 0, tzinfo=UTC)

    def test_subject_format(self):
        subject, _ = format_alarm(
            event="Newsletter Delivery",
            status="FAILED",
            fields={"Error": "boom"},
            project="tech-digest",
        )
        assert subject == "[tech-digest] Newsletter Delivery — FAILED"

    def test_project_is_required(self):
        """It used to default to the literal "tech-digest" and no caller passed
        it, so every alarm named that project whatever the deployment was."""
        with pytest.raises(TypeError):
            format_alarm(event="X", status="ALERT", fields={})

    def test_custom_project_in_subject(self):
        subject, _ = format_alarm(
            event="Crawl Health", status="ALERT", fields={}, project="paper-bridge"
        )
        assert subject == "[paper-bridge] Crawl Health — ALERT"

    def test_inline_fields_aligned(self):
        _, msg = format_alarm(
            event="Newsletter Delivery",
            status="ALERT",
            fields={"Delivered": "2/3", "Failed recipients": "a@b.com"},
            project="tech-digest",
            timestamp=self._TS,
        )
        assert "Delivered:         2/3" in msg
        assert "Failed recipients: a@b.com" in msg
        assert msg.startswith("Newsletter Delivery ALERT\n\n")
        assert msg.endswith("— 2026-06-10 04:12:00 UTC")

    def test_multiline_field_rendered_as_block(self):
        _, msg = format_alarm(
            event="Crawl Health",
            status="ALERT",
            fields={"Failing sources": "2", "Detail": "line1\nline2"},
            project="tech-digest",
            timestamp=self._TS,
        )
        assert "Failing sources: 2" in msg
        assert "Detail:\nline1\nline2" in msg


class TestGetDateRange:
    def test_explicit_end_date(self):
        start, end = get_date_range("2026-06-01", days_back=7)
        assert end == datetime(2026, 6, 1, tzinfo=UTC)
        assert start == datetime(2026, 5, 25, tzinfo=UTC)

    def test_none_end_date_is_midnight_today_utc(self):
        def midnight() -> datetime:
            return datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)

        # Read the clock on both sides, so a run straddling midnight still passes.
        before = midnight()
        start, end = get_date_range(None, days_back=3)
        assert end in {before, midnight()}
        assert (end - start).days == 3

    def test_consecutive_weekly_runs_abut(self):
        # A post published on a run day after the job ran must land in the next
        # issue. The old window ended at 23:59:59 on the run day and the next one
        # started there, so such a post fell between the two.
        from app.domain.post import is_date_in_range

        this_week = get_date_range("2026-10-10", days_back=7)
        next_week = get_date_range("2026-10-17", days_back=7)
        assert this_week[1] == next_week[0]
        published_after_the_run = datetime(2026, 10, 10, 15, 0, tzinfo=UTC)
        assert not is_date_in_range(published_after_the_run, *this_week)
        assert is_date_in_range(published_after_the_run, *next_week)

    def test_a_boundary_post_belongs_to_one_issue(self):
        from app.domain.post import is_date_in_range

        boundary = datetime(2026, 10, 10, tzinfo=UTC)
        weeks = [get_date_range(d, days_back=7) for d in ("2026-10-10", "2026-10-17")]
        assert sum(is_date_in_range(boundary, *w) for w in weeks) == 1


class TestHTMLTagOutputParser:
    def test_single_tag_returns_string(self):
        parser = HTMLTagOutputParser(tag_names="summary")
        assert parser.parse("<summary>hi there</summary>") == "hi there"

    def test_multiple_tags_returns_dict(self):
        parser = HTMLTagOutputParser(tag_names=["title", "score", "reason"])
        result = parser.parse(
            "<title>My Title</title><score>0.84</score><reason>good</reason>"
        )
        assert result == {"title": "My Title", "score": "0.84", "reason": "good"}

    def test_missing_tag_omitted(self):
        parser = HTMLTagOutputParser(tag_names=["title", "score"])
        result = parser.parse("<title>Only Title</title>")
        assert result == {"title": "Only Title"}

    def test_empty_text_list(self):
        parser = HTMLTagOutputParser(tag_names=["a", "b"])
        assert parser.parse("") == {}

    def test_empty_text_single(self):
        parser = HTMLTagOutputParser(tag_names="a")
        assert parser.parse("") == ""

    def test_preserves_inner_html(self):
        parser = HTMLTagOutputParser(tag_names="summary")
        out = parser.parse("<summary><h3>Title</h3><p>body</p></summary>")
        assert "<h3>Title</h3>" in out

    def test_required_tag_missing_raises(self):
        # This is what makes a wrapping OutputFixingParser actually trigger.
        from langchain_core.exceptions import OutputParserException

        parser = HTMLTagOutputParser(
            tag_names=["summary", "tags"], required_tags=["summary"]
        )
        with pytest.raises(OutputParserException):
            parser.parse("<tags>RAG</tags>")  # no <summary>

    def test_required_tag_empty_raises(self):
        from langchain_core.exceptions import OutputParserException

        parser = HTMLTagOutputParser(tag_names=["summary"], required_tags=["summary"])
        with pytest.raises(OutputParserException):
            parser.parse("<summary></summary>")

    def test_required_tag_present_ok(self):
        parser = HTMLTagOutputParser(
            tag_names=["summary", "tags"], required_tags=["summary"]
        )
        result = parser.parse("<summary>real content</summary>")
        assert result["summary"] == "real content"

    @pytest.mark.parametrize("text", ["", "   \n"])
    def test_empty_text_with_required_is_not_handed_to_the_repair_model(self, text):
        # An OutputParserException here sent the EMPTY completion to the repair
        # model, which wrote a summary from nothing that then shipped.
        from langchain_classic.output_parsers import OutputFixingParser
        from langchain_core.language_models.fake_chat_models import (
            FakeListChatModel,
        )

        from app.src.utils import EmptyModelOutput

        parser = OutputFixingParser.from_llm(
            llm=FakeListChatModel(responses=["<summary>invented</summary>"]),
            parser=HTMLTagOutputParser(
                tag_names=["summary"], required_tags=["summary"]
            ),
        )
        with pytest.raises(EmptyModelOutput):
            parser.parse(text)

    def test_output_fixing_parser_actually_repairs(self):
        # The repair path end to end, through the installed langchain_classic:
        # without format instructions its prompt raised KeyError before the
        # fixing model was ever called.
        from langchain_classic.output_parsers import OutputFixingParser
        from langchain_core.language_models.fake_chat_models import (
            FakeListChatModel,
        )

        fixer = FakeListChatModel(responses=["<summary>repaired</summary>"])
        parser = OutputFixingParser.from_llm(
            llm=fixer,
            parser=HTMLTagOutputParser(
                tag_names=["summary", "tags"], required_tags=["summary"]
            ),
        )
        assert parser.parse("an answer with no tags")["summary"] == "repaired"

    def test_format_instructions_name_every_tag(self):
        parser = HTMLTagOutputParser(
            tag_names=["summary", "tags"], required_tags=["summary"]
        )
        text = parser.get_format_instructions()
        assert "<summary>...</summary>" in text
        assert "<tags>...</tags>" in text


class TestIsRetryable:
    @pytest.mark.parametrize(
        "code",
        ["ThrottlingException", "ServiceUnavailableException", "ModelTimeoutException"],
    )
    def test_transient_bedrock_errors_retry(self, code):
        assert is_retryable(_client_error(code))

    @pytest.mark.parametrize(
        "code",
        ["ValidationException", "AccessDeniedException", "ResourceNotFoundException"],
    )
    def test_permanent_bedrock_errors_do_not(self, code):
        assert not is_retryable(_client_error(code))

    def test_connection_and_timeout_errors_retry(self):
        assert is_retryable(EndpointConnectionError(endpoint_url="https://x"))
        assert is_retryable(ReadTimeoutError(endpoint_url="https://x"))

    def test_malformed_output_is_not_regenerated(self):
        # The repair model already had its one try; regenerating the whole
        # summary for a format slip re-paid the full call up to six times.
        assert not is_retryable(OutputParserException("no <summary>"))

    def test_timeout_retries(self):
        assert is_retryable(TimeoutError("read timed out"))

    def test_an_error_raised_while_handling_another_is_judged_on_its_own(self):
        # __context__ (implicit chaining) used to be walked, so a bug inside the
        # repair path inherited the retryable verdict of the error it handled.
        try:
            try:
                raise _client_error("ThrottlingException")
            except ClientError:
                raise KeyError("bug in the handler")  # noqa: B904
        except KeyError as error:
            assert not is_retryable(error)

    @pytest.mark.parametrize("error", [KeyError("x"), TypeError("x"), ValueError("x")])
    def test_bugs_do_not(self, error):
        assert not is_retryable(error)

    def test_cause_chain_is_followed(self):
        try:
            try:
                raise _client_error("ThrottlingException")
            except ClientError as inner:
                raise RuntimeError("wrapped") from inner
        except RuntimeError as outer:
            assert is_retryable(outer)

    def test_decorator_stops_at_a_permanent_error(self):
        calls = {"n": 0}

        @retry_with_backoff("probe", multiplier=1.0, max_wait=0, attempts=5)
        def call():
            calls["n"] += 1
            raise _client_error("ValidationException")

        with pytest.raises(ClientError):
            call()
        assert calls["n"] == 1


class TestBatchProcessor:
    def test_empty_input_returns_empty(self):
        bp = BatchProcessor()
        result = bp.execute_with_fallback(
            items_to_process=[],
            prepare_inputs_func=lambda items: [],
            batch_func=lambda inputs, config=None, return_exceptions=False: [],
            sequential_func=lambda x: x,
            task_name="noop",
        )
        assert result == []

    def test_batch_success_path(self):
        bp = BatchProcessor(max_concurrency=2, batch_size=2)
        items = [1, 2, 3]
        result = bp.execute_with_fallback(
            items_to_process=items,
            prepare_inputs_func=lambda chunk: [{"v": v} for v in chunk],
            batch_func=lambda inputs, config=None, return_exceptions=False: [
                d["v"] * 10 for d in inputs
            ],
            sequential_func=lambda d: d["v"] * 10,
            task_name="mul",
            show_progress=False,
        )
        assert result == [10, 20, 30]

    def test_falls_back_to_sequential_on_batch_failure(self):
        bp = BatchProcessor(max_concurrency=2, batch_size=2, max_retries=1)
        calls = {"batch": 0}

        def failing_batch(inputs, config=None, return_exceptions=False):
            calls["batch"] += 1
            raise RuntimeError("batch broke")

        result = bp.execute_with_fallback(
            items_to_process=[1, 2],
            prepare_inputs_func=lambda chunk: [{"v": v} for v in chunk],
            batch_func=failing_batch,
            sequential_func=lambda d: d["v"] + 100,
            task_name="add",
            show_progress=False,
        )
        assert calls["batch"] >= 1
        assert result == [101, 102]

    def test_sequential_fallback_preserves_alignment_with_none(self):
        # A failing item must yield None at its position (not be dropped), so
        # callers that zip results back onto inputs stay aligned.
        bp = BatchProcessor(max_concurrency=3, batch_size=3, max_retries=1)

        def failing_batch(inputs, config=None, return_exceptions=False):
            raise RuntimeError("force sequential")

        def seq(d):
            if d["v"] == 2:
                raise ValueError("item 2 fails")
            return d["v"] * 10

        result = bp.execute_with_fallback(
            items_to_process=[1, 2, 3],
            prepare_inputs_func=lambda chunk: [{"v": v} for v in chunk],
            batch_func=failing_batch,
            sequential_func=seq,
            task_name="align",
            show_progress=False,
        )
        # Same length as input; the failed middle item is None, not dropped.
        assert result == [10, None, 30]


class TestPartialBatchFailure:
    """One failed item in a chunk must not re-bill the items that succeeded."""

    @staticmethod
    def _batch_with_one_failure(error):
        def batch(inputs, config=None, return_exceptions=False):
            assert return_exceptions, "the batch must keep per-item results"
            return [error if d["v"] == 2 else d["v"] * 10 for d in inputs]

        return batch

    def test_only_the_failed_item_is_called_again(self):
        bp = BatchProcessor(batch_size=5, max_retries=1)
        sequential_calls: list[int] = []

        def seq(d):
            sequential_calls.append(d["v"])
            return d["v"] * 10

        result = bp.execute_with_fallback(
            items_to_process=[1, 2, 3, 4, 5],
            prepare_inputs_func=lambda chunk: [{"v": v} for v in chunk],
            batch_func=self._batch_with_one_failure(
                _client_error("ThrottlingException")
            ),
            sequential_func=seq,
            task_name="partial",
            show_progress=False,
        )
        assert result == [10, 20, 30, 40, 50]
        assert sequential_calls == [2]

    def test_a_permanent_batch_error_is_not_retried(self):
        bp = BatchProcessor(batch_size=5, max_retries=3)
        sequential_calls: list[int] = []

        def seq(d):
            sequential_calls.append(d["v"])
            return d["v"] * 10

        result = bp.execute_with_fallback(
            items_to_process=[1, 2, 3],
            prepare_inputs_func=lambda chunk: [{"v": v} for v in chunk],
            batch_func=self._batch_with_one_failure(
                _client_error("ValidationException")
            ),
            sequential_func=seq,
            task_name="permanent",
            show_progress=False,
        )
        assert result == [10, None, 30]
        assert sequential_calls == []

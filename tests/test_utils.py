"""Unit tests for utils: email validation, date ranges, the HTML tag output
parser, and BatchProcessor fallback semantics. No network access."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.src.utils import (
    BatchProcessor,
    HTMLTagOutputParser,
    format_alarm,
    get_date_range,
    validate_email,
    validate_emails,
)


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
            event="Newsletter Delivery", status="FAILED", fields={"Error": "boom"}
        )
        assert subject == "[tech-digest] Newsletter Delivery — FAILED"

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
            timestamp=self._TS,
        )
        assert "Failing sources: 2" in msg
        assert "Detail:\nline1\nline2" in msg


class TestGetDateRange:
    def test_explicit_end_date(self):
        start, end = get_date_range("2026-06-01", days_back=7)
        assert end.year == 2026 and end.month == 6 and end.day == 1
        assert end.hour == 23 and end.minute == 59
        assert (end - start).days == 7
        assert end.tzinfo == UTC

    def test_none_end_date_uses_now(self):
        start, end = get_date_range(None, days_back=3)
        assert (end - start).days == 3


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

    def test_empty_text_with_required_raises(self):
        from langchain_core.exceptions import OutputParserException

        parser = HTMLTagOutputParser(tag_names=["summary"], required_tags=["summary"])
        with pytest.raises(OutputParserException):
            parser.parse("")


class TestBatchProcessor:
    def test_empty_input_returns_empty(self):
        bp = BatchProcessor()
        result = bp.execute_with_fallback(
            items_to_process=[],
            prepare_inputs_func=lambda items: [],
            batch_func=lambda inputs, config=None: [],
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
            batch_func=lambda inputs, config=None: [d["v"] * 10 for d in inputs],
            sequential_func=lambda d: d["v"] * 10,
            task_name="mul",
            show_progress=False,
        )
        assert result == [10, 20, 30]

    def test_falls_back_to_sequential_on_batch_failure(self):
        bp = BatchProcessor(max_concurrency=2, batch_size=2, max_retries=1)
        calls = {"batch": 0}

        def failing_batch(inputs, config=None):
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

        def failing_batch(inputs, config=None):
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

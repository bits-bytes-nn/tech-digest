"""Tests for run_batch CLI pure logic: the "null"-sentinel parameter
sanitization (round-trips with main.py's decode) and date validation."""

from __future__ import annotations

import sys
from pathlib import Path

# run_batch uses the container import layout (`from src/configs import ...`),
# so it must be imported with app/ on the path.
APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import run_batch  # noqa: E402


class TestSanitizeParameters:
    def test_none_becomes_null_sentinel(self):
        out = run_batch.sanitize_parameters({"end_date": None})
        assert out["end_date"] == "null"

    def test_recipients_spaces_to_commas(self):
        out = run_batch.sanitize_parameters({"recipients": "a@x.com b@y.com"})
        assert out["recipients"] == "a@x.com,b@y.com"

    def test_recipients_null_left_untouched(self):
        out = run_batch.sanitize_parameters({"recipients": "null"})
        assert out["recipients"] == "null"

    def test_plain_value_stringified(self):
        out = run_batch.sanitize_parameters({"language": "ko"})
        assert out["language"] == "ko"

    def test_non_recipient_with_spaces_not_comma_joined(self):
        # Only 'recipients' gets the space->comma treatment.
        out = run_batch.sanitize_parameters({"language": "ko en"})
        assert out["language"] == "ko en"


class TestValidateDate:
    def test_valid_date(self):
        assert run_batch.validate_date("2026-06-01") is True

    def test_null_sentinel_is_valid(self):
        assert run_batch.validate_date("null") is True

    def test_empty_is_valid(self):
        assert run_batch.validate_date("") is True

    def test_bad_format_invalid(self):
        assert run_batch.validate_date("06/01/2026") is False

    def test_garbage_invalid(self):
        assert run_batch.validate_date("not-a-date") is False

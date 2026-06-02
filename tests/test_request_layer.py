"""Tests for the resilient HTTP request layer: header-set rotation, per-domain
caching of the working header set, and transient-error retry."""

from __future__ import annotations

import pytest
import requests

from app.src import feed_parser
from app.src.feed_parser import HeaderCache, _make_robust_request


@pytest.fixture(autouse=True)
def clear_header_cache():
    HeaderCache.clear()
    yield
    HeaderCache.clear()


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    # The retry layer now backs off with time.sleep; stub it so tests are fast.
    monkeypatch.setattr(feed_parser.time, "sleep", lambda *_: None)


class _FakeResponse:
    def __init__(self, status_code=200, headers=None):
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            err = requests.exceptions.HTTPError(f"{self.status_code}")
            err.response = self  # type: ignore[attr-defined]
            raise err


class _FakeSession:
    """Records calls and returns queued responses/exceptions in order."""

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.headers = {}
        self.calls = 0

    def get(self, url, **kwargs):
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _patch_session(monkeypatch, session):
    monkeypatch.setattr(feed_parser.requests, "Session", lambda: session)


class TestMakeRobustRequest:
    def test_success_first_try(self, monkeypatch):
        session = _FakeSession([_FakeResponse(200)])
        _patch_session(monkeypatch, session)
        resp = _make_robust_request("https://example.com")
        assert resp is not None
        assert session.calls == 1

    def test_retries_transient_then_succeeds(self, monkeypatch):
        # First attempt 503 (transient) -> retry same header set -> 200.
        session = _FakeSession([_FakeResponse(503), _FakeResponse(200)])
        _patch_session(monkeypatch, session)
        resp = _make_robust_request("https://example.com")
        assert resp is not None
        assert session.calls == 2

    def test_429_retries_up_to_three_attempts(self, monkeypatch):
        # 429 is transient; one header set now gets up to 3 attempts.
        session = _FakeSession(
            [_FakeResponse(429), _FakeResponse(429), _FakeResponse(200)]
        )
        _patch_session(monkeypatch, session)
        resp = _make_robust_request("https://example.com")
        assert resp is not None
        assert session.calls == 3

    def test_honors_retry_after_header(self, monkeypatch):
        slept: list[float] = []
        monkeypatch.setattr(feed_parser.time, "sleep", lambda s: slept.append(s))
        session = _FakeSession(
            [_FakeResponse(429, headers={"Retry-After": "5"}), _FakeResponse(200)]
        )
        _patch_session(monkeypatch, session)
        _make_robust_request("https://example.com")
        assert slept and slept[0] == 5.0

    def test_rotates_header_sets_on_403(self, monkeypatch):
        # 403 is non-transient -> no retry, move to next header set. Three sets,
        # first two 403, third 200 => 3 calls total.
        session = _FakeSession(
            [_FakeResponse(403), _FakeResponse(403), _FakeResponse(200)]
        )
        _patch_session(monkeypatch, session)
        resp = _make_robust_request("https://example.com")
        assert resp is not None
        assert session.calls == 3

    def test_all_fail_returns_none(self, monkeypatch):
        session = _FakeSession([_FakeResponse(403)] * 3)
        _patch_session(monkeypatch, session)
        assert _make_robust_request("https://example.com") is None

    def test_caches_working_header_index(self, monkeypatch):
        session = _FakeSession([_FakeResponse(403), _FakeResponse(200)])
        _patch_session(monkeypatch, session)
        _make_robust_request("https://example.com")
        # Header set index 1 (the 2nd) worked -> cached for the domain.
        assert HeaderCache.get_cached_header_index("example.com") == 1

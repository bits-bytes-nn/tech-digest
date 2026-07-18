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
    def __init__(
        self,
        status_code=200,
        headers=None,
        url="https://example.com",
        is_redirect=False,
    ):
        self.status_code = status_code
        self.headers = headers or {}
        # The robust request re-checks the final landing host for SSRF; tests
        # default to an external URL so the guard is a no-op unless overridden.
        self.url = url
        # Redirects are now followed manually and every hop is SSRF-validated,
        # so the fake must advertise whether it is a redirect (and carry a
        # Location header) to drive the redirect loop.
        self.is_redirect = is_redirect

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


class TestSSRFGuard:
    @pytest.mark.parametrize(
        "url",
        [
            "http://169.254.169.254/latest/meta-data/",  # cloud metadata
            "http://127.0.0.1/admin",  # loopback
            "http://10.0.0.5/internal",  # RFC1918
            "http://192.168.1.1/",  # RFC1918
            "http://[::1]/",  # IPv6 loopback
            "http://0.0.0.0/",  # unspecified
            "http://[::ffff:169.254.169.254]/",  # IPv4-mapped IPv6 metadata
            # Numeric IPv4 encodings the OS resolver accepts but a plain
            # ip_address() parse would miss (all == 127.0.0.1 / metadata IP):
            "http://2130706433/admin",  # decimal 127.0.0.1
            "http://0x7f000001/admin",  # hex 127.0.0.1
            "http://017700000001/admin",  # octal 127.0.0.1
            "http://2852039166/latest/meta-data/",  # decimal 169.254.169.254
        ],
    )
    def test_blocks_internal_targets_without_requesting(self, monkeypatch, url):
        # A blocked host must short-circuit BEFORE any network call is made.
        session = _FakeSession([_FakeResponse(200)])
        _patch_session(monkeypatch, session)
        assert _make_robust_request(url) is None
        assert session.calls == 0

    def test_blocks_redirect_to_internal_host(self, monkeypatch):
        # External target that 302-redirects to the metadata IP: each hop is
        # validated BEFORE connecting, so the redirect target is rejected before
        # any request is made to it. The block is deterministic across header
        # sets, so every set hits the same redirect -> None.
        meta = "http://169.254.169.254/latest/meta-data/"
        redirect = _FakeResponse(302, headers={"location": meta}, is_redirect=True)
        session = _FakeSession([redirect] * 3)
        _patch_session(monkeypatch, session)
        assert _make_robust_request("https://example.com") is None

    def test_follows_external_redirect(self, monkeypatch):
        # A redirect to another EXTERNAL host is followed to completion.
        redirect = _FakeResponse(
            302,
            headers={"location": "https://example.org/final"},
            is_redirect=True,
        )
        final = _FakeResponse(200, url="https://example.org/final")
        session = _FakeSession([redirect, final])
        _patch_session(monkeypatch, session)
        resp = _make_robust_request("https://example.com")
        assert resp is not None
        assert session.calls == 2

    def test_allows_normal_external_host(self, monkeypatch):
        session = _FakeSession([_FakeResponse(200, url="https://example.com/post")])
        _patch_session(monkeypatch, session)
        assert _make_robust_request("https://example.com") is not None

    def test_blocks_hostname_resolving_to_internal_ip(self, monkeypatch):
        # A hostname (not an IP literal) that resolves to the metadata IP must
        # be blocked before any network call — the DNS-resolution layer of the
        # SSRF guard. getaddrinfo is stubbed to return the metadata address.
        def fake_getaddrinfo(host, *a, **k):
            return [(2, 1, 6, "", ("169.254.169.254", 0))]

        monkeypatch.setattr(feed_parser.socket, "getaddrinfo", fake_getaddrinfo)
        session = _FakeSession([_FakeResponse(200)])
        _patch_session(monkeypatch, session)
        assert _make_robust_request("http://evil.example.com/") is None
        assert session.calls == 0

    def test_allows_hostname_resolving_to_public_ip(self, monkeypatch):
        def fake_getaddrinfo(host, *a, **k):
            return [(2, 1, 6, "", ("93.184.216.34", 0))]  # example.com public IP

        monkeypatch.setattr(feed_parser.socket, "getaddrinfo", fake_getaddrinfo)
        session = _FakeSession([_FakeResponse(200, url="https://example.com/post")])
        _patch_session(monkeypatch, session)
        assert _make_robust_request("https://example.com") is not None

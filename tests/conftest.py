"""Shared pytest fixtures and path setup for the tech-digest test suite.

Everything imports from the one ``app`` package, so the repo root goes on
``sys.path``.
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

# Before any app import: the logger is configured at import time, and outside
# AWS it would otherwise append every test's log lines to logs/logs_<date>.txt,
# the same file a real local run writes.
os.environ.setdefault("LOG_TO_FILE", "false")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def date_range() -> tuple[datetime, datetime]:
    """The window ``get_date_range("2026-06-01", 7)`` builds: [05-25, 06-01)."""
    end = datetime(2026, 6, 1, tzinfo=UTC)
    start = datetime(2026, 5, 25, tzinfo=UTC)
    return start, end


@pytest.fixture
def templates_dir() -> Path:
    return REPO_ROOT / "app" / "templates"


@pytest.fixture
def sample_article_data() -> dict:
    """A minimal valid article payload for renderer tests."""
    return {
        "title": "Scaling Mixture-of-Experts Inference",
        "link": "https://example.com/moe",
        "published_date": "2026-05-30",
        "thumbnail": "openai.png",
        "summary": "<h3>Why This Matters</h3><p>Substantial content here.</p>",
        "one_liner": "Routing only two of eight experts per token cuts serving cost 3x.",
        "reading_minutes": 4,
        "tags": ["Mixture Of Experts", "Inference"],
        "urls": ['<a href="https://example.com/paper">Paper</a>'],
        "score": 0.84,
    }


@pytest.fixture
def propagating_logger():
    """Let ``caplog`` see the app logger.

    ``app.src.logger`` sets ``propagate = False`` so container logs aren't emitted
    twice, which also stops pytest's root-handler capture from seeing them.
    """
    from app.src.logger import logger

    original = logger.propagate
    logger.propagate = True
    try:
        yield logger
    finally:
        logger.propagate = original


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail any test that opens a real connection or resolves a real hostname.

    Several tests reached openai.com and example.com through the scrape fallback
    in ``Post.from_entry``, and the request-layer tests resolved real names, so
    the suite's result depended on third-party sites and took 30s offline.
    Literal IP addresses still resolve: they need no network.
    """
    import ipaddress
    import socket

    real_getaddrinfo = socket.getaddrinfo

    def guarded_getaddrinfo(host, *args, **kwargs):
        try:
            ipaddress.ip_address(str(host).strip("[]"))
        except ValueError as e:
            raise socket.gaierror(f"network disabled in tests: {host}") from e
        return real_getaddrinfo(host, *args, **kwargs)

    def refuse_connect(self, address):
        raise OSError(f"network disabled in tests: {address}")

    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", refuse_connect)
    # The article-page fetch behind Post.from_entry's thin-feed fallback: without
    # this, a blocked request is "transient" and the ladder sleeps through its
    # backoff. Tests that exercise the request layer import the function
    # directly, so this module-level stub does not reach them.
    if module := sys.modules.get("app.src.feed_parser"):
        monkeypatch.setattr(module, "_make_robust_request", lambda url: None)
    # The summarizer fetches each summary image to check mail clients can show
    # it. Offline, every image would be dropped; tests about that check stub
    # it themselves.
    if module := sys.modules.get("app.src.summarizer"):
        monkeypatch.setattr(module, "is_email_renderable_image", lambda url: True)

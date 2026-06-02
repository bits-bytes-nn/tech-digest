"""Shared pytest fixtures and path setup for the tech-digest test suite.

The application uses a mixed import layout: ``app.configs`` / ``app.src`` from
the repo root (used by ``config.py``, ``deploy_infra.py`` and the tests) and
bare ``configs`` / ``src`` inside the container (``WORKDIR=/app``). Tests
standardise on the ``app.*`` form, so we put the repo root on ``sys.path``.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def utc_now() -> datetime:
    """A fixed, timezone-aware reference instant for deterministic tests."""
    return datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def date_range() -> tuple[datetime, datetime]:
    """A one-week window ending 2026-06-01 (inclusive of full end day)."""
    end = datetime(2026, 6, 1, 23, 59, 59, tzinfo=UTC)
    start = datetime(2026, 5, 25, 23, 59, 59, tzinfo=UTC)
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
        "tags": ["Mixture Of Experts", "Inference"],
        "urls": ['<a href="https://example.com/paper">Paper</a>'],
        "score": 0.84,
    }

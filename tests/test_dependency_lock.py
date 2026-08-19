"""The container's dependency lock must stay consistent with the ranges.

The base images are pinned by digest, but the Python dependencies were declared
only as `>=` ranges and re-resolved on every image build — so a `cdk deploy`
could ship a langchain or pydantic release CI never tested, and two deploys of
the same commit could differ. `app/requirements.lock` fixes that; these tests
guard against it drifting back out of sync, which would be silent otherwise.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parent.parent / "app"
REQUIREMENTS = APP / "requirements.txt"
LOCK = APP / "requirements.lock"


def _normalize(name: str) -> str:
    """PEP 503 name normalization, so Pillow/pillow and _/- compare equal."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _entries(path: Path) -> dict[str, str]:
    found = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z0-9._-]+)(?:\[[^\]]*\])?\s*(.*)$", line)
        assert match, f"unparsable line in {path.name}: {raw!r}"
        found[_normalize(match.group(1))] = match.group(2)
    return found


class TestLockExists:
    def test_lock_is_present_and_non_trivial(self):
        assert LOCK.is_file(), "app/requirements.lock is what the image installs"
        assert len(_entries(LOCK)) > 30, "a lock without transitive deps pins nothing"


class TestLockIsFullyPinned:
    def test_every_entry_uses_an_exact_version(self):
        loose = {
            name: spec
            for name, spec in _entries(LOCK).items()
            if not spec.startswith("==")
        }
        assert not loose, f"lock entries must be '==' pinned: {loose}"


class TestLockCoversTheDeclaredRanges:
    def test_every_declared_requirement_is_locked(self):
        """A dependency added to requirements.txt without regenerating the lock
        would simply not be installed in the image."""
        missing = sorted(set(_entries(REQUIREMENTS)) - set(_entries(LOCK)))
        assert not missing, (
            f"declared but not locked: {missing} — regenerate app/requirements.lock "
            f"(the command is in its header)"
        )

    def test_locked_versions_satisfy_the_declared_minimums(self):
        """Catches a lock regenerated against an older resolution than the
        ranges now demand."""
        from packaging.version import Version

        locked = _entries(LOCK)
        for name, spec in _entries(REQUIREMENTS).items():
            match = re.match(r"^>=\s*([0-9][^,\s]*)$", spec)
            if not match:
                continue
            minimum = Version(match.group(1))
            actual = Version(locked[name].removeprefix("=="))
            assert actual >= minimum, f"{name}: locked {actual} < required {minimum}"


class TestDockerfilesInstallFromTheLock:
    @pytest.mark.parametrize("dockerfile", ["Dockerfile-batch", "Dockerfile-lambda"])
    def test_pip_installs_the_lock_not_the_ranges(self, dockerfile):
        body = (APP / dockerfile).read_text(encoding="utf-8")
        install_lines = [line for line in body.splitlines() if "pip install" in line]
        assert install_lines, f"{dockerfile} installs nothing?"
        for line in install_lines:
            assert "requirements.lock" in line, (
                f"{dockerfile} resolves ranges at build time: {line.strip()}"
            )
            assert "-r requirements.txt" not in line, line.strip()

    @pytest.mark.parametrize("dockerfile", ["Dockerfile-batch", "Dockerfile-lambda"])
    def test_lock_is_copied_into_the_build(self, dockerfile):
        body = (APP / dockerfile).read_text(encoding="utf-8")
        assert "requirements.lock" in body.split("RUN")[0] or any(
            "requirements.lock" in line
            for line in body.splitlines()
            if line.startswith("COPY")
        ), f"{dockerfile} never COPYs the lock, so the install would fail"

    @pytest.mark.parametrize("dockerfile", ["Dockerfile-batch", "Dockerfile-lambda"])
    def test_base_image_is_digest_pinned(self, dockerfile):
        body = (APP / dockerfile).read_text(encoding="utf-8")
        from_lines = [ln for ln in body.splitlines() if ln.startswith("FROM ")]
        assert from_lines
        for line in from_lines:
            assert "@sha256:" in line, f"unpinned base image: {line}"


class TestGateToolsArePinned:
    """The tools that decide what merges must be the same locally and in CI.

    They were declared as floors, so CI installed whatever was newest while a
    developer kept whatever was installed. Measured 2026-08-19: ruff 0.15.15 vs
    0.16.3, pytest 7.4.0 vs 9.1.1, mypy 1.15.0 vs 2.3.1 — a major version apart.
    A clean local run therefore said little about CI, and CI could fail on a tool
    release with no code change. It did: ruff 0.16 formats Python inside Markdown
    fences and 0.15 does not.
    """

    GATE_TOOLS = ("pytest", "pytest-cov", "ruff", "mypy")

    def _dev_requirements(self) -> list[str]:
        import re

        text = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text(
            encoding="utf-8"
        )
        block = re.search(r"^dev = \[(.*?)^\]", text, re.S | re.M)
        assert block, "dev extra not found in pyproject.toml"
        return re.findall(r'"([^"]+)"', block.group(1))

    @pytest.mark.parametrize("tool", GATE_TOOLS)
    def test_pinned_to_an_exact_version(self, tool):
        entries = self._dev_requirements()
        entry = next(
            (e for e in entries if e.split("[")[0].split("=")[0] == tool), None
        )
        assert entry, f"{tool} missing from the dev extra"
        assert "==" in entry, (
            f"{tool} is declared as '{entry}'. A gate tool on a floor lets CI and "
            "local diverge; pin it exactly."
        )

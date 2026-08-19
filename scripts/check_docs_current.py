"""Check that the docs still describe the code, by measuring instead of reading.

Stale prose has slipped through this repo repeatedly, and always the same way: the
code was right and the documentation was quietly wrong. Both READMEs claimed 71
pinned dependencies for weeks after the lock dropped to 65; an alert count said
"four" when there were six; the crawl-health description covered one trigger after
a second was added. Prose nobody verifies is worse than no prose, because it is
trusted.

So every numeric claim in the docs is checked against the artefact that produces
it. Run this as the LAST step of a change, before reporting it done:

    python scripts/check_docs_current.py

``tests/test_docs.py`` covers what a test can cover — markdown hazards, the
dependency count, README structural parity, template config keys. Two things it
cannot: the test COUNT (self-referential) and COVERAGE (not knowable from inside a
coverage run). Those need a subprocess, which is why this lives here and not there.

Exits non-zero on any mismatch, so it can gate a commit.
"""

from __future__ import annotations

import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
# Interpreter running this script — the project's env, not whatever `python` is.
PYTHON = sys.executable

failures: list[str] = []
passes: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    (passes if ok else failures).append(f"{label}  ({detail})" if detail else label)


def measure_suite() -> tuple[int, float]:
    """Run the suite and return (test count, coverage percent).

    Regenerates coverage.xml rather than reading whatever is on disk: an earlier
    run's file is exactly how a "verified" number goes stale. The first draft of
    this check read a stale one and reported a mismatch that did not exist.
    """
    result = subprocess.run(
        [
            PYTHON,
            "-m",
            "pytest",
            "-q",
            "--cov=app",
            "--cov-report=xml",
            "-m",
            "not live",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    match = re.search(r"(\d+) passed", result.stdout)
    if not match:
        raise SystemExit(f"could not read the test count:\n{result.stdout[-2000:]}")
    rate = ET.parse(REPO / "coverage.xml").getroot().get("line-rate")
    return int(match.group(1)), round(100 * float(rate or 0), 2)


def main() -> int:
    tests, coverage = measure_suite()
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    readme_ko = (REPO / "README.ko.md").read_text(encoding="utf-8")
    design = (REPO / "docs" / "design.md").read_text(encoding="utf-8")
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    # -- test count and coverage, quoted in four places ---------------------- #
    for name, text, pattern in (
        ("README.md", readme, r"\((\d+) tests, ([\d.]+)% cov\)"),
        ("README.ko.md", readme_ko, r"\((\d+)개 테스트, 커버리지 ([\d.]+)%\)"),
        ("design.md", design, r"(\d+)개 테스트에 커버리지 ([\d.]+)%"),
    ):
        found = re.search(pattern, text)
        if not found:
            check(f"{name} states a test/coverage figure", False, "claim not found")
            continue
        stated_tests, stated_cov = int(found.group(1)), float(found.group(2))
        check(
            f"{name} test count",
            stated_tests == tests,
            f"doc {stated_tests} / actual {tests}",
        )
        # The docs round to one decimal, so allow that much slack and no more.
        check(
            f"{name} coverage",
            abs(stated_cov - coverage) <= 0.05,
            f"doc {stated_cov}% / actual {coverage}%",
        )

    found = re.search(r"coverage is at ([\d.]+)%", ci)
    check(
        "ci.yml coverage comment",
        bool(found) and abs(float(found.group(1)) - coverage) <= 0.05,
        f"doc {found.group(1) if found else '?'}% / actual {coverage}%",
    )
    found = re.search(r"--cov-fail-under=(\d+)", ci)
    floor = int(found.group(1)) if found else 0
    check(
        "ci.yml floor is below actual",
        floor <= coverage,
        f"floor {floor} / actual {coverage}",
    )
    check(
        "ci.yml floor is a real ratchet",
        coverage - floor <= 2.5,
        f"{coverage - floor:.2f} points of slack",
    )

    # -- pinned dependency count -------------------------------------------- #
    pinned = sum(
        1
        for line in (REPO / "app" / "requirements.lock")
        .read_text(encoding="utf-8")
        .splitlines()
        if "==" in line and not line.lstrip().startswith("#")
    )
    for name, text, pattern in (
        ("README.md", readme, r"all (\d+) Python"),
        ("README.ko.md", readme_ko, r"Python 의존성 (\d+)개"),
        ("design.md repo tree", design, r"실제 설치 버전 (\d+)개"),
        ("design.md §19", design, r"전이 의존성까지 (\d+)개"),
    ):
        found = re.search(pattern, text)
        check(
            f"{name} dependency count",
            bool(found) and int(found.group(1)) == pinned,
            f"doc {found.group(1) if found else '?'} / lock {pinned}",
        )

    # -- config facts the docs assert --------------------------------------- #
    for stage in ("dev", "prod"):
        path = REPO / "app" / "configs" / f"config-{stage}.yaml"
        if not path.is_file():  # gitignored; absent on a fresh clone
            continue
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        urls = config["scraping"]["rss_urls"] or []
        for pattern in config["scraping"].get("expected_flaky_urls") or []:
            host = pattern.split("/")[0]
            # A suppressed source must be explained somewhere, or the next reader
            # cannot tell a deliberate silence from a forgotten one.
            check(
                f"design.md explains suppressed source {host} ({stage})",
                host in design,
                "",
            )
            if not any(pattern in url for url in urls):
                # Matching nothing is allowed — prod keeps one as a placeholder for
                # the day its source list widens — but only if it says so, or it
                # reads like an active suppression that is silently doing nothing.
                check(
                    f"config-{stage} labels inert flaky pattern {pattern!r}",
                    "placeholder" in path.read_text(encoding="utf-8"),
                    "matches no configured source and is not labelled",
                )

    print(f"measured: {tests} tests, {coverage:.2f}% coverage, {pinned} pinned deps\n")
    for line in passes:
        print(f"  ok   {line}")
    for line in failures:
        print(f"  FAIL {line}")
    print()
    if failures:
        print(f"{len(failures)} doc claim(s) no longer match the code.")
        return 1
    print(f"All {len(passes)} doc claims match the code.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

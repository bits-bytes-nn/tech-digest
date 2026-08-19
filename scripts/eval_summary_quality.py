"""Score generated summaries against the output-quality rubric.

``eval_filtering.py`` answers "did we pick the right article?". This answers "is
the write-up any good?" — the part the reader actually receives, and therefore
the thing prompt tuning has to be steered by.

Every check is deterministic and maps onto a specific instruction in the
summarization prompt (see ``app/src/quality_metrics``), so a low dimension points
at the sentence of the prompt to change. No AWS calls, no cost.

    python scripts/eval_summary_quality.py                     # latest run
    python scripts/eval_summary_quality.py --date 2026-08-19
    python scripts/eval_summary_quality.py --compare 2026-07-11 2026-08-19
    python scripts/eval_summary_quality.py --min-overall 0.80   # use as a gate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.src.quality_metrics import (
    QualityReport,
    SummaryQuality,
    evaluate_summary,
)

INPUTS_DIR = Path(__file__).resolve().parent.parent / "inputs"


def available_dates() -> list[str]:
    return sorted(p.name for p in INPUTS_DIR.iterdir() if (p / "").is_dir())


def load_report(date: str) -> QualityReport:
    directory = INPUTS_DIR / date
    if not directory.is_dir():
        raise SystemExit(f"no such run: {directory}")
    results = []
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data.get("summary"):
            continue
        results.append(
            evaluate_summary(
                article_id=data.get("title", path.stem),
                summary_html=data["summary"],
                one_liner=data.get("one_liner", ""),
            )
        )
    if not results:
        raise SystemExit(f"no summarized articles in {directory}")
    return QualityReport(results=results)


def print_details(report: QualityReport) -> None:
    """The evidence behind each score, so a number can be checked not trusted."""
    for result in report.results:
        issues = []
        if result.missing_sections:
            issues.append(f"missing sections: {' '.join(result.missing_sections)}")
        if result.han_da:
            issues.append(f"한다체 {len(result.han_da)}: {result.han_da[:2]}")
        if result.repeated_quantities:
            issues.append(f"📊 repeats {result.repeated_quantities[:4]}")
        if result.subsection_headers:
            issues.append(f"subsection headers: {result.subsection_headers}")
        if result.absence_filler:
            issues.append(f"'no information' filler: {result.absence_filler}")
        if result.banned_closings:
            issues.append(f"banned closing: {result.banned_closings}")
        if result.images and result.captioned_images < result.images:
            issues.append(
                f"uncaptioned images: {result.images - result.captioned_images}"
                f"/{result.images}"
            )
        if result.lede_score < 1.0:
            issues.append(f"lede ({len(result.lede)} chars): {result.lede[:60]!r}")
        if issues:
            print(f"\n  {result.article_id[:64]}")
            for issue in issues:
                print(f"      - {issue}")
            print(
                f"      quantities={result.quantity_count} code={result.code_blocks} "
                f"tables={result.tables} images={result.images}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="run directory under inputs/ (default: latest)")
    parser.add_argument(
        "--compare", nargs=2, metavar=("BEFORE", "AFTER"), help="diff two runs"
    )
    parser.add_argument(
        "--min-overall",
        type=float,
        default=None,
        help="exit non-zero below this overall score (use as a gate)",
    )
    args = parser.parse_args()

    if args.compare:
        before, after = (load_report(d) for d in args.compare)
        print(
            f"{'dimension':18s} {args.compare[0]:>12s} {args.compare[1]:>12s} {'Δ':>8s}"
        )
        print("-" * 54)
        for dimension in SummaryQuality.DIMENSIONS:
            b, a = before.mean(dimension), after.mean(dimension)
            flag = "" if abs(a - b) < 0.005 else ("  ↑" if a > b else "  ↓")
            print(f"{dimension:18s} {b:>12.2f} {a:>12.2f} {a - b:>+8.2f}{flag}")
        print("-" * 54)
        print(
            f"{'overall':18s} {before.overall:>12.3f} {after.overall:>12.3f} "
            f"{after.overall - before.overall:>+8.3f}"
        )
        return 0

    date = args.date or (available_dates()[-1] if available_dates() else None)
    if not date:
        raise SystemExit("no runs under inputs/")
    report = load_report(date)
    print(f"run {date} — {len(report.results)} article(s)\n")
    print(report.format_table())
    print_details(report)

    if args.min_overall is not None and report.overall < args.min_overall:
        print(
            f"\nFAIL: overall {report.overall:.3f} < {args.min_overall:.2f}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

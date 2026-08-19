"""Score generated summaries against the output-quality rubric.

``eval_filtering.py`` answers "did we pick the right article?". This answers "is
the write-up any good?" — the part the reader actually receives, and therefore
the thing prompt tuning has to be steered by.

Every check is deterministic and maps onto a specific instruction in the
summarization prompt (see ``app/src/quality_metrics``), so a low dimension points
at the sentence of the prompt to change.

Scoring is free and offline. ``--regenerate`` is the one exception: it re-runs the
summarization chain over a past run's stored articles so a prompt edit can be
A/B'd on identical input, and that makes real Bedrock calls.

    python scripts/eval_summary_quality.py                     # latest run
    python scripts/eval_summary_quality.py --date 2026-08-19
    python scripts/eval_summary_quality.py --compare 2026-07-11 2026-08-19
    python scripts/eval_summary_quality.py --min-overall 0.80   # use as a gate
    python scripts/eval_summary_quality.py --regenerate 2026-08-19   # COSTS MONEY
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.src.quality_metrics import (
    QualityReport,
    SummaryQuality,
    evaluate_summary,
    is_korean,
    visible_text,
)

INPUTS_DIR = Path(__file__).resolve().parent.parent / "inputs"


def available_dates() -> list[str]:
    """Real runs, newest last. ``--regenerate`` output is excluded on purpose: a
    suffixed directory sorts after the date it came from, so it would otherwise
    become "the latest run" and quietly make every default comparison an A/B
    against itself."""
    return sorted(
        p.name
        for p in INPUTS_DIR.iterdir()
        if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.name)
    )


def load_report(date: str) -> QualityReport:
    directory = INPUTS_DIR / date
    if not directory.is_dir():
        raise SystemExit(f"no such run: {directory}")
    results = []
    non_korean = []
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data.get("summary"):
            continue
        # Refuse rather than score: six of the eleven dimensions are Korean-only
        # and return full marks on English prose for want of anything to match,
        # so an English issue would produce a number that looks like a mediocre
        # pass while measuring nothing. See the note in ``quality_metrics``.
        if not is_korean(visible_text(data["summary"])):
            non_korean.append(data.get("title", path.stem))
            continue
        results.append(
            evaluate_summary(
                article_id=data.get("title", path.stem),
                summary_html=data["summary"],
                one_liner=data.get("one_liner", ""),
            )
        )
    if non_korean:
        raise SystemExit(
            f"{len(non_korean)} summary/summaries in {directory} are not Korean "
            f"(e.g. {non_korean[0][:60]!r}). This rubric only measures the Korean "
            f"prompt's rules — register, 번역투, the cliché list and both length "
            f"bands have no English equivalent here — so scoring them would "
            f"report a pass it did not measure."
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
        # Rates, not raw counts: these two dimensions score density, so the count
        # alone would look alarming on a long summary and clean on a short one.
        for label, hits, score in (
            ("번역투", result.translationese, result.translationese_score),
            ("상투구", result.cliches, result.cliche_score),
        ):
            if score < 1.0:
                total = sum(hits.values())
                rate = 1000 * total / result.length if result.length else 0.0
                worst = sorted(hits.items(), key=lambda kv: -kv[1])[:3]
                detail = ", ".join(f"{name} {n}" for name, n in worst)
                issues.append(f"{label} {rate:.1f}/1k자 ({total}회) — {detail}")
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


def regenerate(date: str, stage: str, profile: str | None, suffix: str) -> str:
    """Re-summarize a past run's articles with the CURRENT prompt.

    An A/B on one variable. A prompt edit is only worth keeping if the rubric
    moves, and the rubric only moves comparably when the input is identical — so
    this reuses the stored ``content`` of a past run rather than crawling again.
    Filtering, rendering and email are all skipped: only the summarization chain
    runs, on the handful of articles that were actually published.

    COST NOTE: real Bedrock InvokeModel calls, one per article.
    """
    import boto3

    os.environ.setdefault("CONFIG_FILE_SUFFIX", stage)
    from app.configs import Config
    from app.src import Post, Summarizer, SummarizerSettings

    source = INPUTS_DIR / date
    stored = [
        json.loads(p.read_text(encoding="utf-8")) for p in sorted(source.glob("*.json"))
    ]
    stored = [d for d in stored if d.get("content")]
    if not stored:
        raise SystemExit(f"no articles with stored content in {source}")

    config = Config.load()
    profile_name = profile if profile is not None else config.resources.profile_name
    session = boto3.Session(
        region_name=config.resources.bedrock_region_name,
        profile_name=profile_name or None,
    )
    # The production Summarizer, so the A/B exercises the real chain — same
    # prompt, model and thinking settings — instead of a re-implementation.
    summarizer = Summarizer(
        session, SummarizerSettings.model_validate(config.summarization.model_dump())
    )
    posts = [
        Post(
            title=d["title"],
            link=d["link"],
            published_date=d["published_date"],
            content=d["content"],
            images=d.get("images", []),
            source=d.get("source", "unknown"),
            score=d.get("score", 0.0),
        )
        for d in stored
    ]
    print(f"re-summarizing {len(posts)} article(s) from {date} with the current prompt")
    summarized = summarizer._summarize_posts(posts)

    if len(summarized) < len(posts):
        print(f"WARNING: {len(posts) - len(summarized)} summarization(s) failed")

    target = INPUTS_DIR / f"{date}{suffix}"
    target.mkdir(parents=True, exist_ok=True)
    # Matched by link, not by index: ``_summarize_posts`` returns only the posts
    # that succeeded, so zipping against ``stored`` would silently pair an
    # article's metadata with a different article's summary.
    by_link = {d["link"]: d for d in stored}
    for post in summarized:
        payload = by_link[post.link] | {
            "summary": post.summary,
            "one_liner": post.one_liner,
            "tags": post.tags,
            "urls": post.urls,
        }
        stem = post.title[:60].strip().replace("/", "-") or post.link[-40:]
        (target / f"{stem}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"wrote {len(summarized)} summaries to {target}")
    return target.name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="run directory under inputs/ (default: latest)")
    parser.add_argument(
        "--compare", nargs=2, metavar=("BEFORE", "AFTER"), help="diff two runs"
    )
    parser.add_argument(
        "--regenerate",
        metavar="DATE",
        help="re-summarize that run's stored articles with the current prompt and "
        "diff the result against it (COSTS MONEY: one Bedrock call per article)",
    )
    parser.add_argument(
        "--suffix", default="-rerun", help="suffix for the --regenerate output dir"
    )
    parser.add_argument(
        "--stage", default="dev", help="Config stage (CONFIG_FILE_SUFFIX)"
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="AWS profile override; pass '' for ambient env credentials",
    )
    parser.add_argument(
        "--min-overall",
        type=float,
        default=None,
        help="exit non-zero below this overall score (use as a gate)",
    )
    args = parser.parse_args()

    if args.regenerate:
        produced = regenerate(args.regenerate, args.stage, args.profile, args.suffix)
        args.compare = [args.regenerate, produced]

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

"""Replay the relevance filter over past weeks' real candidates, without labels.

A rubric change is judged by what it does to real issues, not by a labelled set:

    # 1. Freeze the inputs: crawl the candidates of the last N weekly windows once.
    python scripts/replay_filter.py collect --stage dev --weeks 6

    # 2. Score them (COSTS MONEY: one filter call per post per variant).
    python scripts/replay_filter.py score --stage dev --label before
    python scripts/replay_filter.py score --stage dev --label before --variant promo

    # 3. Compare two scorings: line-ups that changed, and the invariants.
    python scripts/replay_filter.py compare --stage dev before after

Variants perturb the input in ways whose effect on the score is known in
direction, so they test the rubric without anyone deciding what the right score
is:

- ``promo``: a promotional paragraph is appended. The score must not rise.
- ``no_included``: the included-topic list is emptied. The score must not rise,
  since included topics are meant to rank ABOVE comparable other content.
- ``shuffled``: the included-topic list is reversed. The band must not change.
- ``repeat``: the unchanged input, scored again. The models score with no fixed
  sampling, so this is the noise floor the other variants are read against.

Everything lands in ``inputs/replay/`` (gitignored, like every other run artefact).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

logger = logging.getLogger(__name__)

OUT_DIR = REPO / "inputs" / "replay"
PROMO = (
    "Ready to get started? Sign up for our free webinar series, book a demo with our "
    "sales team today, and see why industry leaders trust our platform. Contact us "
    "to unlock exclusive pricing for a limited time."
)
VARIANTS = ("base", "promo", "no_included", "shuffled", "repeat")


def _config(stage: str) -> Any:
    os.environ["CONFIG_FILE_SUFFIX"] = stage
    from app.configs import Config

    return Config.load()


def _candidates_path(stage: str) -> Path:
    return OUT_DIR / f"candidates-{stage}.json"


def _scores_path(stage: str, label: str, variant: str) -> Path:
    return OUT_DIR / f"scores-{stage}-{label}-{variant}.json"


def collect(stage: str, weeks: int) -> Path:
    """Crawl the last ``weeks`` windows once and keep what passes the gate.

    Feeds keep only their latest items, so older weeks come back thinner than
    they were live; the per-week counts make that visible.
    """
    from app.domain.post import get_date_range
    from app.src.feed_parser import PostCollector

    config = _config(stage)
    today = datetime.now(UTC).date()
    windows = [
        (issue, *get_date_range(issue, config.scraping.days_back))
        for issue in ((today - timedelta(days=7 * k)).isoformat() for k in range(weeks))
    ]
    earliest, latest = windows[-1][1], windows[0][2]
    posts = PostCollector.from_urls(config.scraping.rss_urls).collect_posts(
        earliest, latest
    )
    out: list[dict[str, Any]] = []
    for post in posts:
        if post.text_length() < config.scraping.min_content_length:
            continue
        week = next(
            w for w, start, end in windows if start <= post.published_date < end
        )
        out.append(
            {
                "week": week,
                "title": post.title,
                "link": str(post.link),
                "source": post.source,
                "published": post.published_date.isoformat(),
                "content": post.content,
            }
        )
    for week, _, _ in windows:
        logger.info(
            "week %s: %d candidate(s)", week, sum(c["week"] == week for c in out)
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = _candidates_path(stage)
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    logger.info("wrote %d candidates to %s", len(out), path)
    return path


def score(
    stage: str, label: str, variant: str, profile: str | None, model: str | None
) -> Path:
    """Score every frozen candidate once under ``variant``, via the pipeline's
    own filter step (same input preparation, retries and response checks)."""
    import boto3

    from app.domain.post import Post
    from app.domain.summary import MODEL_FAILURE_PREFIXES
    from app.src.summarizer import Summarizer, SummarizerSettings

    config = _config(stage)
    candidates = json.loads(_candidates_path(stage).read_text(encoding="utf-8"))
    topics = list(config.summarization.included_topics)
    if variant == "no_included":
        topics = []
    elif variant == "shuffled":
        topics = topics[::-1]
    settings = SummarizerSettings.model_validate(
        config.summarization.model_dump()
        | {
            "min_content_length": config.scraping.min_content_length,
            "included_topics": topics,
            # Every scored post then lands in filtered_out_posts WITH its reason,
            # which a passing post would not carry.
            "min_score": 1.0,
        }
        | ({"filtering_model_id": model} if model else {})
    )
    session = boto3.Session(
        region_name=config.resources.bedrock_region_name,
        profile_name=profile if profile is not None else config.resources.profile_name,
    )
    summarizer = Summarizer(session, settings)
    posts = [
        Post(
            title=c["title"],
            link=c["link"],
            source=c["source"],
            published_date=datetime.fromisoformat(c["published"]),
            content=c["content"] + (f"<p>{PROMO}</p>" if variant == "promo" else ""),
        )
        for c in candidates
    ]
    summarizer._filter_posts(posts)
    results = [
        {
            "link": post.link,
            "score": None if reason.startswith(MODEL_FAILURE_PREFIXES) else post.score,
            "reason": reason,
        }
        for post, reason in summarizer.filtered_out_posts
    ]
    failed = sum(r["score"] is None for r in results)
    path = _scores_path(stage, label, variant)
    path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    logger.info("scored %d (%d failed) -> %s", len(results), failed, path)
    return path


def _load_scores(stage: str, label: str, variant: str) -> dict[str, dict[str, Any]]:
    path = _scores_path(stage, label, variant)
    if not path.is_file():
        return {}
    return {r["link"]: r for r in json.loads(path.read_text(encoding="utf-8"))}


def _lineup(candidates: list[dict[str, Any]], scores: dict, settings: Any) -> list:
    """The posts the pipeline would ship for one week, using its own selection."""
    from app.domain.post import Post
    from app.domain.summary import select_for_digest

    posts = []
    for c in candidates:
        s = scores.get(c["link"], {}).get("score")
        if s is None or s < settings.min_score:
            continue
        post = Post(
            title=c["title"],
            link=c["link"],
            source=c["source"],
            published_date=datetime.fromisoformat(c["published"]),
        )
        post.score = s
        posts.append(post)
    return select_for_digest(posts, settings.max_posts, settings.max_per_source)


def compare(stage: str, before: str, after: str) -> int:
    from app.domain.eval_metrics import band_of_score, is_anchor
    from app.src.summarizer import SummarizerSettings

    config = _config(stage)
    settings = SummarizerSettings.model_validate(config.summarization.model_dump())
    # The pipeline's own INFO lines (a line-up log per week) would bury the report.
    logging.getLogger("app").setLevel(logging.WARNING)
    candidates = json.loads(_candidates_path(stage).read_text(encoding="utf-8"))
    by_link = {c["link"]: c for c in candidates}
    runs = {label: _load_scores(stage, label, "base") for label in (before, after)}
    floor = settings.min_score

    for label, scores in runs.items():
        values = [r["score"] for r in scores.values() if r["score"] is not None]
        print(
            f"{label}: {len(values)} scored, {sum(v >= floor for v in values)} >= "
            f"min_score {floor}, {sum(map(is_anchor, values))}/{len(values)} on an anchor"
        )

    changed_weeks = 0
    for week in sorted({c["week"] for c in candidates}, reverse=True):
        pool = [c for c in candidates if c["week"] == week]
        a = [p.link for p in _lineup(pool, runs[before], settings)]
        b = [p.link for p in _lineup(pool, runs[after], settings)]
        if set(a) == set(b):
            note = "same line-up" if a == b else "same posts, re-ordered"
            print(f"\n{week}: {note} ({len(a)} post(s))")
            continue
        changed_weeks += 1
        print(f"\n{week}: line-up changed")
        for link in dict.fromkeys(a + b):
            mark = "=" if link in a and link in b else ("-" if link in a else "+")
            sa = runs[before].get(link, {})
            sb = runs[after].get(link, {})
            print(
                f"  {mark} {by_link[link]['title'][:70]}"
                f"  [{sa.get('score')} -> {sb.get('score')}]"
            )
            if mark != "=":
                print(f"      {before}: {sa.get('reason', '')[:220]}")
                print(f"      {after}:  {sb.get('reason', '')[:220]}")
    print(f"\n{changed_weeks} week(s) with a different set of posts")

    # Each check is reported next to the same count for an unchanged rescoring:
    # with no fixed sampling, only a rate above that floor says anything.
    base = runs[after]
    repeat = _load_scores(stage, after, "repeat")
    checks = [
        ("rises", "promo", lambda b, v: v > b),
        ("rises", "no_included", lambda b, v: v > b),
    ]
    if len(settings.included_topics) > 1:
        checks.append(
            (
                "changes band",
                "shuffled",
                lambda b, v: band_of_score(v) != band_of_score(b),
            )
        )

    def count(other: dict, test: Any) -> tuple[int, int]:
        pairs = [
            (base[k]["score"], other[k]["score"])
            for k in other
            if k in base
            and base[k]["score"] is not None
            and other[k]["score"] is not None
        ]
        return sum(test(b, v) for b, v in pairs), len(pairs)

    print(f"\nInvariants ({after}), each against the repeat-scoring noise floor:")
    for what, variant, broken in checks:
        perturbed = _load_scores(stage, after, variant)
        if not perturbed or not repeat:
            print(f"  {variant}: not scored (needs {variant} and repeat)")
            continue
        hits, n = count(perturbed, broken)
        noise, _ = count(repeat, broken)
        print(f"  {variant}: score {what} for {hits}/{n} (repeat: {noise})")
    if repeat:
        crossed, n = 0, 0
        for k, r in repeat.items():
            if k in base and None not in (base[k]["score"], r["score"]):
                n += 1
                crossed += (base[k]["score"] >= floor) != (r["score"] >= floor)
        print(f"  repeat: {crossed}/{n} posts land on the other side of min_score")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("collect")
    p.add_argument("--stage", default="dev")
    p.add_argument("--weeks", type=int, default=6)
    p = sub.add_parser("score")
    p.add_argument("--stage", default="dev")
    p.add_argument("--label", required=True)
    p.add_argument("--variant", choices=VARIANTS, default="base")
    p.add_argument("--profile", default=None)
    p.add_argument("--model", default=None, help="filtering model id override")
    p = sub.add_parser("compare")
    p.add_argument("--stage", default="dev")
    p.add_argument("before")
    p.add_argument("after")
    args = parser.parse_args()

    if args.command == "collect":
        collect(args.stage, args.weeks)
    elif args.command == "score":
        score(args.stage, args.label, args.variant, args.profile, args.model)
    else:
        return compare(args.stage, args.before, args.after)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

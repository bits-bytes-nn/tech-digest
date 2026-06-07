"""Evaluate filtering-score behavior against a labeled eval set.

Measures, per article (scored ``--repeats`` times): determinism (identical score
across repeats — expected at temperature=0 with thinking OFF), adherence to the
0.05 anchor grid, and alignment of the mean score with the expected quality band.

COST NOTE: ``--live`` makes real Amazon Bedrock InvokeModel calls
(articles x repeats) and therefore incurs cost. Without ``--live`` it does a
dry run (validates the eval set + harness, no AWS calls, no cost).

Usage:
    # Dry run — no AWS, no cost (validates the set + analysis wiring):
    python scripts/eval_filtering.py

    # Live run — real Bedrock calls (cost!); requires AWS creds + config:
    python scripts/eval_filtering.py --live --repeats 3 --stage dev
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from app.src import logger
from app.src.eval_metrics import band_of_score, build_report, is_on_grid

EVAL_SET = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "eval_data"
    / "filtering_eval_set.json"
)


def load_eval_set(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _band_anchor_score(band: str) -> float:
    """A representative on-grid (0.05) score that lands in ``band``.

    Derived by scanning the grid through ``band_of_score`` rather than hardcoding
    thresholds, so it stays correct if the band cutoffs change. Used only by the
    zero-cost dry-run to exercise the reporting path.
    """
    candidates = [i * 0.05 for i in range(20, -1, -1)]  # 1.00 .. 0.00
    for score in candidates:
        if is_on_grid(score) and band_of_score(score) == band:
            return round(score, 2)
    return 0.0


def _score_live(
    data: dict, repeats: int, stage: str, profile: str | None = None
) -> dict[str, list[float]]:
    """Build the real filter chain and score each article ``repeats`` times.

    Imported lazily so the dry-run path needs no boto3/LangChain/config.
    """
    import boto3

    os.environ.setdefault("CONFIG_FILE_SUFFIX", stage)
    from app.configs import Config
    from app.src import FilteringCriteria, Summarizer

    config = Config.load()
    # --profile overrides the config's profile_name; an empty/None profile means
    # "use the ambient environment credentials (AWS_PROFILE / env vars)".
    profile_name = profile if profile is not None else config.resources.profile_name
    session = boto3.Session(
        region_name=config.resources.bedrock_region_name,
        profile_name=profile_name or None,
    )
    # Reuse the production Summarizer wiring so we evaluate the REAL chain
    # (same prompt, model, temperature, thinking settings) — not a re-impl.
    summarizer = Summarizer(
        session,
        filtering_model_id=config.summarization.filtering_model_id,
        summarization_model_id=config.summarization.summarization_model_id,
        filtering_criteria=FilteringCriteria(data.get("criteria", "all")),
        filtering_enable_thinking=config.summarization.filtering_enable_thinking,
        filtering_thinking_budget_tokens=(
            config.summarization.filtering_thinking_budget_tokens
        ),
    )
    included = ", ".join(data.get("included_topics", []))
    excluded = ", ".join(data.get("excluded_topics", []))

    results: dict[str, list[float]] = {}
    for article in data["articles"]:
        aid = article["id"]
        results[aid] = []
        for i in range(repeats):
            response = summarizer.filter.invoke(
                {
                    "post": article["content"],
                    "original_title": article["title"],
                    "included_topics": included,
                    "excluded_topics": excluded,
                }
            )
            try:
                score = float(response.get("score", 0.0))
            except (ValueError, TypeError):
                logger.warning("Unparseable score for '%s' (run %d)", aid, i + 1)
                score = 0.0
            results[aid].append(score)
            logger.info("%s run %d/%d -> %.2f", aid, i + 1, repeats, score)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Make real Bedrock calls (COST). Omit for a no-cost dry run.",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Scorings per article.")
    parser.add_argument(
        "--stage", default="dev", help="Config stage (CONFIG_FILE_SUFFIX)."
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="AWS profile override. Pass '' to use ambient env credentials "
        "(AWS_PROFILE / env vars) instead of the config's profile_name.",
    )
    args = parser.parse_args()

    data = load_eval_set(EVAL_SET)
    expected = {a["id"]: a["expected_band"] for a in data["articles"]}
    logger.info("Loaded %d eval articles from %s", len(data["articles"]), EVAL_SET)

    if not args.live:
        # Exercise the full analysis wiring (build_report + format_table) at zero
        # cost by feeding each article a synthetic on-grid score in its expected
        # band. This is what makes the CI dry-run a real smoke test of the
        # reporting path — not just "the JSON parsed" — without any Bedrock call.
        synthetic = {
            a["id"]: [_band_anchor_score(a["expected_band"])] for a in data["articles"]
        }
        report = build_report(synthetic, expected)
        print(report.format_table())
        if not (report.grid_rate == 1.0 and report.band_match_rate == 1.0):
            logger.error(
                "Dry-run self-check failed: synthetic scores should be 100%% "
                "on-grid and band-matching (got grid=%.0f%%, band=%.0f%%). "
                "This indicates a regression in the eval-metrics wiring.",
                report.grid_rate * 100,
                report.band_match_rate * 100,
            )
            return 1
        logger.info(
            "DRY RUN (no AWS, no cost): analysis wiring OK for %d articles; "
            "expected bands: %s. Re-run with --live to score against Bedrock.",
            len(data["articles"]),
            sorted(set(expected.values())),
        )
        return 0

    logger.warning(
        "LIVE mode: making %d real Bedrock calls (%d articles x %d repeats). "
        "This incurs cost.",
        len(data["articles"]) * args.repeats,
        len(data["articles"]),
        args.repeats,
    )
    results = _score_live(data, args.repeats, args.stage, args.profile)
    report = build_report(results, expected)
    print(report.format_table())

    # Non-zero exit if behavior regresses, so this is usable as a gate.
    ok = (
        report.determinism_rate >= 0.99
        and report.grid_rate >= 0.99
        and report.band_match_rate >= 0.80
    )
    if not ok:
        logger.error(
            "Filtering behavior below thresholds "
            "(determinism>=99%%, on-grid>=99%%, band-match>=80%%)."
        )
        return 1
    logger.info("Filtering behavior within thresholds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

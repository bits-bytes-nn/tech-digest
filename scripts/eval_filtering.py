"""Evaluate filtering-score behavior against a labeled eval set.

Measures, per article (scored ``--repeats`` times): adherence to the rubric's
anchors, alignment with the expected quality band, and reproducibility (determinism,
band stability, spread).

Only the anchor and band-alignment checks are GATED. Exact determinism is reported
but not gated: models from Sonnet 5 on expose no sampling controls, so greedy
decoding cannot be requested and repeated scorings of identical input differ.
See the comment on the gate in ``main()`` for the measured numbers.

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
from app.src.eval_metrics import ANCHOR_BANDS, build_report

EVAL_SET = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "eval_data"
    / "filtering_eval_set.json"
)


def load_eval_set(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _band_anchor_score(band: str) -> float:
    """The highest anchor in ``band``; the dry run uses it to exercise reporting."""
    return max((a for a, b in ANCHOR_BANDS.items() if b == band), default=0.0)


def _score_live(
    data: dict,
    repeats: int,
    stage: str,
    profile: str | None = None,
    model: str | None = None,
) -> tuple[dict[str, list[float]], int]:
    """Build the real filter chain and score each article ``repeats`` times.

    Imported lazily so the dry-run path needs no boto3/LangChain/config.
    """
    import boto3

    os.environ.setdefault("CONFIG_FILE_SUFFIX", stage)
    from app.configs import Config
    from app.src import FilteringCriteria, Summarizer, SummarizerSettings

    config = Config.load()
    # --profile overrides the config's profile_name; an empty/None profile means
    # "use the ambient environment credentials (AWS_PROFILE / env vars)".
    profile_name = profile if profile is not None else config.resources.profile_name
    session = boto3.Session(
        region_name=config.resources.bedrock_region_name,
        profile_name=profile_name or None,
    )
    # Reuse the production Summarizer wiring so we evaluate the REAL chain
    # (same prompt, model, temperature, thinking settings) — not a re-impl. The
    # eval set picks the criteria; everything else comes from the stage config.
    summarizer = Summarizer(
        session,
        SummarizerSettings.model_validate(
            config.summarization.model_dump()
            | {"filtering_criteria": FilteringCriteria(data.get("criteria", "all"))}
            | ({"filtering_model_id": model} if model else {})
        ),
    )
    included = ", ".join(data.get("included_topics", []))
    excluded = ", ".join(data.get("excluded_topics", []))

    results: dict[str, list[float]] = {}
    failures = 0
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
            # Same rule as the pipeline: a missing, unparseable or out-of-range
            # score is a failed call. Scoring it 0.0 would count as on-anchor and
            # as a correct "reject" for half the set.
            try:
                score = float(response["score"])
            except (KeyError, ValueError, TypeError):
                score = None
            if score is None or not 0.0 <= score <= 1.0:
                logger.error("Failed call for '%s' (run %d): %s", aid, i + 1, response)
                failures += 1
                continue
            results[aid].append(score)
            logger.info("%s run %d/%d -> %.2f", aid, i + 1, repeats, score)
    return results, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Make real Bedrock calls (COST). Omit for a no-cost dry run.",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Scorings per article.")
    parser.add_argument(
        "--model",
        default=None,
        help="Filtering model id to score with instead of the stage config's "
        "(e.g. openai.gpt-6-luna), for comparing candidates on the same set.",
    )
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
        # cost by feeding each article a synthetic on-anchor score in its expected
        # band. This is what makes the CI dry-run a real smoke test of the
        # reporting path — not just "the JSON parsed" — without any Bedrock call.
        synthetic = {
            a["id"]: [_band_anchor_score(a["expected_band"])] for a in data["articles"]
        }
        report = build_report(synthetic, expected)
        print(report.format_table())
        if not (report.anchor_rate == 1.0 and report.band_match_rate == 1.0):
            logger.error(
                "Dry-run self-check failed: synthetic scores should be 100%% "
                "on-anchor and band-matching (got anchor=%.0f%%, band=%.0f%%). "
                "This indicates a regression in the eval-metrics wiring.",
                report.anchor_rate * 100,
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
    results, failures = _score_live(
        data, args.repeats, args.stage, args.profile, args.model
    )
    if failures:
        logger.error("%d filter call(s) failed; the eval is not valid.", failures)
        return 1
    report = build_report(results, expected)
    print(report.format_table())

    # Non-zero exit if behavior regresses, so this is usable as a gate.
    #
    # The gate asserts only the two properties the configured model can actually
    # guarantee, and REPORTS the rest as diagnostics.
    #
    # Determinism is not gated: Claude 5 models take no sampling parameters, so
    # repeated scorings of identical input differ (design.md §9). A gate the
    # default model cannot meet is a gate people stop running, so determinism,
    # band stability and spread are reported as warnings.
    ok = report.anchor_rate >= 0.99 and report.band_match_rate >= 0.80
    if not ok:
        logger.error(
            "Filtering behavior below thresholds (on-anchor>=99%%, band-match>=80%%)."
        )
        return 1
    if report.determinism_rate < 1.0:
        logger.warning(
            "Scores are NOT reproducible: determinism=%.0f%%, band-stable=%.0f%%, "
            "max spread=%.2f. Expected on models without "
            "sampling controls. Practical consequence: a post scoring within "
            "~%.2f of min_score may be included in one run and excluded in the "
            "next. Pin a temperature-capable filtering model if reproducible "
            "scoring matters more than filtering capability.",
            report.determinism_rate * 100,
            report.band_stability_rate * 100,
            report.max_spread,
            report.max_spread,
        )
    logger.info("Filtering behavior within thresholds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

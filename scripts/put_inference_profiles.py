"""Create this project's Bedrock APPLICATION inference profiles, tagged for cost allocation.

On-demand Bedrock has no taggable resource behind ``InvokeModel``, so its token spend
cannot be attributed to a cost-allocation tag — in a shared account (this one bills
several workloads against the same Claude models) the Bedrock line is a single
unattributable total. An application inference profile IS taggable, and invoking
through its ARN attributes the usage to those tags.

Not CloudFormation: the profiles must live in ``resources.bedrock_region_name``
(us-west-2) while the stacks deploy to ``resources.default_region_name``
(ap-northeast-2), and a second regional stack plus its bootstrap is disproportionate
for a reporting concern.

    python scripts/put_inference_profiles.py [--dry-run] [--delete] [--stage dev]

Idempotent: an existing profile with the expected name is left alone. The runtime looks
profiles up by that name and falls back to the system-defined inference profile when
none exists, so running this is optional — it only changes how the usage is BILLED,
never whether a call works.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import boto3

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.src import EnvVars, LanguageModelId, logger
from app.src.model_factory import BedrockCrossRegionModelHelper


def _configured_models(config) -> dict[LanguageModelId, str]:
    """Every model this deployment can invoke, with the setting that selects it.

    A profile is only worth creating for models actually in use — the registry
    carries many that are not. ``fixing_model_id`` is included because the
    output-repair model is invoked on malformed summaries and its spend would
    otherwise be untagged.
    """
    summarization = config.summarization
    wanted: dict[LanguageModelId, list[str]] = {}
    for setting, model in (
        ("summarization.filtering_model_id", summarization.filtering_model_id),
        ("summarization.summarization_model_id", summarization.summarization_model_id),
        ("summarization.greeting_model_id", summarization.greeting_model_id),
        # The output-fixing model. NOT a config key — ``Summarization`` has no
        # ``fixing_model_id`` field, so this mirrors the Summarizer's coded
        # default. Labelled as such rather than as "summarization.fixing_model_id",
        # which named a setting nobody can actually set.
        ("Summarizer default (output-fixing)", LanguageModelId.CLAUDE_V4_5_HAIKU),
    ):
        if model is not None:
            wanted.setdefault(model, []).append(setting)
    return {model: ", ".join(settings) for model, settings in wanted.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="report what would change, write nothing"
    )
    parser.add_argument(
        "--delete", action="store_true", help="delete this project's profiles instead"
    )
    parser.add_argument(
        "--stage",
        default=None,
        help="config stage (default: CONFIG_FILE_SUFFIX or dev)",
    )
    args = parser.parse_args()

    if args.stage:
        os.environ[EnvVars.CONFIG_FILE_SUFFIX.value] = args.stage

    from app.configs import Config

    config = Config.load()
    project, stage = config.resources.project_name, config.resources.stage
    # The runtime derives the profile name from these env vars; set them here so the
    # script and the runtime cannot disagree about what a profile is called. Named
    # through EnvVars rather than as literals for exactly that reason — spelling them
    # out here would have re-introduced the disagreement this comment claims to
    # prevent the moment either enum value was renamed.
    os.environ.setdefault(EnvVars.PROJECT_NAME.value, project)
    os.environ[EnvVars.STAGE.value] = stage

    region = config.resources.bedrock_region_name
    session = boto3.Session(
        region_name=region, profile_name=config.resources.profile_name or None
    )
    client = session.client("bedrock", region_name=region)

    existing: dict[str, str] = {}
    paginator = client.get_paginator("list_inference_profiles")
    for page in paginator.paginate(typeEquals="APPLICATION"):
        for summary in page.get("inferenceProfileSummaries", []):
            existing[summary["inferenceProfileName"]] = summary["inferenceProfileArn"]

    models = _configured_models(config)
    logger.info(
        "project=%s stage=%s region=%s | %d configured model(s) | "
        "%d existing application profile(s)",
        project,
        stage,
        region,
        len(models),
        len(existing),
    )

    account_id: str | None = None
    for model, settings in sorted(models.items(), key=lambda kv: kv[0].value):
        name = BedrockCrossRegionModelHelper.application_profile_name(model)
        arn = existing.get(name)

        if args.delete:
            if not arn:
                logger.info("  absent   : %s", name)
            elif args.dry_run:
                logger.info("  would delete: %s", name)
            else:
                client.delete_inference_profile(inferenceProfileIdentifier=arn)
                logger.info("  deleted  : %s", name)
            continue

        if arn:
            logger.info("  exists   : %s  (%s)", name, settings)
            continue

        # copyFrom takes the SYSTEM-DEFINED cross-region profile the runtime would
        # otherwise use, so the application profile inherits the same routing rather
        # than pinning a single region.
        source = BedrockCrossRegionModelHelper.get_cross_region_model_id(
            session, model, region
        )
        if source.startswith("arn:"):
            source_arn = source
        else:
            if account_id is None:
                account_id = session.client("sts").get_caller_identity()["Account"]
            source_arn = (
                f"arn:aws:bedrock:{region}:{account_id}:inference-profile/{source}"
            )
        if args.dry_run:
            logger.info("  would create: %s  from %s  (%s)", name, source_arn, settings)
            continue
        created = client.create_inference_profile(
            inferenceProfileName=name,
            # Bedrock restricts descriptions to ([0-9a-zA-Z:.][ _-]?)+ — no
            # em-dashes, no doubled separators — so build it from safe characters.
            description=f"{project} {stage} cost attribution for {model.value}",
            modelSource={"copyFrom": source_arn},
            tags=[
                {"key": "Project", "value": project},
                {"key": "Stage", "value": stage},
            ],
        )
        logger.info(
            "  created  : %s  -> %s  (%s)",
            name,
            created["inferenceProfileArn"],
            settings,
        )

    if not args.delete and not args.dry_run:
        logger.info(
            "Activate the 'Project' cost allocation tag in Billing for these to show "
            "up in Cost Explorer (takes up to 24h, and is NOT retroactive)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

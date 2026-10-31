"""The container's entry point and the composition root.

``handler`` builds one adapter per outside system and hands them to the
application services; nothing else in ``app`` constructs an adapter.
"""

import argparse
import functools
import os
import sys
from pathlib import Path
from typing import Any

import boto3

from app.adapters.aws import (
    S3ObjectStore,
    SesMailer,
    SnsAlarmPublisher,
    get_ssm_param_value,
)
from app.adapters.bedrock import BedrockLanguageModelFactory
from app.adapters.feeds import PostCollector
from app.adapters.http import is_email_renderable_image
from app.adapters.rendering import NewsletterBuilder
from app.application.alerts import (
    maybe_send_clipped_newsletter_alert,
    maybe_send_empty_digest_alert,
    maybe_send_model_failure_alert,
    maybe_send_partial_delivery_alert,
    send_crawl_health_alert,
    send_failure_notification,
    send_no_recipients_alert,
)
from app.application.delivery import archive_suffix, process_newsletter_emails
from app.application.pipeline import (
    fetch_and_filter_posts,
    log_run_summary,
    process_posts_and_create_newsletter,
)
from app.configs import Config
from app.domain.constants import AppConstants, EnvVars, Language, LocalPaths, SSMParams
from app.logger import is_running_in_aws, logger

ROOT_DIR: Path = (
    Path("/tmp") if is_running_in_aws() else Path(__file__).resolve().parent.parent
)


def _boto_sessions(config: Config) -> tuple[boto3.Session, boto3.Session]:
    """(default-region session, Bedrock-region session) for this run.

    In AWS the container's job role supplies credentials, so no profile is named:
    the stage configs set ``profile_name`` for local runs, and that profile does
    not exist inside the container.
    """
    profile = None if is_running_in_aws() else config.resources.profile_name
    return (
        boto3.Session(
            region_name=config.resources.default_region_name, profile_name=profile
        ),
        boto3.Session(
            region_name=config.resources.bedrock_region_name, profile_name=profile
        ),
    )


def handler(event: dict[str, Any], context: Any) -> dict[str, int | str]:
    config = Config.load()
    default_boto_session, bedrock_boto_session = _boto_sessions(config)
    # Each stage that calls a model builds its own client, and only when it
    # runs: a run with nothing to summarize never creates one.
    models = functools.partial(
        BedrockLanguageModelFactory, boto_session=bedrock_boto_session
    )
    try:
        _setup_aws_env(config, default_boto_session)
        end_date_str = event.get("END_DATE")
        language_str = event.get("LANGUAGE")
        recipients = event.get("RECIPIENTS")
        # Resolved once so every stage (filtering, greeting, filenames) agrees.
        language = Language(language_str) if language_str else Language.KO
        posts, date_suffix, filtered_out_posts, crawl_report = fetch_and_filter_posts(
            config,
            PostCollector.from_urls,
            models,
            is_email_renderable_image,
            end_date_str=end_date_str,
            language=language,
        )
        log_run_summary(crawl_report, posts, filtered_out_posts, language)
        topic_arn = os.environ.get(EnvVars.TOPIC_ARN.value)
        # Alarms are published from AWS only, and only to a configured topic.
        alarms = (
            SnsAlarmPublisher(default_boto_session, topic_arn)
            if is_running_in_aws() and topic_arn
            else None
        )
        if alarms and (
            crawl_report.failed
            or crawl_report.likely_broken
            or crawl_report.pages_unreadable
        ):
            send_crawl_health_alert(
                alarms,
                config.resources.project_name,
                crawl_report,
                config.scraping.expected_flaky_urls,
            )
        if not posts:
            # Distinguish "nothing survived filtering" from "nothing was
            # collected". The former — especially when every post was dropped by
            # a model error — is a silent failure worth alerting on, since the
            # run still exits successfully and no email goes out.
            if alarms:
                maybe_send_empty_digest_alert(
                    alarms,
                    config.resources.project_name,
                    crawl_report,
                    filtered_out_posts,
                )
            logger.info("No posts to process. Exiting gracefully.")
            return {"statusCode": 200, "body": "No posts found to process."}
        if alarms:
            maybe_send_model_failure_alert(
                alarms,
                config.resources.project_name,
                posts,
                filtered_out_posts,
            )
        store = S3ObjectStore(default_boto_session, config.resources.s3_bucket_name)
        newsletter_path = process_posts_and_create_newsletter(
            posts,
            date_suffix,
            config,
            store,
            models,
            functools.partial(_newsletter_builder, config),
            ROOT_DIR,
            language=language,
            archive_suffix=archive_suffix(
                config, store, date_suffix, language, recipients
            ),
        )
        if alarms:
            maybe_send_clipped_newsletter_alert(
                alarms,
                config.resources.project_name,
                newsletter_path,
                len(posts),
            )
        if config.newsletter.send_emails:
            outcome = process_newsletter_emails(
                newsletter_path,
                date_suffix,
                config,
                store,
                SesMailer(default_boto_session),
                language,
                ROOT_DIR,
                None if recipients is None else recipients.split(","),
            )
            # Everyone was served earlier, apart from addresses SES suppresses.
            retry_of_completed_send = (
                outcome.skipped > 0
                and outcome.skipped + outcome.suppressed == outcome.resolved
            )
            if retry_of_completed_send:
                # Not a failure and not alarm-worthy: the ledger did its job.
                logger.info(
                    "All %d recipient(s) had already received this issue; "
                    "nothing was sent. This run is a retry of a completed send.",
                    outcome.skipped,
                )
            if alarms:
                if not outcome.attempted and not retry_of_completed_send:
                    # Nobody to send to: no recipient resolved, or every one not
                    # already served is on the SES suppression list. The
                    # partial-delivery alert below keys on FAILED sends and
                    # neither case has any, so without this the issue is built,
                    # uploaded and never sent while the run returns 200.
                    send_no_recipients_alert(alarms, config, crawl_report, outcome)
                maybe_send_partial_delivery_alert(
                    alarms,
                    config.resources.project_name,
                    outcome.succeeded,
                    outcome.attempted,
                    outcome.failed,
                    crawl_report,
                )
        return {"statusCode": 200, "body": "Newsletter processed successfully"}
    except Exception as e:
        logger.error("An error occurred: %s", e, exc_info=True)
        topic_arn = os.environ.get(EnvVars.TOPIC_ARN.value)
        if is_running_in_aws() and topic_arn:
            send_failure_notification(
                SnsAlarmPublisher(default_boto_session, topic_arn),
                config.resources.project_name,
                str(e),
            )
        return {"statusCode": 500, "body": f"An error occurred: {e}"}


def _newsletter_builder(
    config: Config, inputs_dir: Path, outputs_dir: Path
) -> NewsletterBuilder:
    """The renderer for one run's saved posts, with this deployment's templates and
    logos."""
    return NewsletterBuilder(
        inputs_dir,
        outputs_dir,
        Path(__file__).resolve().parent / LocalPaths.TEMPLATES_DIR.value,
        logos=config.newsletter.logos,
    )


def _setup_aws_env(config: Config, boto_session: boto3.Session) -> None:
    if not is_running_in_aws():
        return
    try:
        base_path = f"/{config.resources.project_name}/{config.resources.stage}"
        param_name = f"{base_path}/{SSMParams.LANGCHAIN_API_KEY.value}"
        param_value = get_ssm_param_value(boto_session, param_name)
        if param_value:
            os.environ[EnvVars.LANGCHAIN_API_KEY.value] = param_value
            logger.info(
                "Set environment variable '%s' from SSM",
                EnvVars.LANGCHAIN_API_KEY.value,
            )
    except Exception as e:
        logger.error("Failed to get an API key from SSM: %s", e)


def cli(argv: list[str] | None = None) -> int:
    """Parse the container's arguments, run the handler, return the exit code."""
    parser = argparse.ArgumentParser(
        description="Tech Digest: Generate and send AI/ML tech blog digest newsletter"
    )
    parser.add_argument(
        "--end-date", type=str, default=None, help="Base date for newsletter content"
    )
    parser.add_argument(
        "--language", type=str, default=None, help="Language for the newsletter"
    )
    parser.add_argument(
        "--recipients",
        type=str,
        nargs="+",
        default=None,
        help="List of recipient email addresses",
    )
    args = parser.parse_args(argv)
    event = {
        "END_DATE": (
            args.end_date
            if args.end_date and args.end_date.lower() != AppConstants.NULL_STRING
            else None
        ),
        "LANGUAGE": (
            args.language
            if args.language and args.language.lower() != AppConstants.NULL_STRING
            else None
        ),
        "RECIPIENTS": None,
    }
    if args.recipients and args.recipients[0].lower() != AppConstants.NULL_STRING:
        event["RECIPIENTS"] = ",".join(args.recipients)
    logger.info(
        "Processing newsletter with end_date='%s', language='%s', recipients='%s'",
        event["END_DATE"] or "",
        event["LANGUAGE"] or "",
        event["RECIPIENTS"] or "",
    )
    result = handler(event, None)
    # A process reports failure only through its exit code, which is all Batch
    # retries and the FAILED-state alarm can see.
    return 0 if int(result["statusCode"]) < 400 else 1


if __name__ == "__main__":
    sys.exit(cli())

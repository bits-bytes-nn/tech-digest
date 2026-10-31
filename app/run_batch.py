import argparse
import sys
from datetime import UTC, datetime
from typing import Any

import boto3

from app.adapters.aws import (
    get_ssm_param_value,
    submit_batch_job,
    wait_for_batch_job_completion,
)
from app.configs import Config
from app.domain.alarms import validate_email
from app.domain.constants import AppConstants, SSMParams
from app.logger import is_running_in_aws, logger


def main(job_prefix: str, **kwargs: Any) -> None:
    config = Config.load()
    profile_name = None if is_running_in_aws() else config.resources.profile_name
    boto_session = boto3.Session(
        region_name=config.resources.default_region_name, profile_name=profile_name
    )
    job_queue_name, job_definition_name = get_batch_job_names(boto_session, config)
    if not job_queue_name or not job_definition_name:
        raise ValueError(
            "Batch job queue or definition name not found in SSM parameters"
        )
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    job_name = f"{config.resources.project_name}-{config.resources.stage}-{job_prefix}-{timestamp}"
    sanitized_params = sanitize_parameters(kwargs)
    logger.info(
        "Submitting batch job '%s' with parameters: %s", job_name, sanitized_params
    )
    job_id = submit_batch_job(
        boto_session,
        job_name,
        job_queue_name,
        job_definition_name,
        parameters=sanitized_params,
    )
    if not job_id:
        raise RuntimeError("Failed to submit batch job")
    logger.info("Batch job submitted with ID '%s'", job_id)
    if not wait_for_batch_job_completion(boto_session, job_id):
        raise RuntimeError(f"Batch job '{job_id}' failed or timed out.")


def get_batch_job_names(
    boto_session: boto3.Session, config: Config
) -> tuple[str | None, str | None]:
    base_path = f"/{config.resources.project_name}/{config.resources.stage}"
    try:
        queue = get_ssm_param_value(
            boto_session, f"{base_path}/{SSMParams.BATCH_JOB_QUEUE.value}"
        )
        definition = get_ssm_param_value(
            boto_session, f"{base_path}/{SSMParams.BATCH_JOB_DEFINITION.value}"
        )
        return queue, definition
    except Exception as e:
        logger.error("Failed to retrieve batch job names from SSM: %s", e)
        return None, None


def sanitize_parameters(params: dict[str, Any]) -> dict[str, str]:
    """Batch job parameters as strings, an unset one as the null sentinel.

    Every one is sent because the job command names each as ``Ref::<key>``, and an
    unsubstituted ``Ref::`` reaches ``main.cli`` verbatim; ``main.cli`` decodes the
    sentinel back to "not given".
    """
    return {
        key: AppConstants.NULL_STRING if value is None else str(value)
        for key, value in params.items()
    }


def is_valid_date(date_str: str) -> bool:
    """Whether ``date_str`` is an accepted ``--end-date`` (or the null sentinel)."""
    if not date_str or date_str == AppConstants.NULL_STRING:
        return True
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tech Digest: Submit a newsletter generation batch job"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Base date for newsletter content (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language for the newsletter (e.g., 'ko', 'en')",
    )
    parser.add_argument(
        "--recipients",
        type=str,
        nargs="+",
        default=None,
        help="List of recipient email addresses",
    )
    args = parser.parse_args()

    if args.end_date and not is_valid_date(args.end_date):
        logger.error("Invalid date format for --end-date. Expected format: YYYY-MM-DD")
        sys.exit(1)

    recipients: str | None = None
    if args.recipients and args.recipients[0].lower() != AppConstants.NULL_STRING:
        valid_emails = [email for email in args.recipients if validate_email(email)]
        if len(valid_emails) < len(args.recipients):
            logger.warning(
                "Filtered out %d invalid email addresses",
                len(args.recipients) - len(valid_emails),
            )
        if not valid_emails:
            # No recipients means "the whole S3 list": a mistyped test address
            # must stop here, not reach every subscriber.
            logger.error(
                "--recipients was given but none of the addresses is valid; "
                "refusing to fall back to the full recipients list."
            )
            sys.exit(1)
        recipients = ",".join(valid_emails)

    try:
        main(
            "newsletter",
            end_date=args.end_date,
            language=args.language,
            recipients=recipients,
        )
    except Exception as e:
        logger.error("Failed to submit or execute batch job: %s", e)
        sys.exit(1)

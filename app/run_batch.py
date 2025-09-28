import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import boto3
from pytz import timezone

sys.path.append(str(Path(__file__).parent.parent))
from configs import Config
from src import (
    AppConstants,
    EnvVars,
    SSMParams,
    is_running_in_aws,
    logger,
    get_ssm_param_value,
    submit_batch_job,
    wait_for_batch_job_completion,
)


def main(job_prefix: str, **kwargs: Any) -> None:
    config = Config.load()
    profile_name = (
        os.environ.get(EnvVars.AWS_PROFILE_NAME.value)
        if is_running_in_aws()
        else config.resources.profile_name
    )
    boto_session = boto3.Session(
        region_name=config.resources.default_region_name, profile_name=profile_name
    )
    job_queue_name, job_definition_name = get_batch_job_names(boto_session, config)
    if not job_queue_name or not job_definition_name:
        raise ValueError(
            "Batch job queue or definition name not found in SSM parameters"
        )
    timestamp = datetime.now(timezone("UTC")).strftime("%Y%m%d%H%M%S")
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
    result = {}
    for key, value in params.items():
        if value is None:
            result[key] = AppConstants.NULL_STRING
        elif (
            key == "recipients"
            and isinstance(value, str)
            and value != AppConstants.NULL_STRING
        ):
            result[key] = value.replace(" ", ",")
        else:
            result[key] = str(value)
    return result


def validate_date(date_str: str) -> bool:
    if not date_str or date_str == AppConstants.NULL_STRING:
        return True
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def validate_email(email: str) -> bool:
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email.strip()))


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

    if args.end_date and not validate_date(args.end_date):
        logger.error("Invalid date format for --end-date. Expected format: YYYY-MM-DD")
        sys.exit(1)

    end_date = args.end_date if args.end_date is not None else AppConstants.NULL_STRING
    language = args.language if args.language is not None else AppConstants.NULL_STRING
    recipients_str = AppConstants.NULL_STRING
    if args.recipients:
        if args.recipients[0].lower() != AppConstants.NULL_STRING:
            valid_emails = [email for email in args.recipients if validate_email(email)]
            if len(valid_emails) < len(args.recipients):
                logger.warning(
                    "Filtered out %d invalid email addresses",
                    len(args.recipients) - len(valid_emails),
                )
            if valid_emails:
                recipients_str = " ".join(valid_emails)
            else:
                logger.warning(
                    "No valid email addresses provided, setting recipients to null."
                )

    try:
        main(
            "newsletter",
            end_date=end_date,
            language=language,
            recipients=recipients_str,
        )
    except Exception as e:
        logger.error("Failed to submit or execute batch job: %s", e)
        sys.exit(1)

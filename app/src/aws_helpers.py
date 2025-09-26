import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import make_msgid
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from .logger import logger

DEFAULT_BATCH_POLL_INTERVAL: int = 30
DEFAULT_BATCH_TIMEOUT: int = 3600


def check_and_download_from_s3(
    boto_session: boto3.Session,
    s3_bucket_name: str,
    s3_key: str,
    local_path: Path,
) -> bool:
    if not all([s3_bucket_name, s3_key]):
        logger.error("S3 bucket name and key are required.")
        return False
    s3_client = boto_session.client("s3")
    try:
        s3_client.head_object(Bucket=s3_bucket_name, Key=s3_key)
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            logger.warning("File not found in S3: s3://%s/%s", s3_bucket_name, s3_key)
        else:
            logger.error("Failed to check S3 object existence: %s", e)
        return False
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(s3_bucket_name, s3_key, str(local_path))
        logger.info(
            "Successfully downloaded 's3://%s/%s' to '%s'",
            s3_bucket_name,
            s3_key,
            local_path,
        )
        return True
    except ClientError as e:
        logger.error("Failed to download from S3: %s", e)
        return False


def get_account_id(boto_session: boto3.Session) -> str:
    try:
        sts_client = boto_session.client("sts")
        return sts_client.get_caller_identity()["Account"]
    except ClientError as e:
        logger.error("Failed to get AWS Account ID: %s", e)
        raise


def get_ssm_param_value(boto_session: boto3.Session, param_name: str) -> str:
    ssm_client = boto_session.client("ssm")
    try:
        response = ssm_client.get_parameter(Name=param_name, WithDecryption=True)
        return response["Parameter"]["Value"]
    except ClientError as e:
        logger.error("Failed to get SSM parameter '%s': %s", param_name, e)
        raise


def send_email(
    boto_session: boto3.Session,
    subject: str,
    sender: str,
    recipients: list[str],
    html_content: str,
) -> bool:
    if not recipients:
        logger.warning("No recipients provided for email, skipping send.")
        return False
    ses_client = boto_session.client("ses")
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    if "@" in sender:
        msg["Message-ID"] = make_msgid(domain=sender.split("@")[1])
    else:
        msg["Message-ID"] = make_msgid()
    for header in ["In-Reply-To", "References"]:
        if header in msg:
            del msg[header]
    msg.attach(MIMEText(html_content, "html", "utf-8"))
    try:
        response = ses_client.send_raw_email(
            Source=sender,
            Destinations=recipients,
            RawMessage={"Data": msg.as_string()},
        )
        logger.info(
            "Email sent to %d recipients. MessageId: '%s'",
            len(recipients),
            response["MessageId"],
        )
        return True
    except ClientError as e:
        logger.error("Failed to send email via SES: %s", e)
        return False


def submit_batch_job(
    boto_session: boto3.Session,
    job_name: str,
    job_queue: str,
    job_definition: str,
    parameters: dict[str, str] | None = None,
) -> str | None:
    batch_client = boto_session.client("batch")
    try:
        response = batch_client.submit_job(
            jobName=job_name,
            jobQueue=job_queue,
            jobDefinition=job_definition,
            parameters=parameters or {},
        )
        job_id = response["jobId"]
        logger.info(
            "Successfully submitted Batch job '%s' (Job ID: '%s')", job_name, job_id
        )
        return job_id
    except ClientError as e:
        logger.error("Failed to submit Batch job '%s': %s", job_name, e)
        return None


def upload_to_s3(
    boto_session: boto3.Session,
    local_path: Path,
    s3_bucket_name: str,
    s3_prefix: str = "",
) -> bool:
    if not local_path.is_file():
        logger.error("File to upload does not exist: %s", local_path)
        return False
    if not s3_bucket_name:
        logger.error("S3 bucket name is required for upload.")
        return False
    prefix = s3_prefix.strip("/")
    s3_key = f"{prefix}/{local_path.name}" if prefix else local_path.name
    s3_client = boto_session.client("s3")
    try:
        s3_client.upload_file(str(local_path), s3_bucket_name, s3_key)
        logger.info(
            "Successfully uploaded '%s' to 's3://%s/%s'",
            local_path.name,
            s3_bucket_name,
            s3_key,
        )
        return True
    except ClientError as e:
        logger.error("Failed to upload to S3: %s", e)
        return False


def wait_for_batch_job_completion(
    boto_session: boto3.Session,
    job_id: str,
    poll_interval_seconds: int = DEFAULT_BATCH_POLL_INTERVAL,
    timeout_seconds: int = DEFAULT_BATCH_TIMEOUT,
) -> bool:
    batch_client = boto_session.client("batch")
    start_time = time.time()
    logger.info("Waiting for Batch job '%s' to complete...", job_id)
    while time.time() - start_time < timeout_seconds:
        try:
            response = batch_client.describe_jobs(jobs=[job_id])
            job = response.get("jobs", [{}])[0]
            status = job.get("status")
            if not status:
                logger.error(
                    "Could not find status for job ID '%s'. Aborting wait.", job_id
                )
                return False
            if status == "SUCCEEDED":
                logger.info("Job '%s' completed successfully.", job_id)
                return True
            if status in ["FAILED", "CANCELLED"]:
                reason = job.get("statusReason", "No reason provided.")
                logger.error("Job '%s' %s. Reason: %s", job_id, status.lower(), reason)
                return False
            time.sleep(poll_interval_seconds)
        except ClientError as e:
            logger.error(
                "Error while checking Batch job status for '%s': %s", job_id, e
            )
            return False
    logger.warning(
        "Timed out waiting for job '%s' to complete after %d seconds.",
        job_id,
        timeout_seconds,
    )
    return False

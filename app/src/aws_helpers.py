import json
import re
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import make_msgid
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError, HTTPClientError
from botocore.exceptions import ConnectionError as BotoConnectionError
from bs4 import BeautifulSoup, NavigableString

from app.domain.constants import (
    BATCH_ATTEMPT_TIMEOUT_MINUTES,
    BATCH_ATTEMPTS,
    BATCH_RUNNABLE_LIMIT_MINUTES,
)
from app.logger import logger

DEFAULT_BATCH_POLL_INTERVAL: int = 30
# The longest a submitted job can live: queued in RUNNABLE until the queue cancels
# it, then every attempt running to its timeout. The waiter must outlast that, or
# run_batch.py reports "failed or timed out" for a job that later succeeds. The
# three limits are the ones deploy_infra configures, so they live in constants.
DEFAULT_BATCH_TIMEOUT: int = (
    BATCH_RUNNABLE_LIMIT_MINUTES + BATCH_ATTEMPTS * BATCH_ATTEMPT_TIMEOUT_MINUTES
) * 60 + 300
# Error codes from describe_jobs that a later poll can get past: throttling and
# Batch's own 5xx (ServerException).
_TRANSIENT_BATCH_ERROR_CODES: frozenset[str] = frozenset(
    {"ServerException", "ThrottlingException", "TooManyRequestsException"}
)


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
    except (ClientError, BotoCoreError) as e:
        if isinstance(e, ClientError) and e.response["Error"]["Code"] == "404":
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
    except (ClientError, BotoCoreError) as e:
        logger.error("Failed to download from S3: %s", e)
        return False


def read_s3_json(
    boto_session: boto3.Session, s3_bucket_name: str, s3_key: str
) -> dict[str, Any] | None:
    """A small JSON object from S3, or None when there is no usable answer.

    None covers BOTH "the object does not exist" and "the read failed", because
    every caller so far treats them the same way — and the log line says which it
    was. A caller that needs to distinguish them must not use this.
    """
    if not all([s3_bucket_name, s3_key]):
        logger.error("S3 bucket name and key are required.")
        return None
    try:
        response = boto_session.client("s3").get_object(
            Bucket=s3_bucket_name, Key=s3_key
        )
        payload = json.loads(response["Body"].read().decode("utf-8"))
    except (ClientError, BotoCoreError) as e:
        code = e.response["Error"]["Code"] if isinstance(e, ClientError) else ""
        if code in ("NoSuchKey", "404"):
            logger.info("No object at 's3://%s/%s'.", s3_bucket_name, s3_key)
        else:
            logger.error("Failed to read 's3://%s/%s': %s", s3_bucket_name, s3_key, e)
        return None
    except (ValueError, UnicodeDecodeError) as e:
        logger.error(
            "Object 's3://%s/%s' is not readable JSON: %s", s3_bucket_name, s3_key, e
        )
        return None
    if not isinstance(payload, dict):
        logger.error(
            "Object 's3://%s/%s' is JSON but not an object.", s3_bucket_name, s3_key
        )
        return None
    return payload


def write_s3_json(
    boto_session: boto3.Session,
    s3_bucket_name: str,
    s3_key: str,
    payload: dict[str, Any],
) -> bool:
    """Write a small JSON object to S3, overwriting. False on any failure."""
    if not all([s3_bucket_name, s3_key]):
        logger.error("S3 bucket name and key are required.")
        return False
    try:
        boto_session.client("s3").put_object(
            Bucket=s3_bucket_name,
            Key=s3_key,
            Body=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        return True
    except (ClientError, BotoCoreError) as e:
        logger.error("Failed to write 's3://%s/%s': %s", s3_bucket_name, s3_key, e)
        return False


def get_ssm_param_value(boto_session: boto3.Session, param_name: str) -> str:
    ssm_client = boto_session.client("ssm")
    try:
        response = ssm_client.get_parameter(Name=param_name, WithDecryption=True)
        return response["Parameter"]["Value"]
    except ClientError as e:
        logger.error("Failed to get SSM parameter '%s': %s", param_name, e)
        raise


def suppressed_destinations(boto_session: boto3.Session) -> set[str]:
    """Addresses on the SES account suppression list, lowercased.

    SES adds an address after a hard bounce or a complaint and silently drops any
    later send to it, so sending again only wastes a send slot. An empty set on
    failure: the list is an optimisation, and not reading it must not stop
    delivery.
    """
    client = boto_session.client("sesv2")
    found: set[str] = set()
    kwargs: dict[str, Any] = {}
    try:
        while True:
            page = client.list_suppressed_destinations(**kwargs)
            found.update(
                d["EmailAddress"].strip().lower()
                for d in page.get("SuppressedDestinationSummaries", [])
            )
            if not (token := page.get("NextToken")):
                return found
            kwargs["NextToken"] = token
    except (ClientError, BotoCoreError) as e:
        logger.warning("Could not read the SES suppression list: %s", e)
        return set()


# Elements that start a new line in the plain-text part.
_TEXT_BLOCK_TAGS: tuple[str, ...] = (
    "blockquote",
    "div",
    "figcaption",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "hr",
    "li",
    "p",
    "pre",
    "tr",
)
_HIDDEN_STYLE = re.compile(r"display\s*:\s*none", re.IGNORECASE)


def html_to_text(html: str) -> str:
    """The plain-text alternative to an HTML email.

    One line per block element, a blank line between paragraphs at most, and
    each link's URL after its text so a text-only reader can still open the
    article. Hidden elements (the inbox preheader) are left out.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["head", "style", "script"]):
        tag.decompose()
    for tag in soup.find_all(style=_HIDDEN_STYLE):
        tag.decompose()
    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"])
        if href and href != anchor.get_text(strip=True):
            anchor.append(f" ({href})")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for tag in soup.find_all(_TEXT_BLOCK_TAGS):
        tag.insert_before(NavigableString("\n"))
        tag.append("\n")
    lines = [" ".join(line.split()) for line in soup.get_text().splitlines()]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


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
        # A working opt-out path (RFC 2369). Bulk-sender reputation systems at
        # the major providers expect it on newsletter mail, and without one the
        # only way to stop delivery is to mark the message as spam — which is
        # the outcome that actually damages the sending identity. mailto is the
        # honest mechanism here: the sender mailbox is monitored by the operator,
        # so no fictitious one-click endpoint is advertised.
        msg["List-Unsubscribe"] = f"<mailto:{sender}?subject=unsubscribe>"
    else:
        msg["Message-ID"] = make_msgid()
    for header in ["In-Reply-To", "References"]:
        if header in msg:
            del msg[header]
    # Plain text first: in multipart/alternative the last part is the preferred
    # one. Without a text part, spam filters score the message as HTML-only bulk
    # mail and text-only clients show nothing.
    msg.attach(MIMEText(html_to_text(html_content), "plain", "utf-8"))
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
    except (ClientError, BotoCoreError) as e:
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
    except (ClientError, BotoCoreError) as e:
        logger.error("Failed to upload to S3: %s", e)
        return False


def _is_transient_poll_error(error: ClientError | BotoCoreError) -> bool:
    """Whether a failed ``describe_jobs`` call is worth repeating.

    Throttling, a Batch server fault, a dropped connection or a timed-out read
    can pass; anything else (expired credentials, a denied call) will not.
    """
    if isinstance(error, ClientError):
        code = error.response.get("Error", {}).get("Code", "")
        return code in _TRANSIENT_BATCH_ERROR_CODES
    return isinstance(error, BotoConnectionError | HTTPClientError)


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
            # describe_jobs can return {"jobs": []} right after submit (eventual
            # consistency). That job exists and will run, so keep polling:
            # reporting it as failed invites a re-run that sends the issue twice.
            jobs = response.get("jobs") or []
            status = jobs[0].get("status") if jobs else None
            if not status:
                logger.debug("Job '%s' not visible yet; polling again", job_id)
                time.sleep(poll_interval_seconds)
                continue
            job = jobs[0]
            if status == "SUCCEEDED":
                logger.info("Job '%s' completed successfully.", job_id)
                return True
            if status == "FAILED":
                reason = job.get("statusReason", "No reason provided.")
                logger.error("Job '%s' failed. Reason: %s", job_id, reason)
                return False
            time.sleep(poll_interval_seconds)
        except (ClientError, BotoCoreError) as e:
            # A throttled or dropped status call says nothing about the job, which
            # is still running: keep polling, within the same overall timeout.
            if _is_transient_poll_error(e):
                logger.warning(
                    "Transient error checking Batch job '%s' (%s); polling again.",
                    job_id,
                    e,
                )
                time.sleep(poll_interval_seconds)
                continue
            logger.error(
                "Could not check the status of Batch job '%s' (%s); the job may "
                "still be running.",
                job_id,
                e,
            )
            return False
    logger.warning(
        "Timed out waiting for job '%s' to complete after %d seconds.",
        job_id,
        timeout_seconds,
    )
    return False

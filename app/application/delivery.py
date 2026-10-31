"""Sending the issue: who gets it, and a ledger so a retry never sends twice."""

import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from app.application.ports import Mailer, ObjectStore
from app.configs import Config
from app.domain.alarms import validate_emails
from app.domain.constants import Language, LocalPaths, S3Paths
from app.domain.newsletter import header_title
from app.logger import logger

# SES caps this account at 1 message per second (MaxSendRate, checked
# 2026-10-10; the account is still in the SES sandbox). 0.5s sent at twice
# that and relied on burst tolerance; a throttled send is a reader dropped.
SEND_INTERVAL_SECONDS: Final[float] = 1.1
# Where an explicit --recipients run archives its build, under the newsletters/
# and articles/ prefixes, so a test send never replaces a delivered issue.
TEST_RUN_PREFIX: Final[str] = "tests"
# Where a retry of an already-(partly-)sent issue archives its rebuild.
RETRY_PREFIX: Final[str] = "retries"


def archive_suffix(
    config: Config,
    store: ObjectStore,
    date_suffix: str,
    language: Language,
    recipients: str | None,
) -> str:
    """Where under newsletters/ and articles/ this run's build is archived.

    The archive key is the issue's stage, date and language, and the bucket keeps
    no versions, so only the run that actually delivers the issue may write the
    canonical key. Two other runs build the same issue: an explicit --recipients
    test send, and a retry of a run that had already sent to someone (a Spot
    reclaim or OOM after the send loop began). Either would replace the record of
    what readers received with a rebuild that can carry different posts.
    """
    if recipients is not None:
        return TEST_RUN_PREFIX
    ledger_key = _delivery_ledger_key(config, date_suffix, language)
    if _already_delivered(store, config, ledger_key):
        attempt = os.environ.get("AWS_BATCH_JOB_ATTEMPT") or datetime.now(UTC).strftime(
            "%Y%m%dT%H%M%S"
        )
        logger.warning(
            "This issue was already sent to some readers; archiving this "
            "rebuild under '%s/%s' instead of replacing the delivered copy.",
            RETRY_PREFIX,
            attempt,
        )
        return f"{RETRY_PREFIX}/{attempt}"
    return ""


@dataclass
class DeliveryOutcome:
    """What one run's send phase actually did.

    ``skipped`` distinguishes a retry that had nothing left to do from a run that
    found nobody to send to; both attempt zero sends, and only the second is
    alarm-worthy.
    """

    #: Recipients the run resolved, before the already-delivered ledger.
    resolved: int = 0
    #: Recipients skipped because the ledger says this issue already reached them.
    skipped: int = 0
    #: Recipients skipped because SES suppresses them (hard bounce or complaint).
    suppressed: int = 0
    #: Recipients this run actually tried to send to.
    attempted: int = 0
    succeeded: int = 0
    failed: list[str] = field(default_factory=list)


def process_newsletter_emails(
    newsletter_path: Path,
    date_suffix: str,
    config: Config,
    store: ObjectStore,
    mailer: Mailer,
    language: Language,
    root_dir: Path,
    recipients: list[str] | None = None,
) -> DeliveryOutcome:
    override = recipients is not None
    recipients = _get_recipients(config, store, root_dir, recipients)
    if not recipients:
        logger.warning("No recipients found, skipping email sending.")
        return DeliveryOutcome()
    newsletter_content = _get_newsletter_content(newsletter_path)
    if not newsletter_content:
        logger.error("Failed to read newsletter content, skipping email sending.")
        return DeliveryOutcome(
            resolved=len(recipients), attempted=len(recipients), failed=recipients
        )
    # An explicit --recipients run is the operator addressing a specific mailbox on
    # purpose (a test send, a resend to one reader), so it neither reads nor writes
    # the ledger. The ledger governs the list-driven issue, which is the only thing
    # a retry can duplicate.
    ledger_key = (
        None if override else _delivery_ledger_key(config, date_suffix, language)
    )
    return _send_emails_to_recipients(
        recipients,
        newsletter_content,
        date_suffix,
        config,
        store,
        mailer,
        ledger_key=ledger_key,
    )


def _delivery_ledger_key(config: Config, date_suffix: str, language: Language) -> str:
    """S3 key of the delivery ledger for one issue.

    Keyed by stage, date and language because those are exactly what identify an
    issue — the same three parts the rendered filename uses. Two runs that differ
    in any of them are different issues and must not share a ledger.
    """
    prefix = (config.resources.s3_prefix or "").strip("/")
    name = f"delivered-{config.resources.stage}-{date_suffix}-{language.value}.json"
    return f"{prefix}/{S3Paths.DELIVERIES.value}/{name}".lstrip("/")


def _already_delivered(store: ObjectStore, config: Config, ledger_key: str) -> set[str]:
    """Addresses this issue has already reached, lowercased for comparison.

    Fails OPEN: an unreadable or absent ledger yields an empty set, so the run
    sends to everyone. That is the pre-existing behaviour, and it is the right way
    to be wrong — a duplicate issue is an annoyance, an issue nobody receives is
    the failure this pipeline exists to avoid.
    """
    ledger = store.read_json(ledger_key)
    if not ledger:
        return set()
    delivered = ledger.get("delivered")
    if not isinstance(delivered, list):
        logger.warning(
            "Delivery ledger 's3://%s/%s' has no usable 'delivered' list; "
            "treating this issue as undelivered.",
            config.resources.s3_bucket_name,
            ledger_key,
        )
        return set()
    return {item.strip().lower() for item in delivered if isinstance(item, str)}


def _record_delivery(
    store: ObjectStore,
    config: Config,
    ledger_key: str,
    delivered: list[str],
) -> None:
    """Persist the delivered set. Called after EVERY successful send.

    Once per recipient rather than once per run, because the crash this guards
    against can land in the middle of the loop: a ledger written only at the end
    would be absent exactly when it is needed, and the retry would re-send to
    everyone — which is the bug.
    """
    written = store.write_json(
        ledger_key,
        {
            "stage": config.resources.stage,
            "updated_at": datetime.now(UTC).isoformat(),
            # A snapshot: the caller keeps appending to its list as the loop runs,
            # and a payload that aliases it would not describe this moment.
            "delivered": list(delivered),
        },
    )
    if not written:
        logger.warning(
            "Could not update the delivery ledger 's3://%s/%s'. A retry of this "
            "run may re-send to recipients already served.",
            config.resources.s3_bucket_name,
            ledger_key,
        )


def _get_recipients(
    config: Config,
    store: ObjectStore,
    root_dir: Path,
    recipients: list[str] | None = None,
) -> list[str]:
    if recipients:
        return _unique(validate_emails(recipients))
    filename = recipients_filename(config)
    recipients_path = root_dir / LocalPaths.ASSETS_DIR.value / filename
    if _download_recipients_file(store, config, filename, recipients_path):
        return _unique(
            validate_emails(recipients_path.read_text(encoding="utf-8").splitlines())
        )
    return []


def _unique(recipients: list[str]) -> list[str]:
    """Drop repeated addresses, case-insensitively, keeping the first spelling.

    The ledger is read once before the send loop, so it cannot catch a repeat
    WITHIN one run: a list holding both ``a@x.com`` and ``A@x.com`` sent that
    reader the issue twice.
    """
    seen: set[str] = set()
    unique: list[str] = []
    for address in recipients:
        key = address.strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(address)
    return unique


def recipients_filename(config: Config) -> str:
    base_name, ext = LocalPaths.RECIPIENTS_FILE.value.rsplit(".", 1)
    return f"{base_name}-{config.resources.stage}.{ext}"


def _download_recipients_file(
    store: ObjectStore, config: Config, filename: str, recipients_path: Path
) -> bool:
    s3_key = f"{config.resources.s3_prefix or ''}/{S3Paths.RECIPIENTS.value}/{filename}".lstrip(
        "/"
    )
    return store.download(s3_key, recipients_path)


def _get_newsletter_content(newsletter_path: Path) -> str | None:
    try:
        return newsletter_path.read_text(encoding="utf-8")
    except OSError as e:
        logger.error("Failed to read newsletter file: %s", e)
        return None


def _send_emails_to_recipients(
    recipients: list[str],
    content: str,
    date_suffix: str,
    config: Config,
    store: ObjectStore,
    mailer: Mailer,
    ledger_key: str | None = None,
) -> DeliveryOutcome:
    """Send the issue to each recipient, skipping any the ledger already covers.

    The ledger makes the send loop idempotent PER RECIPIENT, which is the unit the
    harm is measured in. A Batch retry re-runs ``main.py`` from the start, so
    without it a job that died after the first send (most likely a Spot reclaim)
    would deliver the issue again; with it, the retry serves exactly the readers
    not yet reached. An issue-level marker could only say "all" or "none".
    """
    subject = f"[{date_suffix}] {header_title(config.newsletter.header_title)}"
    already = _already_delivered(store, config, ledger_key) if ledger_key else set()
    blocked = mailer.suppressed_destinations()
    outcome = DeliveryOutcome(resolved=len(recipients))
    delivered: list[str] = []

    for recipient in recipients:
        if recipient.strip().lower() in already:
            outcome.skipped += 1
            delivered.append(recipient)
            logger.info(
                "Skipping '%s': this issue was already delivered to them.", recipient
            )
            continue
        if recipient.strip().lower() in blocked:
            outcome.suppressed += 1
            logger.warning(
                "Skipping '%s': on the SES suppression list (bounced or "
                "complained); remove it from the recipients file.",
                recipient,
            )
            continue
        outcome.attempted += 1
        if mailer.send(subject, str(config.newsletter.sender), [recipient], content):
            outcome.succeeded += 1
            delivered.append(recipient)
            if ledger_key:
                _record_delivery(store, config, ledger_key, delivered)
        else:
            outcome.failed.append(recipient)
        # Only after a real SES call — a skipped recipient has nothing to pace.
        time.sleep(SEND_INTERVAL_SECONDS)

    logger.info(
        "Email sending complete: %d/%d successful; of %d resolved recipients, "
        "%d already delivered and %d suppressed by SES were skipped.",
        outcome.succeeded,
        outcome.attempted,
        outcome.resolved,
        outcome.skipped,
        outcome.suppressed,
    )
    return outcome

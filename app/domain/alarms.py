"""The alarm format, and the address check that decides who an issue goes to."""

from datetime import UTC, datetime

from email_validator import EmailNotValidError
from email_validator import validate_email as _validate_email

from app.logger import logger


def format_alarm(
    *,
    event: str,
    status: str,
    fields: dict[str, str],
    project: str,
    timestamp: datetime | None = None,
) -> tuple[str, str]:
    """Build a ``(subject, message)`` pair in the project family's unified alarm
    format, shared verbatim across omnisummary/paper-bridge/scholar-lens:

        Subject: [<project>] <event> — <STATUS>

        <event> <STATUS>

        Key:   Value

        — 2026-06-10 04:12:00 UTC

    ``status`` is a short uppercase state (``FAILED``/``ALERT``). ``fields`` is an
    ordered mapping; single-line values render as an aligned ``Key: Value`` block,
    multi-line values render under their own ``Key:`` header. Omit a row by leaving
    it out of the dict.

    ``project`` is required so the subject names the deployment
    (``resources.project_name``), not a hardcoded default.
    """
    ts = (timestamp or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M:%S")
    subject = f"[{project}] {event} — {status}"

    inline = {k: v for k, v in fields.items() if "\n" not in v}
    block = {k: v for k, v in fields.items() if "\n" in v}

    lines = [f"{event} {status}", ""]
    if inline:
        width = max(len(k) for k in inline)
        lines += [f"{k + ':':<{width + 1}} {v}" for k, v in inline.items()]
    for k, v in block.items():
        lines += ["", f"{k}:", v.strip("\n")]
    lines.append("")
    lines.append(f"— {ts} UTC")

    return subject, "\n".join(lines)


def validate_email(email: str) -> bool:
    """Whether ``email`` is a syntactically valid address.

    Delegates to ``email_validator``, which the project already depends on (it is
    what backs pydantic's ``EmailStr``, used for ``newsletter.sender``). This was
    a hand-written regex, which is both a second definition of "valid address"
    and a weaker one — it rejects tagged locals with unusual characters and any
    internationalized domain, and a recipient it drops is a reader who silently
    stops receiving the digest.

    Deliverability is NOT checked: that would issue an MX lookup per recipient
    from inside the run.
    """
    try:
        _validate_email(email.strip(), check_deliverability=False)
    except EmailNotValidError:
        return False
    return True


def validate_emails(emails: list[str]) -> list[str]:
    # The address is returned as written (stripped), not in the validator's
    # normalized form: SES sends to what the recipients file says, and silently
    # rewriting an address is not this function's job.
    valid_emails = [email.strip() for email in emails if validate_email(email)]
    if len(valid_emails) < len(emails):
        logger.warning(
            "Filtered out %d invalid email addresses", len(emails) - len(valid_emails)
        )
    return valid_emails

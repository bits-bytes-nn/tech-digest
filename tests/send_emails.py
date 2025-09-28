import os
import re
import sys
from pathlib import Path
from typing import Final
import boto3

sys.path.append(str(Path(__file__).parent.parent))
from app.configs import Config
from app.src import EnvVars, Language, LocalPaths, get_date_range, logger, send_email

END_DATE: Final[str | None] = None
RECIPIENTS: Final[list[str] | None] = None
LANGUAGE: Final[Language] = Language.KO

if __name__ == "__main__":
    config = Config.load()
    profile_name = os.environ.get(EnvVars.AWS_PROFILE_NAME.value)
    boto_session = boto3.Session(
        profile_name=profile_name, region_name=config.resources.default_region_name
    )

    _, end_date = get_date_range(END_DATE, config.scraping.days_back)
    date_suffix = end_date.strftime("%Y-%m-%d")
    root_dir = Path(__file__).resolve().parent.parent

    valid_recipients = []
    if RECIPIENTS:
        valid_recipients = RECIPIENTS
    else:
        filename, ext = LocalPaths.RECIPIENTS_FILE.value.split(".", 1)
        filename = f"{filename}-{config.resources.stage}.{ext}"
        recipients_path = root_dir / "assets" / filename
        with open(recipients_path, "r", encoding="utf-8") as file:
            recipients = file.read().splitlines()

        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        valid_recipients = [
            r.strip() for r in recipients if re.match(email_pattern, r.strip())
        ]
        if len(valid_recipients) < len(recipients):
            logger.warning(
                "Filtered out %d invalid email addresses",
                len(recipients) - len(valid_recipients),
            )

    if not valid_recipients:
        logger.warning("No valid recipients found. Exiting.")
        sys.exit(0)

    newsletter_filename = (
        "-".join(["newsletter", config.resources.stage, date_suffix, LANGUAGE.value])
        + ".html"
    )
    newsletter_path = root_dir / LocalPaths.OUTPUTS_DIR.value / newsletter_filename

    try:
        newsletter_content = newsletter_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.error(
            "Newsletter file not found at '%s'. Please build it first.", newsletter_path
        )
        sys.exit(1)

    subject = f"[{date_suffix}] {config.newsletter.header_title}"
    success_count = 0
    for recipient in valid_recipients:
        if send_email(
            boto_session,
            subject,
            str(config.newsletter.sender),
            [recipient],
            newsletter_content,
        ):
            success_count += 1
        else:
            logger.error("Failed to send email to '%s'", recipient)

    logger.info(
        "Email sending complete: %d/%d successful", success_count, len(valid_recipients)
    )

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from pprint import pformat
from typing import Any, Final

import boto3

sys.path.append(str(Path(__file__).parent.parent))
from configs import Config
from src import (
    GMAIL_CLIP_BYTES,
    AppConstants,
    BuildConfiguration,
    CrawlReport,
    EnvVars,
    Greeter,
    Language,
    LocalPaths,
    NewsletterBuilder,
    Post,
    PostCollector,
    S3Paths,
    SSMParams,
    Summarizer,
    SummarizerSettings,
    check_and_download_from_s3,
    format_alarm,
    get_date_range,
    get_ssm_param_value,
    is_running_in_aws,
    logger,
    read_s3_json,
    send_email,
    upload_to_s3,
    validate_emails,
    write_s3_json,
)

SEND_INTERVAL_SECONDS: Final[float] = 0.5
ROOT_DIR: Path = (
    Path("/tmp") if is_running_in_aws() else Path(__file__).resolve().parent.parent
)
DEFAULT_BOTO_SESSION: boto3.Session = boto3.Session(
    region_name=os.environ.get(EnvVars.DEFAULT_REGION_NAME.value)
)
BEDROCK_BOTO_SESSION: boto3.Session = boto3.Session(
    region_name=os.environ.get(EnvVars.BEDROCK_REGION_NAME.value)
)


def handler(event: dict[str, Any], context: Any) -> dict[str, int | str]:
    config = Config.load()
    profile_name = (
        os.environ.get(EnvVars.AWS_PROFILE_NAME.value)
        if is_running_in_aws()
        else config.resources.profile_name
    )
    default_boto_session = (
        DEFAULT_BOTO_SESSION
        if is_running_in_aws()
        else boto3.Session(
            region_name=config.resources.default_region_name, profile_name=profile_name
        )
    )
    bedrock_boto_session = (
        BEDROCK_BOTO_SESSION
        if is_running_in_aws()
        else boto3.Session(
            region_name=config.resources.bedrock_region_name, profile_name=profile_name
        )
    )
    try:
        _setup_aws_env(config, default_boto_session)
        end_date_str = event.get("END_DATE")
        language_str = event.get("LANGUAGE")
        recipients = event.get("RECIPIENTS")
        # Resolve once here so every downstream stage (filtering, greeting,
        # newsletter build/filenames) uses the SAME language. Passing it only to
        # filtering — as before — left the greeting and output filenames stuck on
        # the KO default even when --language en was requested.
        language = Language(language_str) if language_str else Language.KO
        posts, date_suffix, filtered_out_posts, crawl_report = _fetch_and_filter_posts(
            config,
            bedrock_boto_session,
            end_date_str=end_date_str,
            language=language,
        )
        _log_run_summary(crawl_report, posts, filtered_out_posts, language)
        topic_arn = os.environ.get(EnvVars.TOPIC_ARN.value)
        if is_running_in_aws() and topic_arn and crawl_report.failed:
            _send_crawl_health_alert(
                default_boto_session,
                topic_arn,
                config.resources.project_name,
                crawl_report,
                config.scraping.expected_flaky_urls,
            )
        if not posts:
            # Distinguish "nothing survived filtering" from "nothing was
            # collected". The former — especially when every post was dropped by
            # a model error — is a silent failure worth alerting on, since the
            # run still exits successfully and no email goes out.
            if is_running_in_aws() and topic_arn:
                _maybe_send_empty_digest_alert(
                    default_boto_session,
                    topic_arn,
                    config.resources.project_name,
                    crawl_report,
                    filtered_out_posts,
                )
            logger.info("No posts to process. Exiting gracefully.")
            return {"statusCode": 200, "body": "No posts found to process."}
        newsletter_path = _process_posts_and_create_newsletter(
            posts,
            date_suffix,
            config,
            default_boto_session,
            bedrock_boto_session,
            language=language,
        )
        if is_running_in_aws() and topic_arn:
            _maybe_send_clipped_newsletter_alert(
                default_boto_session,
                topic_arn,
                config.resources.project_name,
                newsletter_path,
                len(posts),
            )
        if config.newsletter.send_emails:
            outcome = _process_newsletter_emails(
                newsletter_path,
                date_suffix,
                config,
                default_boto_session,
                language,
                None if recipients is None else recipients.split(","),
            )
            if outcome.skipped and not outcome.attempted:
                # A retry of an issue that was already fully delivered. Not a
                # failure and not alarm-worthy — the guard did its job.
                logger.info(
                    "All %d recipient(s) had already received this issue; "
                    "nothing was sent. This run is a retry of a completed send.",
                    outcome.skipped,
                )
            if is_running_in_aws() and topic_arn:
                if outcome.resolved == 0:
                    # No recipients could be RESOLVED at all — keyed on the
                    # resolved count, not on how many were attempted, so a retry
                    # that legitimately skipped everyone does not read as "nobody
                    # to send to". The partial-delivery alert below cannot catch
                    # this either: it keys on failed recipients, and a run with no
                    # recipients has none. So an unreadable or missing recipients
                    # file in S3 meant the issue was built, uploaded and silently
                    # never sent, with the run still returning 200.
                    _send_no_recipients_alert(
                        default_boto_session, topic_arn, config, crawl_report
                    )
                _maybe_send_partial_delivery_alert(
                    default_boto_session,
                    topic_arn,
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
            _send_failure_notification(
                default_boto_session, topic_arn, config.resources.project_name, str(e)
            )
        return {"statusCode": 500, "body": f"An error occurred: {e}"}


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


def _fetch_and_filter_posts(
    config: Config,
    boto_session: boto3.Session,
    end_date_str: str | None = None,
    language: Language = Language.KO,
) -> tuple[list[Post], str, list[tuple[Post, str]], CrawlReport]:
    start_date, end_date = get_date_range(end_date_str, config.scraping.days_back)
    logger.info("Fetching posts from '%s' to '%s'", start_date.date(), end_date.date())

    collector = PostCollector.from_urls(config.scraping.rss_urls)
    posts = collector.collect_posts(start_date, end_date)
    crawl_report = collector.report

    date_suffix = end_date.strftime("%Y-%m-%d")
    if not posts:
        logger.warning("No posts found in the specified date range")
        return [], date_suffix, [], crawl_report

    logger.info("Found %d posts", len(posts))
    logger.info("Posts: %s", [post.title for post in posts])
    logger.debug(pformat(posts))

    # main is the composition root, so config -> settings translation lives here
    # rather than src importing configs (which would invert the dependency).
    # Field names match the `summarization` config section, so the whole section
    # maps across without restating every key; extras are ignored by pydantic.
    summarizer = Summarizer(
        boto_session,
        SummarizerSettings.model_validate(
            config.summarization.model_dump()
            | {
                "min_content_length": config.scraping.min_content_length,
                "language": language,
            }
        ),
    )

    # The score-rank and max_posts cap are applied inside process_posts BEFORE
    # summarization, so we never pay to summarize posts that get discarded.
    filtered_posts = summarizer.process_posts(posts)

    logger.info("Successfully summarized %d posts", len(filtered_posts))
    logger.info("Filtered posts: %s", [post.title for post in filtered_posts])
    logger.debug(pformat(filtered_posts))

    return filtered_posts, date_suffix, summarizer.filtered_out_posts, crawl_report


def _process_posts_and_create_newsletter(
    posts: list[Post],
    date_suffix: str,
    config: Config,
    default_boto_session: boto3.Session,
    bedrock_boto_session: boto3.Session,
    language: Language = Language.KO,
) -> Path:
    inputs_dir = _prepare_and_save_posts(posts, date_suffix)
    first_section_intro = _generate_greeting(
        posts, date_suffix, config, bedrock_boto_session, language
    )
    outputs_dir = ROOT_DIR / LocalPaths.OUTPUTS_DIR.value
    outputs_dir.mkdir(parents=True, exist_ok=True)
    newsletter_path, article_filenames = _build_newsletter(
        inputs_dir, outputs_dir, date_suffix, first_section_intro, config, language
    )
    _upload_files_to_s3(
        newsletter_path, S3Paths.NEWSLETTERS.value, config, default_boto_session
    )
    if article_filenames:
        article_paths = [
            outputs_dir / LocalPaths.ARTICLES_DIR.value / filename
            for filename in article_filenames
        ]
        _upload_files_to_s3(
            article_paths, S3Paths.ARTICLES.value, config, default_boto_session
        )
    return newsletter_path


def _log_run_summary(
    crawl_report: CrawlReport,
    posts: list[Post],
    filtered_out_posts: list[tuple[Post, str]],
    language: Language,
) -> None:
    """Emit the run's funnel as one JSON line.

    CloudWatch Logs Insights can then chart collection/pass rates over time from
    a single queryable record, instead of requiring someone to reconstruct the
    funnel by reading a few hundred prose log lines per run.
    """
    summary = {
        "event": "run_summary",
        "language": language.value,
        "sources_ok": len(crawl_report.ok),
        "sources_empty": len(crawl_report.empty),
        "sources_failed": len(crawl_report.failed),
        "posts_collected": crawl_report.total_posts,
        "posts_filtered_out": len(filtered_out_posts),
        "posts_in_digest": len(posts),
        "digest": [
            {"source": p.source, "score": round(p.score, 2), "title": p.title}
            for p in posts
        ],
    }
    logger.info("RUN_SUMMARY %s", json.dumps(summary, ensure_ascii=False))


def _prepare_and_save_posts(posts: list[Post], date_suffix: str) -> Path:
    inputs_dir = ROOT_DIR / LocalPaths.INPUTS_DIR.value / date_suffix
    if inputs_dir.exists():
        shutil.rmtree(inputs_dir)
    inputs_dir.mkdir(parents=True)
    for post in posts:
        filename = _generate_post_filename(post)
        article_path = inputs_dir / filename
        _save_post_to_file(post, article_path)
    return inputs_dir


def _generate_post_filename(post: Post) -> str:
    sanitized_title = re.sub(r"[^\w\s-]", "", post.title.lower()).replace(" ", "_")
    # Append a short stable hash of the link so two distinct posts (different
    # URLs) that share a source and sanitize to the same title string don't
    # overwrite each other on disk — which would silently drop one from the
    # digest after we already paid to summarize it. The link is unique per post
    # (the collector dedupes on it), so the hash disambiguates deterministically.
    link_hash = hashlib.sha1(post.link.encode("utf-8")).hexdigest()[:8]
    return f"{post.source}-{sanitized_title}-{link_hash}.json"


def _save_post_to_file(post: Post, path: Path) -> None:
    post_dict = post.model_dump(mode="json")
    json_str = json.dumps(post_dict, ensure_ascii=False, indent=2, default=str)
    path.write_text(json_str, encoding="utf-8")


def _generate_greeting(
    posts: list[Post],
    date_suffix: str,
    config: Config,
    boto_session: boto3.Session,
    language: Language,
) -> str:
    greeter = Greeter(boto_session, config.summarization.greeting_model_id, language)
    titles = [post.title for post in posts]
    context = f"Today is {date_suffix}. Today's articles include: {', '.join(titles)}"
    greeting = greeter.greet(context=context)
    logger.debug("Generated greeting: %s", greeting)
    _save_greeting_to_file(greeting, date_suffix)
    return greeting


def _save_greeting_to_file(greeting: str, date_suffix: str) -> None:
    greeting_path = (
        ROOT_DIR
        / LocalPaths.INPUTS_DIR.value
        / date_suffix
        / LocalPaths.GREETING_FILE.value
    )
    greeting_path.write_text(greeting, encoding="utf-8")


def _build_newsletter(
    inputs_dir: Path,
    outputs_dir: Path,
    date_suffix: str,
    first_section_intro: str,
    config: Config,
    language: Language,
) -> tuple[Path, list[str]]:
    builder = NewsletterBuilder(
        inputs_dir.parent,
        outputs_dir,
        Path(__file__).resolve().parent / LocalPaths.TEMPLATES_DIR.value,
        logos=config.newsletter.logos,
    )
    # Only pass the chrome fields the config actually sets, so an unset one falls
    # through to BuildConfiguration's default. Coercing None to "" (as this used
    # to) overrode those defaults with the empty string, which made every entry in
    # NewsletterConfig.DEFAULT_STYLES unreachable — and, for the one field the
    # renderer requires to be non-empty, turned "no header_title configured" into
    # a Header validation error that fails the whole build.
    chrome = {
        "header_title": config.newsletter.header_title,
        "header_description": config.newsletter.header_description,
        "header_thumbnail": config.newsletter.header_thumbnail,
        "first_section_intro": first_section_intro,
        "footer_title": config.newsletter.footer_title,
    }
    build_config = BuildConfiguration(
        stage=config.resources.stage,
        date_suffix=date_suffix,
        language=language,
        save_individual_articles=config.newsletter.save_articles,
        convert_to_images=config.newsletter.convert_to_images,
        **{key: value for key, value in chrome.items() if value},
    )
    newsletter_path, article_filenames = builder.build(build_config)
    logger.info("Newsletter created: '%s'", newsletter_path)
    return newsletter_path, article_filenames


def _upload_files_to_s3(
    local_paths: Path | list[Path],
    s3_prefix: str,
    config: Config,
    boto_session: boto3.Session,
) -> None:
    s3_destination = (
        f"{config.resources.s3_prefix}/{s3_prefix}"
        if config.resources.s3_prefix
        else s3_prefix
    )
    paths_to_upload = [local_paths] if isinstance(local_paths, Path) else local_paths
    for path in paths_to_upload:
        if path.exists():
            upload_to_s3(
                boto_session,
                path,
                config.resources.s3_bucket_name,
                s3_prefix=s3_destination,
            )
        else:
            logger.warning("File not found, skipping upload: '%s'", path)


@dataclass
class DeliveryOutcome:
    """What one run's send phase actually did.

    ``skipped`` is what distinguishes a retry that had nothing left to do from a
    run that found nobody to send to — the two used to be indistinguishable at the
    call site (both produced ``total == 0``), and only one of them is alarm-worthy.
    """

    #: Recipients the run resolved, before the already-delivered ledger.
    resolved: int = 0
    #: Recipients skipped because the ledger says this issue already reached them.
    skipped: int = 0
    #: Recipients this run actually tried to send to.
    attempted: int = 0
    succeeded: int = 0
    failed: list[str] = field(default_factory=list)


def _process_newsletter_emails(
    newsletter_path: Path,
    date_suffix: str,
    config: Config,
    boto_session: boto3.Session,
    language: Language,
    recipients: list[str] | None = None,
) -> DeliveryOutcome:
    override = recipients is not None
    recipients = _get_recipients(config, boto_session, recipients)
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
        boto_session,
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


def _already_delivered(
    boto_session: boto3.Session, config: Config, ledger_key: str
) -> set[str]:
    """Addresses this issue has already reached, lowercased for comparison.

    Fails OPEN: an unreadable or absent ledger yields an empty set, so the run
    sends to everyone. That is the pre-existing behaviour, and it is the right way
    to be wrong — a duplicate issue is an annoyance, an issue nobody receives is
    the failure this pipeline exists to avoid.
    """
    ledger = read_s3_json(boto_session, config.resources.s3_bucket_name, ledger_key)
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
    boto_session: boto3.Session,
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
    written = write_s3_json(
        boto_session,
        config.resources.s3_bucket_name,
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
    config: Config, boto_session: boto3.Session, recipients: list[str] | None = None
) -> list[str]:
    if recipients:
        return validate_emails(recipients)
    filename = _generate_recipients_filename(config)
    recipients_path = ROOT_DIR / LocalPaths.ASSETS_DIR.value / filename
    if _download_recipients_file(boto_session, config, filename, recipients_path):
        return validate_emails(recipients_path.read_text(encoding="utf-8").splitlines())
    return []


def _generate_recipients_filename(config: Config) -> str:
    base_name, ext = LocalPaths.RECIPIENTS_FILE.value.rsplit(".", 1)
    return f"{base_name}-{config.resources.stage}.{ext}"


def _download_recipients_file(
    boto_session: boto3.Session, config: Config, filename: str, recipients_path: Path
) -> bool:
    s3_key = f"{config.resources.s3_prefix or ''}/{S3Paths.RECIPIENTS.value}/{filename}".lstrip(
        "/"
    )
    return check_and_download_from_s3(
        boto_session, config.resources.s3_bucket_name, s3_key, recipients_path
    )


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
    boto_session: boto3.Session,
    ledger_key: str | None = None,
) -> DeliveryOutcome:
    """Send the issue to each recipient, skipping any the ledger already covers.

    The ledger makes the send loop idempotent PER RECIPIENT, which is the unit the
    harm is measured in. Both retry paths in this pipeline (the Batch job
    definition's ``retry_attempts=2`` and the EventBridge target's) re-run
    ``main.py`` from the start, so a job that died anywhere after the first send —
    a Spot reclaim being the likeliest cause now that Spot is the preferred compute
    environment — used to deliver the whole issue a second time.

    Recipient-level granularity also makes a mid-loop crash strictly better than
    before rather than merely not-worse: the retry serves exactly the readers who
    had not been reached, instead of everyone (duplicate) or nobody (an
    issue-level marker would have claimed the whole issue was done).
    """
    subject = f"[{date_suffix}] {config.newsletter.header_title}"
    already = (
        _already_delivered(boto_session, config, ledger_key) if ledger_key else set()
    )
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
        outcome.attempted += 1
        if send_email(
            boto_session, subject, str(config.newsletter.sender), [recipient], content
        ):
            outcome.succeeded += 1
            delivered.append(recipient)
            if ledger_key:
                _record_delivery(boto_session, config, ledger_key, delivered)
        else:
            outcome.failed.append(recipient)
        # Only after a real SES call — a skipped recipient has nothing to pace.
        time.sleep(SEND_INTERVAL_SECONDS)

    if outcome.skipped:
        logger.info(
            "Email sending complete: %d/%d successful, %d already delivered "
            "(skipped) of %d resolved recipients.",
            outcome.succeeded,
            outcome.attempted,
            outcome.skipped,
            outcome.resolved,
        )
    else:
        logger.info(
            "Email sending complete: %d/%d successful",
            outcome.succeeded,
            outcome.attempted,
        )
    return outcome


def _publish_alarm(
    boto_session: boto3.Session,
    topic_arn: str,
    *,
    project: str,
    event: str,
    status: str,
    fields: dict[str, str],
) -> None:
    """Format and publish one alarm to SNS.

    Single place where the project's alarm format meets the SNS client; the five
    alert paths below differ only in their event name and fields. ``project`` is
    threaded from config rather than left to a default in ``format_alarm``, which
    is how every alarm subject came to read "[tech-digest]" whatever the
    deployment was actually called.
    """
    subject, message = format_alarm(
        event=event, status=status, fields=fields, project=project
    )
    boto_session.client("sns").publish(
        TopicArn=topic_arn, Subject=subject, Message=message
    )


def _maybe_send_partial_delivery_alert(
    boto_session: boto3.Session,
    topic_arn: str,
    project: str,
    success_count: int,
    total_recipients: int,
    failed_recipients: list[str],
    crawl_report: CrawlReport | None = None,
) -> None:
    """Alert when delivery completed but some recipients failed. A fully
    successful run is not alarm-worthy, so nothing is sent in that case."""
    if not failed_recipients:
        return
    fields = {
        "Delivered": f"{success_count}/{total_recipients}",
        "Failed recipients": ", ".join(failed_recipients),
    }
    if crawl_report is not None:
        fields["Crawl health"] = crawl_report.summary_line()
    _publish_alarm(
        boto_session,
        topic_arn,
        project=project,
        event="Newsletter Delivery",
        status="ALERT",
        fields=fields,
    )


def _maybe_send_empty_digest_alert(
    boto_session: boto3.Session,
    topic_arn: str,
    project: str,
    crawl_report: CrawlReport,
    filtered_out_posts: list[tuple[Post, str]],
) -> None:
    """Alert when posts were collected but none survived filtering, so the run
    produced an empty digest and sent no email. A genuinely empty crawl (nothing
    collected at all) is not alarm-worthy here — crawl-health alerts cover source
    failures. The reason breakdown lets the reader tell a mass model failure
    (e.g. every call rejected) from an unusually strict but healthy week."""
    if crawl_report.total_posts == 0 or not filtered_out_posts:
        return
    reason_counts: dict[str, int] = {}
    for _post, reason in filtered_out_posts:
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    breakdown = "\n".join(
        f"  - {count}x {reason}"
        for reason, count in sorted(
            reason_counts.items(), key=lambda kv: kv[1], reverse=True
        )
    )
    _publish_alarm(
        boto_session,
        topic_arn,
        project=project,
        event="Empty Digest",
        status="ALERT",
        fields={
            "Collected": str(crawl_report.total_posts),
            "Survived filtering": "0",
            "Filtered out": str(len(filtered_out_posts)),
            "Reasons": breakdown,
        },
    )


def _send_no_recipients_alert(
    boto_session: boto3.Session,
    topic_arn: str,
    config: Config,
    crawl_report: CrawlReport,
) -> None:
    """Alert when a built issue had nobody to send to.

    This is the pipeline's worst silent failure: the crawl worked, the summaries
    cost real money, the file went to S3 — and every status says success while no
    reader receives anything. The cause is always outside the code (the
    per-stage recipients object missing from S3, empty, or holding no address
    that passes validation), so the alarm names where to look.
    """
    _publish_alarm(
        boto_session,
        topic_arn,
        project=config.resources.project_name,
        event="No Recipients",
        status="ALERT",
        fields={
            "Delivered": "0/0",
            "Expected object": (
                f"s3://{config.resources.s3_bucket_name}/"
                f"{(config.resources.s3_prefix or '').strip('/')}/"
                f"{S3Paths.RECIPIENTS.value}/{_generate_recipients_filename(config)}"
            ),
            "Crawl health": crawl_report.summary_line(),
        },
    )


def _maybe_send_clipped_newsletter_alert(
    boto_session: boto3.Session,
    topic_arn: str,
    project: str,
    newsletter_path: Path,
    article_count: int,
) -> None:
    """Alert when the built issue exceeds the mail-client clip threshold.

    The renderer already logs this, but a log line on an unattended weekly job is
    seen by nobody, and the failure is invisible from the outside: the mail sends
    fine and simply stops rendering partway, so the last articles are replaced by
    a "View entire message" link. Measured on real issues, the margin is real but
    not generous — 2026-06-02 shipped at 101.4% of the limit and was clipped,
    2026-07-11 at 91.5% — so this is worth paging on rather than trusting.
    """
    try:
        size = newsletter_path.stat().st_size
    except OSError as e:
        logger.warning("Could not size the newsletter for the clip check: %s", e)
        return
    if size <= GMAIL_CLIP_BYTES:
        logger.info(
            "Newsletter is %.1f KB, %.0f%% of the %.0f KB clip limit.",
            size / 1024,
            100 * size / GMAIL_CLIP_BYTES,
            GMAIL_CLIP_BYTES / 1024,
        )
        return
    _publish_alarm(
        boto_session,
        topic_arn,
        project=project,
        event="Newsletter Clipped",
        status="ALERT",
        fields={
            "Size": f"{size / 1024:.1f} KB",
            "Limit": f"{GMAIL_CLIP_BYTES / 1024:.0f} KB",
            "Articles": str(article_count),
            "Effect": (
                "Mail clients truncate the message; the last articles are hidden "
                "behind a 'View entire message' link. Lower summarization."
                "max_posts or shorten the length budget."
            ),
        },
    )


def _send_crawl_health_alert(
    boto_session: boto3.Session,
    topic_arn: str,
    project: str,
    crawl_report: CrawlReport,
    expected_flaky: list[str],
) -> None:
    """Notify when a crawl source failed, so a broken source is noticed promptly
    instead of silently dropping out of the digest.

    Sources listed in ``scraping.expected_flaky_urls`` are excluded from the
    trigger (they still appear in the report body). Some sources — notably x.ai
    and occasionally ai.meta.com — reject AWS datacenter IPs every single week,
    which fired this alert on every run. An alarm that always fires is an alarm
    nobody reads, so a *newly* broken source has to be distinguishable from a
    known-broken one.
    """
    unexpected = [
        s
        for s in crawl_report.failed
        if not any(pattern and pattern in s.url for pattern in expected_flaky)
    ]
    if not unexpected:
        if crawl_report.failed:
            logger.info(
                "All %d failing source(s) are configured as expected-flaky; "
                "no crawl-health alert sent. Failing: %s",
                len(crawl_report.failed),
                [s.url for s in crawl_report.failed],
            )
        return
    _publish_alarm(
        boto_session,
        topic_arn,
        project=project,
        event="Crawl Health",
        status="ALERT",
        fields={
            "Unexpected failures": str(len(unexpected)),
            "Summary": crawl_report.summary_line(),
            "Detail": crawl_report.format_alert(),
        },
    )


def _send_failure_notification(
    boto_session: boto3.Session, topic_arn: str, project: str, error_message: str
) -> None:
    _publish_alarm(
        boto_session,
        topic_arn,
        project=project,
        event="Newsletter Delivery",
        status="FAILED",
        fields={"Error": error_message},
    )


if __name__ == "__main__":
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
    args = parser.parse_args()
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
    handler(event, None)

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from pprint import pformat
from typing import Any, Final

import boto3

sys.path.append(str(Path(__file__).parent.parent))
from configs import Config
from src import (
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
    check_and_download_from_s3,
    format_alarm,
    get_date_range,
    get_ssm_param_value,
    is_running_in_aws,
    logger,
    send_email,
    upload_to_s3,
    validate_emails,
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
        topic_arn = os.environ.get(EnvVars.TOPIC_ARN.value)
        if is_running_in_aws() and topic_arn and crawl_report.failed:
            _send_crawl_health_alert(default_boto_session, topic_arn, crawl_report)
        if not posts:
            # Distinguish "nothing survived filtering" from "nothing was
            # collected". The former — especially when every post was dropped by
            # a model error — is a silent failure worth alerting on, since the
            # run still exits successfully and no email goes out.
            if is_running_in_aws() and topic_arn:
                _maybe_send_empty_digest_alert(
                    default_boto_session,
                    topic_arn,
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
        if config.newsletter.send_emails:
            success, failed, total = _process_newsletter_emails(
                newsletter_path,
                date_suffix,
                config,
                default_boto_session,
                None if recipients is None else recipients.split(","),
            )
            if is_running_in_aws() and topic_arn:
                _maybe_send_partial_delivery_alert(
                    default_boto_session,
                    topic_arn,
                    success,
                    total,
                    failed,
                    crawl_report,
                )
        return {"statusCode": 200, "body": "Newsletter processed successfully"}
    except Exception as e:
        logger.error("An error occurred: %s", e, exc_info=True)
        topic_arn = os.environ.get(EnvVars.TOPIC_ARN.value)
        if is_running_in_aws() and topic_arn:
            _send_failure_notification(default_boto_session, topic_arn, str(e))
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

    summarizer = Summarizer(
        boto_session,
        config.summarization.filtering_model_id,
        config.summarization.summarization_model_id,
        filtering_criteria=config.summarization.filtering_criteria,
        filtering_enable_thinking=config.summarization.filtering_enable_thinking,
        summarization_enable_thinking=config.summarization.summarization_enable_thinking,
        language=language,
        max_concurrency=config.summarization.max_concurrency,
        min_score=config.summarization.min_score,
        min_content_length=config.scraping.min_content_length,
        filtering_thinking_budget_tokens=(
            config.summarization.filtering_thinking_budget_tokens
        ),
        summarization_thinking_budget_tokens=(
            config.summarization.summarization_thinking_budget_tokens
        ),
        summarization_max_tokens=config.summarization.summarization_max_tokens,
        thinking_effort=config.summarization.thinking_effort,
    )

    # The score-rank and max_posts cap are applied inside process_posts BEFORE
    # summarization, so we never pay to summarize posts that get discarded.
    filtered_posts = summarizer.process_posts(
        posts,
        config.summarization.use_filtering,
        config.summarization.included_topics,
        config.summarization.excluded_topics,
        max_posts=config.summarization.max_posts,
    )

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
    build_config = BuildConfiguration(
        stage=config.resources.stage,
        date_suffix=date_suffix,
        language=language,
        header_title=config.newsletter.header_title or "",
        header_description=config.newsletter.header_description or "",
        header_thumbnail=config.newsletter.header_thumbnail or "",
        first_section_intro=first_section_intro,
        footer_title=config.newsletter.footer_title or "",
        save_individual_articles=config.newsletter.save_articles,
        convert_to_images=config.newsletter.convert_to_images,
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


def _process_newsletter_emails(
    newsletter_path: Path,
    date_suffix: str,
    config: Config,
    boto_session: boto3.Session,
    recipients: list[str] | None = None,
) -> tuple[int, list[str], int]:
    recipients = _get_recipients(config, boto_session, recipients)
    if not recipients:
        logger.warning("No recipients found, skipping email sending.")
        return 0, [], 0
    newsletter_content = _get_newsletter_content(newsletter_path)
    if not newsletter_content:
        logger.error("Failed to read newsletter content, skipping email sending.")
        return 0, recipients, len(recipients)
    success_count, failed_recipients = _send_emails_to_recipients(
        recipients, newsletter_content, date_suffix, config, boto_session
    )
    return success_count, failed_recipients, len(recipients)


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
) -> tuple[int, list[str]]:
    subject = f"[{date_suffix}] {config.newsletter.header_title}"
    success_count = 0
    failed_recipients = []

    for recipient in recipients:
        if send_email(
            boto_session, subject, str(config.newsletter.sender), [recipient], content
        ):
            success_count += 1
        else:
            failed_recipients.append(recipient)

        time.sleep(SEND_INTERVAL_SECONDS)

    logger.info(
        "Email sending complete: %d/%d successful", success_count, len(recipients)
    )
    return success_count, failed_recipients


def _maybe_send_partial_delivery_alert(
    boto_session: boto3.Session,
    topic_arn: str,
    success_count: int,
    total_recipients: int,
    failed_recipients: list[str],
    crawl_report: CrawlReport | None = None,
) -> None:
    """Alert when delivery completed but some recipients failed. A fully
    successful run is not alarm-worthy, so nothing is sent in that case."""
    if not failed_recipients:
        return
    sns_client = boto_session.client("sns")
    fields = {
        "Delivered": f"{success_count}/{total_recipients}",
        "Failed recipients": ", ".join(failed_recipients),
    }
    if crawl_report is not None:
        fields["Crawl health"] = crawl_report.summary_line()
    subject, message = format_alarm(
        event="Newsletter Delivery", status="ALERT", fields=fields
    )
    sns_client.publish(TopicArn=topic_arn, Subject=subject, Message=message)


def _maybe_send_empty_digest_alert(
    boto_session: boto3.Session,
    topic_arn: str,
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
    sns_client = boto_session.client("sns")
    subject, message = format_alarm(
        event="Empty Digest",
        status="ALERT",
        fields={
            "Collected": str(crawl_report.total_posts),
            "Survived filtering": "0",
            "Filtered out": str(len(filtered_out_posts)),
            "Reasons": breakdown,
        },
    )
    sns_client.publish(TopicArn=topic_arn, Subject=subject, Message=message)


def _send_crawl_health_alert(
    boto_session: boto3.Session, topic_arn: str, crawl_report: CrawlReport
) -> None:
    """Notify when one or more crawl sources failed, so a broken source is
    noticed promptly instead of silently dropping out of the digest."""
    sns_client = boto_session.client("sns")
    subject, message = format_alarm(
        event="Crawl Health",
        status="ALERT",
        fields={
            "Failing sources": str(len(crawl_report.failed)),
            "Summary": crawl_report.summary_line(),
            "Detail": crawl_report.format_alert(),
        },
    )
    sns_client.publish(TopicArn=topic_arn, Subject=subject, Message=message)


def _send_failure_notification(
    boto_session: boto3.Session, topic_arn: str, error_message: str
) -> None:
    sns_client = boto_session.client("sns")
    subject, message = format_alarm(
        event="Newsletter Delivery",
        status="FAILED",
        fields={"Error": error_message},
    )
    sns_client.publish(TopicArn=topic_arn, Subject=subject, Message=message)


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

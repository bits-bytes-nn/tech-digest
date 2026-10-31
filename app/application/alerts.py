"""The operator alarms a run raises, and what triggers each one."""

from pathlib import Path

from app.application.delivery import DeliveryOutcome, recipients_filename
from app.application.ports import AlarmPublisher
from app.configs import Config
from app.domain.alarms import format_alarm
from app.domain.constants import S3Paths
from app.domain.crawl import CrawlReport, SourceHealth
from app.domain.newsletter import GMAIL_CLIP_BYTES
from app.domain.post import Post
from app.domain.summary import MODEL_FAILURE_PREFIXES
from app.logger import logger


def _publish_alarm(
    alarms: AlarmPublisher,
    *,
    project: str,
    event: str,
    status: str,
    fields: dict[str, str],
) -> None:
    """Format and publish one alarm.

    Single place where the project's alarm format meets the publisher; the six
    alert paths below differ only in their event name and fields. ``project`` is
    threaded from config rather than left to a default in ``format_alarm``, which
    is how every alarm subject came to read "[tech-digest]" whatever the
    deployment was actually called.
    """
    subject, message = format_alarm(
        event=event, status=status, fields=fields, project=project
    )
    # An alert is a side channel. Several are published BEFORE the send loop
    # (crawl health, clipped newsletter), so a publish error must not escape and
    # abort the run with the issue built but unsent.
    try:
        alarms.publish(subject, message)
    except Exception as e:
        logger.error("Could not publish the '%s %s' alert: %s", event, status, e)


def maybe_send_partial_delivery_alert(
    alarms: AlarmPublisher,
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
        alarms,
        project=project,
        event="Newsletter Delivery",
        status="ALERT",
        fields=fields,
    )


def maybe_send_model_failure_alert(
    alarms: AlarmPublisher,
    project: str,
    posts: list[Post],
    filtered_out_posts: list[tuple[Post, str]],
) -> None:
    """Alert when an issue ships while some posts were lost to failed model calls.

    The empty-digest alert only fires when NOTHING survives, so a run that lost
    the top post to a failed summary, or a third of the filter calls to errors,
    sent a thinner issue with exit code 0 and nothing but a log line. Failed
    summaries are backfilled from the next-ranked posts; this says it happened.
    """
    failures = [
        (post, reason)
        for post, reason in filtered_out_posts
        if reason.startswith(MODEL_FAILURE_PREFIXES)
    ]
    if not failures:
        return
    _publish_alarm(
        alarms,
        project=project,
        event="Model Failures",
        status="WARN",
        fields={
            "In digest": str(len(posts)),
            "Failed calls": str(len(failures)),
            "Posts": "\n".join(f"  - {p.title}: {r}" for p, r in failures[:10]),
        },
    )


def maybe_send_empty_digest_alert(
    alarms: AlarmPublisher,
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
        alarms,
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


def send_no_recipients_alert(
    alarms: AlarmPublisher,
    config: Config,
    crawl_report: CrawlReport,
    outcome: DeliveryOutcome,
) -> None:
    """Alert when a built issue had nobody to send to.

    This is the pipeline's worst silent failure: the crawl worked, the summaries
    cost real money, the file went to S3 — and every status says success while no
    reader receives anything. The cause is always outside the code (the
    per-stage recipients object missing from S3, empty, holding no address that
    passes validation, or listing only addresses SES suppresses), so the alarm
    names where to look.
    """
    fields = {
        # Recipients an earlier attempt already served are not owed this send.
        "Delivered": f"0/{outcome.resolved - outcome.skipped}",
        "Expected object": (
            f"s3://{config.resources.s3_bucket_name}/"
            f"{(config.resources.s3_prefix or '').strip('/')}/"
            f"{S3Paths.RECIPIENTS.value}/{recipients_filename(config)}"
        ),
    }
    if outcome.suppressed:
        fields["Suppressed by SES"] = (
            f"{outcome.suppressed} (bounced or complained; remove them from the list)"
        )
    fields["Crawl health"] = crawl_report.summary_line()
    _publish_alarm(
        alarms,
        project=config.resources.project_name,
        event="No Recipients",
        status="ALERT",
        fields=fields,
    )


def maybe_send_clipped_newsletter_alert(
    alarms: AlarmPublisher,
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
        alarms,
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


def send_crawl_health_alert(
    alarms: AlarmPublisher,
    project: str,
    crawl_report: CrawlReport,
    expected_flaky: list[str],
) -> None:
    """Notify when a crawl source failed, produced nothing to filter, or left
    every post without its article page.

    Two triggers, and the second one is the point. A FAILED source is loud — the
    fetch itself errored. A source that fetched fine and offered ZERO items is just
    as broken (a moved feed, a scraper whose selector stopped matching) but would
    read as one more EMPTY source among many. ``CrawlReport.likely_broken`` separates it
    from a blog that was merely quiet, using evidence from the same run rather than
    cross-run state.

    A quiet blog never triggers this: it offers its back catalogue and the date
    window filters it out, so its candidate count is non-zero. That matters more
    than the coverage — an alarm that fires on the sources that are empty in a
    normal week is an alarm nobody reads, which is the same failure ``expected_flaky_urls`` exists to
    prevent.

    Sources listed in ``scraping.expected_flaky_urls`` are excluded from BOTH
    triggers (they still appear in the report body): a source that fails every
    week by design would make an alarm that always fires, hiding the one that
    means something.
    """

    def unexpected(sources: list[SourceHealth]) -> list[SourceHealth]:
        return [
            s
            for s in sources
            if not any(pattern and pattern in s.url for pattern in expected_flaky)
        ]

    failed = unexpected(crawl_report.failed)
    broken = unexpected(crawl_report.likely_broken)
    unreadable = unexpected(crawl_report.pages_unreadable)
    if not failed and not broken and not unreadable:
        suppressed = (
            crawl_report.failed
            + crawl_report.likely_broken
            + crawl_report.pages_unreadable
        )
        if suppressed:
            logger.info(
                "All %d unhealthy source(s) are configured as expected-flaky; "
                "no crawl-health alert sent. Suppressed: %s",
                len(suppressed),
                [s.url for s in suppressed],
            )
        return
    fields = {
        "Unexpected failures": str(len(failed)),
        # Named for what it is rather than "empty": the operator's next action is
        # to check the feed URL or the scraper's selector, not to wait a week.
        "Sources offering nothing": str(len(broken)),
        # The feed is fine but its posts never reach the summarizer: check
        # whether the site now blocks the crawler (a 403 or a bot challenge).
        "Unreadable article pages": str(len(unreadable)),
        "Summary": crawl_report.summary_line(),
        "Detail": crawl_report.format_alert(),
    }
    if broken or unreadable:
        fields["Check these"] = ", ".join(s.url for s in broken + unreadable)
    _publish_alarm(
        alarms,
        project=project,
        event="Crawl Health",
        status="ALERT",
        fields=fields,
    )


def send_failure_notification(
    alarms: AlarmPublisher, project: str, error_message: str
) -> None:
    _publish_alarm(
        alarms,
        project=project,
        event="Newsletter Delivery",
        status="FAILED",
        fields={"Error": error_message},
    )

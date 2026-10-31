"""One issue end to end: crawl, filter and summarize, then build and archive."""

import hashlib
import json
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from pprint import pformat
from typing import Final

from bs4 import BeautifulSoup

from app.application.greet import Greeter
from app.application.ports import (
    ChatModelProvider,
    ImageProbe,
    NewsletterBuilderPort,
    ObjectStore,
    PostSource,
)
from app.application.summarize import Summarizer, SummarizerSettings
from app.application.title_case import TitleCaser
from app.configs import Config
from app.domain.constants import Language, LocalPaths, S3Paths
from app.domain.crawl import CrawlReport
from app.domain.newsletter import BuildConfiguration, header_title
from app.domain.post import Post, get_date_range
from app.logger import logger

# The intro used when the greeting model fails, in the issue's own language (the
# renderer's built-in default is English, which read oddly atop a Korean issue).
FALLBACK_INTRO: Final[dict[Language, str]] = {
    Language.KO: "이번 주 AI 기술 블로그에서 눈여겨볼 글을 골라 요약했습니다.",
    Language.EN: "Here are this week's AI tech blog posts worth your time, summarized.",
}


def fetch_and_filter_posts(
    config: Config,
    make_source: Callable[[list[str]], PostSource],
    models: Callable[[], ChatModelProvider],
    image_probe: ImageProbe,
    end_date_str: str | None = None,
    language: Language = Language.KO,
) -> tuple[list[Post], str, list[tuple[Post, str]], CrawlReport]:
    start_date, end_date = get_date_range(end_date_str, config.scraping.days_back)
    logger.info(
        "Fetching posts from '%s' up to (not including) '%s'",
        start_date.date(),
        end_date.date(),
    )

    collector = make_source(config.scraping.rss_urls)
    posts = collector.collect_posts(start_date, end_date)
    crawl_report = collector.report

    date_suffix = end_date.strftime("%Y-%m-%d")
    if not posts:
        logger.warning("No posts found in the specified date range")
        return [], date_suffix, [], crawl_report

    logger.info("Found %d posts", len(posts))
    logger.info("Posts: %s", [post.title for post in posts])
    logger.debug(pformat(posts))

    # Config -> settings translation lives here rather than in the Summarizer,
    # which takes validated settings and never reads the config itself.
    # Field names match the `summarization` config section, so the whole section
    # maps across without restating every key; extras are ignored by pydantic.
    summarizer = Summarizer(
        models(),
        SummarizerSettings.model_validate(
            config.summarization.model_dump()
            | {
                "min_content_length": config.scraping.min_content_length,
                "language": language,
            }
        ),
        image_probe,
    )

    # The score-rank and max_posts cap are applied inside process_posts BEFORE
    # summarization, so we never pay to summarize posts that get discarded.
    filtered_posts = summarizer.process_posts(posts)

    logger.info("Successfully summarized %d posts", len(filtered_posts))
    logger.info("Filtered posts: %s", [post.title for post in filtered_posts])
    logger.debug(pformat(filtered_posts))

    return filtered_posts, date_suffix, summarizer.filtered_out_posts, crawl_report


def process_posts_and_create_newsletter(
    posts: list[Post],
    date_suffix: str,
    config: Config,
    store: ObjectStore,
    models: Callable[[], ChatModelProvider],
    make_builder: Callable[[Path, Path], NewsletterBuilderPort],
    root_dir: Path,
    language: Language = Language.KO,
    archive_suffix: str = "",
) -> Path:
    """Build the issue and archive it in the store, under ``archive_suffix`` if
    given (see ``delivery.archive_suffix``)."""
    _title_case_titles(posts, config, models)
    inputs_dir = _prepare_and_save_posts(posts, date_suffix, root_dir)
    first_section_intro = _generate_greeting(
        posts, date_suffix, config, models, language, root_dir
    )
    outputs_dir = root_dir / LocalPaths.OUTPUTS_DIR.value
    outputs_dir.mkdir(parents=True, exist_ok=True)
    newsletter_path, article_filenames = _build_newsletter(
        make_builder,
        inputs_dir,
        outputs_dir,
        date_suffix,
        first_section_intro,
        config,
        language,
    )
    suffix = f"/{archive_suffix}" if archive_suffix else ""
    _upload_files(
        newsletter_path,
        f"{S3Paths.NEWSLETTERS.value}{suffix}",
        config,
        store,
    )
    if article_filenames:
        article_paths = [
            outputs_dir / LocalPaths.ARTICLES_DIR.value / filename
            for filename in article_filenames
        ]
        _upload_files(
            article_paths,
            f"{S3Paths.ARTICLES.value}{suffix}",
            config,
            store,
        )
    return newsletter_path


def log_run_summary(
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
        # Empty AND offering nothing to filter — a moved feed or a stale selector,
        # as opposed to a blog that was quiet. Charting this separately is the
        # point: the empty count is ~12 every week and says nothing.
        "sources_likely_broken": len(crawl_report.likely_broken),
        "sources_failed": len(crawl_report.failed),
        "sources_pages_unreadable": len(crawl_report.pages_unreadable),
        # Feeds whose oldest entry is inside the window: posts may have scrolled
        # off. Counted only; overlapping feeds can cover the gap.
        "sources_truncated": len(crawl_report.truncated),
        "posts_collected": crawl_report.total_posts,
        "posts_filtered_out": len(filtered_out_posts),
        "posts_in_digest": len(posts),
        "digest": [
            {"source": p.source, "score": round(p.score, 2), "title": p.title}
            for p in posts
        ],
    }
    logger.info("RUN_SUMMARY %s", json.dumps(summary, ensure_ascii=False))


def _prepare_and_save_posts(
    posts: list[Post], date_suffix: str, root_dir: Path
) -> Path:
    inputs_dir = root_dir / LocalPaths.INPUTS_DIR.value / date_suffix
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


def _title_case_titles(
    posts: list[Post], config: Config, models: Callable[[], ChatModelProvider]
) -> None:
    """Recase the shipping titles in place: feeds mix sentence and title case."""
    if not posts:
        return
    try:
        caser = TitleCaser(models(), config.summarization.greeting_model_id)
    except Exception as e:
        # Like the greeting: cosmetic, and every summary is already paid for.
        logger.error("Title caser unavailable (%s); keeping the feed titles.", e)
        return
    for post, title in zip(posts, caser.recase([p.title for p in posts]), strict=True):
        post.title = title


def _generate_greeting(
    posts: list[Post],
    date_suffix: str,
    config: Config,
    models: Callable[[], ChatModelProvider],
    language: Language,
    root_dir: Path,
) -> str:
    greeter = Greeter(models(), config.summarization.greeting_model_id, language)
    # The summary's opening gives the model a detail beyond the card's own lede.
    lines = "\n".join(
        f"- {post.title}\n  card lede (already shown): {post.one_liner}"
        f"\n  summary opening: {_summary_opening(post.summary)}"
        for post in posts
    )
    context = f"Today is {date_suffix}. This week's articles:\n{lines}"
    try:
        greeting = greeter.greet(context=context)
    except Exception as e:
        # The intro is one paragraph above the articles, and by now every summary
        # has been paid for: losing it must not lose the issue.
        logger.error("Greeting generation failed (%s); using the fixed intro.", e)
        return FALLBACK_INTRO[language]
    logger.debug("Generated greeting: %s", greeting)
    _save_greeting_to_file(greeting, date_suffix, root_dir)
    return greeting


def _summary_opening(summary_html: str) -> str:
    """The first paragraph of a summary as plain text, cut at 300 chars."""
    paragraph = BeautifulSoup(summary_html or "", "html.parser").find("p")
    text = " ".join(paragraph.get_text(" ").split()) if paragraph else ""
    return text[:300]


def _save_greeting_to_file(greeting: str, date_suffix: str, root_dir: Path) -> None:
    greeting_path = (
        root_dir
        / LocalPaths.INPUTS_DIR.value
        / date_suffix
        / LocalPaths.GREETING_FILE.value
    )
    greeting_path.write_text(greeting, encoding="utf-8")


def _build_newsletter(
    make_builder: Callable[[Path, Path], NewsletterBuilderPort],
    inputs_dir: Path,
    outputs_dir: Path,
    date_suffix: str,
    first_section_intro: str,
    config: Config,
    language: Language,
) -> tuple[Path, list[str]]:
    builder = make_builder(inputs_dir.parent, outputs_dir)
    # Only the chrome fields the config sets are passed, so an unset one falls
    # through to BuildConfiguration's default rather than becoming "" (which the
    # renderer rejects for the title).
    chrome = {
        "header_title": header_title(config.newsletter.header_title),
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


def _upload_files(
    local_paths: Path | list[Path],
    s3_prefix: str,
    config: Config,
    store: ObjectStore,
) -> None:
    s3_destination = (
        f"{config.resources.s3_prefix}/{s3_prefix}"
        if config.resources.s3_prefix
        else s3_prefix
    )
    paths_to_upload = [local_paths] if isinstance(local_paths, Path) else local_paths
    for path in paths_to_upload:
        if path.exists():
            store.upload(path, prefix=s3_destination)
        else:
            logger.warning("File not found, skipping upload: '%s'", path)

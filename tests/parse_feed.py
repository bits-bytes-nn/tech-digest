import sys
from pathlib import Path
from pprint import pformat
from typing import Final

sys.path.append(str(Path(__file__).parent.parent))
from app.configs import Config
from app.src import (
    AnthropicBlogScraper,
    AppConstants,
    GoogleBlogScraper,
    LinkedInBlogScraper,
    MetaAIBlogScraper,
    QwenBlogScraper,
    PostCollector,
    PostFetcher,
    RssFetcher,
    XAIBlogScraper,
    get_date_range,
    logger,
)

END_DATE: Final[str | None] = None

if __name__ == "__main__":
    config = Config.load()
    start_date, end_date = get_date_range(END_DATE, config.scraping.days_back)
    logger.info("Fetching posts from '%s' to '%s'", start_date.date(), end_date.date())

    fetchers: list[PostFetcher] = []
    for url in config.scraping.rss_urls:
        if AppConstants.External.ANTHROPIC_ENGINEERING.value in url:
            logger.info("Using AnthropicBlogScraper for URL: '%s'", url)
            fetchers.append(AnthropicBlogScraper(page_url=url, source="anthropic"))

        elif AppConstants.External.GOOGLE_RESEARCH.value in url:
            logger.info("Using GoogleBlogScraper for URL: '%s'", url)
            fetchers.append(GoogleBlogScraper(page_url=url, source="google"))

        elif AppConstants.External.LINKEDIN_ENGINEERING.value in url:
            logger.info("Using LinkedInBlogScraper for URL: '%s'", url)
            fetchers.append(LinkedInBlogScraper(page_url=url, source="linkedin"))

        elif AppConstants.External.META_AI.value in url:
            logger.info("Using MetaAIBlogScraper for URL: '%s'", url)
            fetchers.append(MetaAIBlogScraper(page_url=url, source="meta"))

        elif AppConstants.External.QWEN.value in url:
            logger.info("Using QwenBlogScraper for URL: '%s'", url)
            fetchers.append(QwenBlogScraper(page_url=url, source="qwen"))

        elif AppConstants.External.XAI.value in url:
            logger.info("Using XAIBlogScraper for URL: '%s'", url)
            fetchers.append(XAIBlogScraper(page_url=url, source="xai"))

        else:
            fetchers.append(RssFetcher(url))

    collector = PostCollector(fetchers)
    posts = collector.collect_posts(start_date, end_date)

    logger.info("Found %d posts", len(posts))
    if posts:
        logger.info("Posts: '%s'", [post.title for post in posts])
        logger.debug(pformat([post.model_dump_json() for post in posts]))

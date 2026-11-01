# NOTE: If there is no RSS feed available, you need to create a custom scraper here

import socket
from contextlib import contextmanager
from datetime import UTC, datetime
from html import unescape
from typing import ClassVar, Protocol
from urllib.parse import urljoin

import feedparser
import requests
from bs4 import BeautifulSoup, Tag
from feedparser.exceptions import CharacterEncodingOverride

from app.adapters.http import (
    make_robust_request,
    parse_page,
    try_request,
)
from app.domain.content import visible_text
from app.domain.crawl import CrawlReport, SourceHealth, SourceStatus
from app.domain.post import (
    Post,
    is_date_in_range,
    normalize_title_key,
    parse_published_date,
    try_parse_published_date,
)
from app.logger import logger


class ScraperConfig:
    # Measured across every configured host: only these two ever held the
    # article. Otherwise <body> is used, unless the feed carries its own body.
    CONTENT_SELECTORS: ClassVar[list[str]] = ["article", "main"]
    # Share of a page's visible text a CONTENT_SELECTORS match must hold before it
    # is preferred over the whole <body>. See ``_scrape_full_content``.
    CONTENT_CONTAINER_MIN_SHARE: ClassVar[float] = 0.5
    # Visible-text length below which a feed entry is a teaser and the article
    # page is fetched instead. Not ``scraping.min_content_length``, the ~600-char
    # gate on whether a post is worth summarizing at all.
    FULL_SCRAPE_TEXT_THRESHOLD: ClassVar[int] = 3000


class SourceFetchError(Exception):
    """Raised when a source cannot be fetched at all (network error, anti-bot
    block, HTTP error) — as opposed to fetching successfully but finding no
    posts. The collector uses this to mark a source FAILED vs. EMPTY."""


def post_from_entry(entry: feedparser.FeedParserDict) -> Post:
    """Build a Post from a feed entry or a scraper-built ``FeedParserDict``.

    ``source`` is always DERIVED from the link via ``SOURCE_MAPPING``, never
    read off the entry: that map is the single source of truth, and a
    per-scraper source would be a second copy of it that could drift.
    """
    link_str = getattr(entry, "link", "")
    content, page_unreadable = _extract_content_from_entry(entry, link_str)
    return Post(
        # Some feeds escape the title twice (Amazon Science sends
        # "don&amp;#8217;t"), so feedparser hands back "don&#8217;t" and the
        # card showed the entity literally. Decoding is idempotent on a
        # title that was already plain text.
        title=unescape(getattr(entry, "title", "No Title")).strip(),
        link=link_str,
        published_date=parse_published_date(entry_date_text(entry)),
        source=Post._determine_source(link_str),
        content=content,
        # ``summary`` is not seeded from the feed's teaser: it is the slot
        # the LLM summary lands in, and a teaser there could ship as the
        # article body of a post whose summarization failed.
        images=Post._extract_images(content, link_str),
        page_unreadable=page_unreadable,
    )


def _extract_content_from_entry(
    entry: feedparser.FeedParserDict, link: str
) -> tuple[str, bool]:
    """The post's content, and whether the article page was needed but could
    not be read."""
    content_items = getattr(entry, "content", [])
    content = (
        content_items[0].get("value")
        if content_items
        else getattr(entry, "summary", "")
    )
    # Visible text, not raw HTML length, so the scrape decision agrees with
    # the content gate on what a short post is.
    feed_text_len = len(visible_text(content))
    if feed_text_len >= ScraperConfig.FULL_SCRAPE_TEXT_THRESHOLD:
        return content, False
    # An entry with its own body (content:encoded) holds the article or its
    # lead; the page's whole <body> is that article plus the site around it
    # (amazon.science: the article is 16% of the page), so it is no better.
    # A summary-only feed has nothing better, so <body> is still taken there.
    scraped_content = _scrape_full_content(
        link, body_fallback=not (content_items and feed_text_len)
    )
    # None: the page could not be read. "": it was read but holds no article
    # container we trust, so a teaser kept from the feed is still unreadable.
    if not scraped_content:
        return content, True
    # A thin page (JS-rendered, consent wall) must not replace a fuller feed
    # body and push the post below the content gate.
    if len(visible_text(scraped_content)) > feed_text_len:
        content = scraped_content
    return content, False


def _scrape_full_content(url: str, *, body_fallback: bool) -> str | None:
    """The article's HTML from its own page.

    Returns None if the page cannot be fetched or parsed, and "" if it holds
    no trusted article container and ``body_fallback`` is off.

    The container chosen is the one that HOLDS the prose, not the first
    selector that matches: on Hugging Face's blog ``<article>`` is a
    near-empty wrapper (82 visible characters against 21,669 on the page).
    """
    if not url or not (response := make_robust_request(url)):
        return None
    try:
        soup = parse_page(response)
        body = soup.find("body")
        if body is None:
            return None
        # get_text on the parsed element, not visible_text on its string: the
        # latter would re-parse every candidate.
        body_length = len(body.get_text(separator=" ", strip=True))
        best: Tag | None = None
        best_length = 0
        for selector in ScraperConfig.CONTENT_SELECTORS:
            for element in soup.select(selector):
                length = len(element.get_text(separator=" ", strip=True))
                if length > best_length:
                    best, best_length = element, length
        # A narrower container excludes nav/footer chrome, but is trusted only
        # if it holds most of the page's prose: the HF wrapper held 0.4%, a
        # real <article> 60-90% (NVIDIA: 94%).
        if best is not None and best_length >= body_length * (
            ScraperConfig.CONTENT_CONTAINER_MIN_SHARE
        ):
            return str(best)
        return str(body) if body_fallback else ""
    except Exception as e:
        logger.error("Error parsing content for '%s': %s", url, e)
        return None


@contextmanager
def force_ipv4():
    original_getaddrinfo = socket.getaddrinfo

    def getaddrinfo_ipv4(*args, **kwargs):
        return [
            res
            for res in original_getaddrinfo(*args, **kwargs)
            if res[0] == socket.AF_INET
        ]

    socket.getaddrinfo = getaddrinfo_ipv4
    try:
        yield
    finally:
        socket.getaddrinfo = original_getaddrinfo


def entry_date_text(entry: feedparser.FeedParserDict) -> str:
    """The publication-date text of a feed entry, or "" if it carries none.

    Reads ``published`` first and falls back to ``updated``. Atom makes
    ``<published>`` OPTIONAL while ``<updated>`` is REQUIRED, and feedparser maps
    the two to distinct attributes, so a valid Atom feed that carries only
    ``<updated>`` has no ``entry.published`` at all. Without the fallback every
    entry of such a feed would fail the fail-closed date gate, silently: the
    source would still report candidates and read as a quiet blog.

    ``published`` wins when both exist. ``updated`` is a MODIFICATION time, so
    preferring it would let an old post edited this week re-enter the weekly
    window. When ``published`` is absent, ``updated`` is the only date the feed
    offers, and using it beats dropping the post.
    """
    return getattr(entry, "published", "") or getattr(entry, "updated", "")


class PostFetcher(Protocol):
    source_url: str
    #: Items this fetcher last offered BEFORE date filtering. Set by ``fetch``; see
    #: ``SourceHealth.candidates`` for why the collector records it.
    candidates: int | None

    def fetch(self, start_date: datetime, end_date: datetime) -> list[Post]: ...


class RssFetcher:
    FEED_HEADERS: ClassVar[dict[str, str]] = {
        "User-Agent": feedparser.USER_AGENT,
        "Accept": "application/xml, application/rss+xml",
    }

    def __init__(self, rss_url: str):
        self.rss_url = rss_url
        self.source_url = rss_url
        self.candidates: int | None = None
        #: The feed's oldest dated entry is newer than the window start, so posts
        #: from the start of the window may already have scrolled off it.
        self.truncated = False

    def fetch(self, start_date: datetime, end_date: datetime) -> list[Post]:
        logger.info("Fetching posts from RSS feed: '%s'", self.rss_url)
        # Reset first: a stale count from a previous call would describe the wrong
        # fetch if this collector is reused.
        self.candidates = None
        self.truncated = False
        posts = []
        try:
            with force_ipv4():
                feed = self._fetch_feed()

                if feed.bozo:
                    bozo_exc = feed.get("bozo_exception")
                    is_encoding_error = isinstance(bozo_exc, CharacterEncodingOverride)
                    (logger.warning if is_encoding_error else logger.error)(
                        f"Error parsing feed '{self.rss_url}': {bozo_exc}"
                    )
                    # A non-encoding bozo error on a feed with no usable entries
                    # is a genuine fetch/parse failure (e.g. HTTP error page,
                    # malformed XML) — surface it so the source is marked FAILED.
                    if not is_encoding_error and not feed.entries:
                        raise SourceFetchError(f"Feed parse error: {bozo_exc}")

                # Fail closed: an entry whose date cannot be parsed is excluded
                # rather than dated to "now" and let through.
                dates = [
                    try_parse_published_date(entry_date_text(entry))
                    for entry in feed.entries
                ]
                # Entries the feed offered WITH a readable date, before the date
                # window. A feed that carries none has moved or broken, and so has
                # one whose dates no longer parse: either way nothing can ever
                # pass the window. A quiet blog still lists a dated back catalogue.
                self.candidates = sum(d is not None for d in dates)
                self._check_coverage([d for d in dates if d is not None], start_date)

                skipped = 0
                for entry, published_date in zip(feed.entries, dates, strict=True):
                    if published_date is None or not is_date_in_range(
                        published_date, start_date, end_date
                    ):
                        continue
                    # One malformed entry (e.g. a link urlparse rejects) must
                    # cost that entry, not every valid post in the feed.
                    try:
                        posts.append(post_from_entry(entry))
                    except Exception as e:
                        skipped += 1
                        logger.error(
                            "Skipping an entry of '%s' that could not be read: %s",
                            self.rss_url,
                            e,
                        )
                # Entries in the window existed and every one failed: that is a
                # broken source, not a quiet week, and must not read as EMPTY.
                if skipped and not posts:
                    raise SourceFetchError(
                        f"All {skipped} in-window entries of '{self.rss_url}' "
                        "failed to parse"
                    )

        except SourceFetchError:
            raise
        except Exception as e:
            raise SourceFetchError(
                f"Failed to fetch or process feed '{self.rss_url}': {e}"
            ) from e

        return posts

    def _check_coverage(self, dates: list[datetime], start_date: datetime) -> None:
        """Flag a feed whose oldest dated entry is newer than the window start.

        A feed lists only its newest N entries, so a busy blog can push the start
        of the week off it before the run. Nothing is stored between runs: if the
        oldest entry offered is inside the window, the feed did not cover it.
        """
        if not dates:
            return
        oldest = min(dates)
        if oldest > start_date.astimezone(UTC):
            self.truncated = True
            logger.info(
                "Feed '%s' does not reach back to the window start %s: its oldest "
                "entry is dated %s, so earlier posts may be missing.",
                self.rss_url,
                start_date.date(),
                oldest.isoformat(),
            )

    def _fetch_feed(self) -> feedparser.FeedParserDict:
        """Download the feed through the request layer, then parse the bytes.

        ``feedparser.parse(url)`` fetches with urllib and sets no socket timeout,
        so a feed host that stalls would hold the job until the Batch timeout.
        The request layer adds the timeout, the transient-error retry and the
        per-hop SSRF check. The headers are the ones feedparser sends itself.
        """
        response = try_request(requests.Session(), self.rss_url, self.FEED_HEADERS)
        if response is None:
            raise SourceFetchError(f"Could not fetch feed '{self.rss_url}'")
        headers = {key.lower(): value for key, value in response.headers.items()}
        # Relative links in the feed resolve against where it was actually served.
        headers["content-location"] = response.url
        return feedparser.parse(response.content, response_headers=headers)


class GenericPageScraper:
    """Base for a site with no feed: fetch its index page and parse each card.

    Subclasses set ``ITEM_SELECTOR`` and implement ``_parse_item``. A feed is
    always the better choice (RssFetcher needs no code), so this exists only for
    sites that offer none.
    """

    ITEM_SELECTOR: ClassVar[str] = ""

    def __init__(self, page_url: str):
        self.page_url = page_url
        self.source_url = page_url
        self.candidates: int | None = None
        self._dated = 0

    def _parse_date(self, text: str) -> datetime | None:
        """Parse a card's date, counting each success toward ``candidates``.

        ``candidates`` is the number of cards with a READABLE date, before the
        window: a scraper whose date markup broke reports 0 and is flagged as
        likely broken, instead of reading as a quiet blog.
        """
        parsed = try_parse_published_date(text)
        if parsed is not None:
            self._dated += 1
        return parsed

    def _fetch_page(self) -> BeautifulSoup:
        if response := make_robust_request(self.page_url):
            return parse_page(response)
        raise SourceFetchError(
            f"All request attempts failed for '{self.page_url}' "
            f"(likely anti-bot block, datacenter-IP block, or network error)"
        )

    def fetch(self, start_date: datetime, end_date: datetime) -> list[Post]:
        soup = self._fetch_page()
        posts = []
        items = soup.select(self.ITEM_SELECTOR)
        # Zero dated items means the selector or the date markup no longer
        # matches this site, which is indistinguishable from "no posts this
        # week" without this count (see ``_parse_date``).
        self._dated = 0
        for item in items:
            try:
                if not (entry_data := self._parse_item(item)):
                    continue
                published_value = entry_data.get("published", "")
                published_str = (
                    published_value if isinstance(published_value, str) else ""
                )
                pub_date = self._parse_date(published_str)
                if pub_date is not None and is_date_in_range(
                    pub_date, start_date, end_date
                ):
                    posts.append(post_from_entry(entry_data))
            except Exception as e:
                logger.error("Error processing item from %s: %s", self.page_url, e)
        self.candidates = self._dated
        return posts

    def _parse_item(self, item: Tag) -> feedparser.FeedParserDict | None:
        raise NotImplementedError


class LinkedInBlogScraper(GenericPageScraper):
    ITEM_SELECTOR: ClassVar[str] = "li.post-list__item"
    # The index cuts long titles to "...". The full one is only on the post page.
    TRUNCATION_MARKERS: ClassVar[tuple[str, ...]] = ("...", "\u2026")

    def fetch(self, start_date: datetime, end_date: datetime) -> list[Post]:
        posts = super().fetch(start_date, end_date)
        for post in posts:
            if post.title.endswith(self.TRUNCATION_MARKERS):
                post.title = self._full_title(str(post.link)) or post.title
        return posts

    @staticmethod
    def _full_title(url: str) -> str:
        if not (response := make_robust_request(url)):
            return ""
        soup = parse_page(response)
        meta = soup.find("meta", attrs={"property": "og:title"})
        if isinstance(meta, Tag) and (content := meta.get("content")):
            return str(content).strip()
        return soup.title.get_text(strip=True) if soup.title else ""

    def _parse_item(self, item: Tag) -> feedparser.FeedParserDict | None:
        date_elem = item.select_one("p.grid-post__date")
        title_link_elem = item.select_one("a.grid-post__link")

        if not (date_elem and title_link_elem):
            return None

        href = title_link_elem.get("href")
        if not href:
            return None

        return feedparser.FeedParserDict(
            title=title_link_elem.text.strip(),
            # A relative href would resolve the source to "unknown" and leave the
            # full-content scrape an empty host.
            link=urljoin(self.page_url, str(href)),
            published=date_elem.text.strip(),
        )


class ScraperRegistry:
    # URL fragment -> scraper. No source is recorded alongside the class: the
    # source comes from ``SOURCE_MAPPING`` via the post's link, so pairing one
    # here would be a second copy of that map.
    _SCRAPER_MAPPING: ClassVar[dict[str, type[GenericPageScraper]]] = {
        "www.linkedin.com/blog/engineering": LinkedInBlogScraper,
        # NOTE: add new scrapers here (only for sites with no feed)
    }

    @classmethod
    def get_fetcher(cls, url: str) -> PostFetcher:
        for url_pattern, scraper_class in cls._SCRAPER_MAPPING.items():
            if url_pattern in url:
                logger.info("Using %s for URL: '%s'", scraper_class.__name__, url)
                return scraper_class(page_url=url)
        return RssFetcher(url)

    @classmethod
    def create_fetchers(cls, urls: list[str]) -> list[PostFetcher]:
        return [cls.get_fetcher(url) for url in urls]


class PostCollector:
    def __init__(self, fetchers: list[PostFetcher]):
        self.fetchers = fetchers
        self.report = CrawlReport()

    @classmethod
    def from_urls(cls, urls: list[str]) -> "PostCollector":
        return cls(ScraperRegistry.create_fetchers(urls))

    def collect_posts(self, start_date: datetime, end_date: datetime) -> list[Post]:
        logger.info(
            f"Collecting posts from '{start_date.date()}' to '{end_date.date()}'"
        )
        self.report = CrawlReport()
        all_posts: list[Post] = []
        seen_links: set[str] = set()
        seen_titles: set[str] = set()
        for fetcher in self.fetchers:
            url = getattr(fetcher, "source_url", type(fetcher).__name__)
            fetcher_name = type(fetcher).__name__
            try:
                fetched = fetcher.fetch(start_date, end_date)
                contributed = 0
                unreadable = 0
                for post in fetched:
                    if post.link in seen_links:
                        continue
                    title_key = normalize_title_key(post.title)
                    # Skip a syndicated duplicate of a post we already collected
                    # from an earlier source. Keeping the first occurrence makes
                    # the winner deterministic (feeds are iterated in config
                    # order), so the same week always resolves the same way.
                    if title_key and title_key in seen_titles:
                        logger.info(
                            "Skipping cross-source duplicate '%s' from '%s'.",
                            post.title,
                            url,
                        )
                        continue
                    all_posts.append(post)
                    contributed += 1
                    unreadable += post.page_unreadable
                    seen_links.add(post.link)
                    if title_key:
                        seen_titles.add(title_key)
                status = SourceStatus.OK if fetched else SourceStatus.EMPTY
                self.report.sources.append(
                    SourceHealth(
                        url=url,
                        fetcher=fetcher_name,
                        status=status,
                        # Read defensively, as ``source_url`` is: a fetcher that
                        # reports nothing yields None, which the report reads as
                        # "unknown" rather than "zero".
                        candidates=getattr(fetcher, "candidates", None),
                        # Posts this source CONTRIBUTED, not posts it returned,
                        # so syndicated duplicates do not inflate the health line.
                        post_count=contributed,
                        duplicates=len(fetched) - contributed,
                        unreadable_pages=unreadable,
                        # Only RssFetcher can tell; other fetchers report False.
                        truncated=getattr(fetcher, "truncated", False),
                    )
                )
                if not fetched:
                    logger.warning("Source produced no posts in range: '%s'", url)
                elif not contributed:
                    # INFO: overlapping feeds (the AWS category feeds) are
                    # configured on purpose.
                    logger.info(
                        "Source '%s' returned %d post(s), all already collected "
                        "from an earlier source.",
                        url,
                        len(fetched),
                    )
            except Exception as e:
                logger.error("Fetcher '%s' failed for '%s': %s", fetcher_name, url, e)
                self.report.sources.append(
                    SourceHealth(
                        url=url,
                        fetcher=fetcher_name,
                        status=SourceStatus.FAILED,
                        error=str(e),
                    )
                )
        logger.info("Crawl health: %s", self.report.summary_line())
        if self.report.likely_broken:
            # Named here, not only in the SNS alert, which is suppressed for
            # expected-flaky sources and depends on a live subscription.
            logger.warning(
                "%d source(s) offered nothing for the date window to filter — a "
                "moved feed or a stale selector rather than a quiet blog: %s",
                len(self.report.likely_broken),
                [s.url for s in self.report.likely_broken],
            )
        if self.report.pages_unreadable:
            logger.warning(
                "%d source(s) left every post with only its feed teaser — the "
                "article pages could not be read: %s",
                len(self.report.pages_unreadable),
                [s.url for s in self.report.pages_unreadable],
            )
        return sorted(all_posts, key=lambda p: p.published_date, reverse=True)

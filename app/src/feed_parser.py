# NOTE: If there is no RSS feed available, you need to create a custom scraper here

import ipaddress
import re
import socket
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import ClassVar, Literal, Protocol, TypeAlias
from urllib.parse import urljoin, urlparse

import feedparser
import requests
from bs4 import BeautifulSoup, Tag
from feedparser.exceptions import CharacterEncodingOverride
from pydantic import BaseModel, Field, field_validator
from requests.exceptions import RequestException

from .constants import AppConstants
from .logger import logger

# Cap on redirects we will follow per request. Bounds redirect-chain abuse and
# keeps a misbehaving source from looping us.
MAX_REDIRECTS: int = 5


def _is_blocked_host(host: str) -> bool:
    """True if ``host`` is an IP literal in a private / loopback / link-local /
    reserved range — the SSRF-sensitive targets (notably the cloud metadata
    endpoint 169.254.169.254). Hostnames are not resolved here: this is a
    cheap literal guard, not a full DNS-rebinding defense. Crawl targets are an
    operator-controlled allow-list (config rss_urls), so the realistic risk is a
    source REDIRECTING to an internal address, which this catches for IP-literal
    redirect targets.
    """
    if not host:
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False  # Not an IP literal; treat as a normal external hostname.
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


SourceType: TypeAlias = Literal[
    "airbnb",
    "amazon",
    "anthropic",
    "aws",
    "deepmind",
    "google",
    "huggingface",
    "kakao",
    "linkedin",
    "meta",
    "microsoft",
    "ncsoft",
    "netflix",
    "nvidia",
    "openai",
    "palantir",
    "pinterest",
    "qwen",
    "xai",
    "unknown",
    # NOTE: add new sources here
]


class HeaderCache:
    _cache: ClassVar[dict[str, int]] = {}

    @classmethod
    def get_cached_header_index(cls, domain: str) -> int | None:
        return cls._cache.get(domain)

    @classmethod
    def cache_header_index(cls, domain: str, index: int) -> None:
        cls._cache[domain] = index

    @classmethod
    def clear(cls) -> None:
        cls._cache.clear()


class ScraperConfig:
    CONTENT_SELECTORS: ClassVar[list[str]] = [
        "article",
        "main",
        "div.post-content",
        "div.entry-content",
        "div.content",
        "div.blog-content",
        "div.article-content",
    ]
    DATE_FORMATS: ClassVar[tuple[str, ...]] = (
        "%a, %d %b %Y %H:%M:%S GMT",
        "%a, %d %b %Y %H:%M:%S %z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%b %d, %Y",
        "%B %d, %Y",
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%d.%m.%Y",
    )
    DEFAULT_TAGS: ClassVar[list[str]] = ["uncategorized"]
    MIN_CONTENT_LENGTH: ClassVar[int] = 3000
    MARKDOWN_IMAGE_PATTERN: ClassVar[re.Pattern] = re.compile(r"!\[.*?]\((.*?)\)")
    MAX_TAGS: ClassVar[int] = 5
    # Full, realistic browser header sets. The Sec-Fetch-* / Sec-Ch-Ua and
    # Upgrade-Insecure-Requests headers materially reduce 403s from anti-bot
    # filters (e.g. Meta, Medium) that reject bare User-Agent-only requests.
    REQUEST_HEADERS_OPTIONS: ClassVar[list[dict[str, str]]] = [
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,ko;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
        },
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Upgrade-Insecure-Requests": "1",
        },
        {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Upgrade-Insecure-Requests": "1",
        },
    ]
    REQUEST_TIMEOUT: ClassVar[int] = 30
    SOURCE_MAPPING: ClassVar[dict[str, SourceType]] = {
        "airbnb-engineering": "airbnb",
        "amazon.science": "amazon",
        "anthropic.com": "anthropic",
        "aws.amazon.com": "aws",
        "deepmind.google": "google",
        "research.google": "google",
        "huggingface.co": "huggingface",
        "tech.kakao.com": "kakao",
        "linkedin.com": "linkedin",
        "ai.meta.com": "meta",
        "engineering.fb.com": "meta",
        "microsoft.com": "microsoft",
        "ncsoft.github.io": "ncsoft",
        "netflixtechblog.com": "netflix",
        "netflix-techblog": "netflix",  # Medium publication slug
        "developer.nvidia.com": "nvidia",
        "openai.com": "openai",
        "blog.palantir.com": "palantir",
        "palantir": "palantir",  # Medium @-handle
        "pinterest-engineering": "pinterest",
        "pinterest_engineering": "pinterest",  # Medium @-handle
        "qwenlm.github.io": "qwen",
        "x.ai": "xai",
        # NOTE: add new sources here
    }


class SourceFetchError(Exception):
    """Raised when a source cannot be fetched at all (network error, anti-bot
    block, HTTP error) — as opposed to fetching successfully but finding no
    posts. The collector uses this to mark a source FAILED vs. EMPTY."""


def _visible_text_length(html: str) -> int:
    """Number of visible-text characters in an HTML string (markup stripped).

    Single source of truth for "how much readable prose is here", used both by
    the full-article scrape decision and by the content-sufficiency gate so the
    two stay consistent.
    """
    if not html:
        return 0
    try:
        return len(
            BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)
        )
    except Exception:
        return len(html)


def _try_request(
    session: requests.Session, url: str, headers: dict[str, str]
) -> requests.Response | None:
    """Single GET attempt with one retry for transient errors (timeout / 5xx /
    429). Returns the response on success, or None if this header set fails."""
    session.headers.update(headers)
    session.max_redirects = MAX_REDIRECTS
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            response = session.get(
                url,
                timeout=ScraperConfig.REQUEST_TIMEOUT,
                allow_redirects=True,
                verify=True,
            )
            response.raise_for_status()
            # Re-check the host we actually landed on: a source could redirect to
            # an internal IP literal (e.g. the cloud metadata endpoint). Block it.
            final_host = urlparse(response.url).hostname or ""
            if _is_blocked_host(final_host):
                logger.warning(
                    "Request to '%s' redirected to blocked host '%s'; refusing.",
                    url,
                    final_host,
                )
                return None
            return response
        except RequestException as e:
            resp = getattr(e, "response", None)
            status = getattr(resp, "status_code", None)
            transient = status in (429, 500, 502, 503, 504) or status is None
            if transient and attempt < max_attempts - 1:
                # Exponential backoff with jitter; honor Retry-After (429) when
                # present. An immediate re-fire against a rate limiter just earns
                # another 429, so a real delay is required to be useful.
                delay = _retry_delay(resp, attempt)
                logger.debug(
                    "Transient error for '%s' (%s); retry %d after %.1fs",
                    url,
                    status,
                    attempt + 1,
                    delay,
                )
                time.sleep(delay)
                continue
            logger.warning("Request to '%s' failed: %s", url, e)
            return None
    return None


def _retry_delay(response: requests.Response | None, attempt: int) -> float:
    """Backoff seconds: honor a numeric Retry-After header, else exponential
    (1s, 2s, ...) with a small deterministic jitter derived from the attempt."""
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            return min(float(retry_after), 30.0)
    base = float(2**attempt)
    jitter = 0.1 * (attempt + 1)
    return min(base + jitter, 30.0)


def _make_robust_request(url: str) -> requests.Response | None:
    """Fetch a URL trying several realistic browser header sets, remembering the
    one that works per-domain. Returns None only if every header set fails
    (network error, persistent anti-bot block, HTTP error)."""
    parsed = urlparse(url)
    if _is_blocked_host(parsed.hostname or ""):
        logger.warning("Refusing request to blocked host: '%s'", url)
        return None

    session = requests.Session()
    domain = parsed.netloc.lower()

    # Try the header set that previously worked for this domain first.
    ordered_indices = list(range(len(ScraperConfig.REQUEST_HEADERS_OPTIONS)))
    cached_index = HeaderCache.get_cached_header_index(domain)
    if cached_index is not None:
        ordered_indices.remove(cached_index)
        ordered_indices.insert(0, cached_index)

    for i in ordered_indices:
        headers = ScraperConfig.REQUEST_HEADERS_OPTIONS[i]
        if response := _try_request(session, url, headers):
            HeaderCache.cache_header_index(domain, i)
            logger.info("Successfully fetched '%s' with header set %d", url, i + 1)
            return response

    logger.error("All header attempts failed for '%s'", url)
    return None


class Post(BaseModel):
    title: str
    link: str
    published_date: datetime
    content: str = ""
    images: list[str] = Field(default_factory=list)
    source: SourceType = "unknown"
    summary: str = ""
    tags: list[str] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)
    score: float = 0.0

    @field_validator("tags", mode="before")
    @classmethod
    def validate_tags(cls, v):
        if not v:
            return ScraperConfig.DEFAULT_TAGS.copy()
        # Preserve the source ordering (tags are emitted most-significant-first)
        # while de-duplicating, then cap. dict.fromkeys keeps first-seen order;
        # sorting here would discard relevance and drop important trailing tags.
        unique_tags = list(dict.fromkeys(tag for tag in v if isinstance(tag, str)))
        return unique_tags[: ScraperConfig.MAX_TAGS]

    @classmethod
    def from_entry(cls, entry: feedparser.FeedParserDict) -> "Post":
        link_str = getattr(entry, "link", "")
        content_html = cls._extract_content_from_entry(entry, link_str)
        return cls(
            title=getattr(entry, "title", "No Title").strip(),
            link=link_str,
            published_date=parse_published_date(getattr(entry, "published", "")),
            source=cls._determine_source(link_str),
            content=content_html,
            summary=getattr(entry, "summary", ""),
            images=cls._extract_images(content_html, link_str),
            tags=[
                tag.term for tag in getattr(entry, "tags", []) if hasattr(tag, "term")
            ],
        )

    @classmethod
    def _extract_content_from_entry(
        cls, entry: feedparser.FeedParserDict, link: str
    ) -> str:
        content_items = getattr(entry, "content", [])
        content = (
            content_items[0].get("value")
            if content_items
            else getattr(entry, "summary", "")
        )
        # Decide whether to scrape the full article based on VISIBLE text length,
        # not raw HTML length. A feed teaser that is mostly markup can exceed the
        # raw-HTML threshold yet have little readable prose; measuring visible
        # text keeps the scrape decision consistent with the content gate (\u00a711)
        # so we don't skip scraping and then drop the thin post.
        if _visible_text_length(content) < ScraperConfig.MIN_CONTENT_LENGTH and (
            scraped_content := cls._scrape_full_content(link)
        ):
            content = scraped_content
        return content.replace("\u202f", " ")

    @staticmethod
    def _scrape_full_content(url: str) -> str:
        if not url or not (response := _make_robust_request(url)):
            return ""
        try:
            soup = BeautifulSoup(response.text, "html.parser")
            for selector in ScraperConfig.CONTENT_SELECTORS:
                if content_element := soup.select_one(selector):
                    return str(content_element)
            return str(soup.find("body") or "")
        except Exception as e:
            logger.error(f"Error parsing content for '{url}': {e}")
            return ""

    @staticmethod
    def _determine_source(url: str) -> SourceType:
        if not url:
            return "unknown"
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower().removeprefix("www.")
        if domain == AppConstants.External.MEDIUM_DOMAIN.value:
            path_parts = parsed_url.path.strip("/").split("/")
            if path_parts and path_parts[0]:
                # Medium slugs/handles are case-insensitive (e.g. the feed
                # "@Pinterest_Engineering" maps to the lowercase key).
                handle = path_parts[0].lstrip("@").lower()
                source = ScraperConfig.SOURCE_MAPPING.get(handle)
                if source is not None:
                    return source
        return ScraperConfig.SOURCE_MAPPING.get(domain, "unknown")

    def text_length(self) -> int:
        """Length of the post's *visible* text, ignoring HTML markup.

        Used by the content-sufficiency gate: a post whose HTML is large only
        because of boilerplate/markup (nav bars, scripts, style) but whose
        readable prose is thin should not reach the summarizer, where it would
        otherwise produce an empty or low-quality summary.
        """
        return _visible_text_length(self.content)

    @staticmethod
    def _extract_images(content_html: str, base_url: str) -> list[str]:
        if not content_html:
            return []
        try:
            soup = BeautifulSoup(content_html, "html.parser")
            image_urls: set[str] = {
                urljoin(base_url, img["src"])
                for img in soup.find_all("img")
                if img.get("src")
            }
            image_urls.update(
                ScraperConfig.MARKDOWN_IMAGE_PATTERN.findall(content_html)
            )
            return [url for url in image_urls if url.startswith("http")]
        except Exception as e:
            logger.warning(f"Failed to extract images from '{base_url}': {e}")
            return []


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


def is_date_in_range(
    target_date: datetime, start_date: datetime, end_date: datetime
) -> bool:
    target_utc = (
        target_date.astimezone(UTC)
        if target_date.tzinfo
        else target_date.replace(tzinfo=UTC)
    )
    return start_date.astimezone(UTC) <= target_utc <= end_date.astimezone(UTC)


def try_parse_published_date(date_str: str) -> datetime | None:
    """Parse a date string to an aware UTC datetime, or return None on failure.

    This is the fail-CLOSED primitive used by the date-range gate: a post whose
    date cannot be parsed must be EXCLUDED, not silently treated as "now" (which
    would let undated/unparseable posts slip into every weekly window). Use
    ``parse_published_date`` only where a non-null sort key is required and the
    post has already cleared the gate.
    """
    if not isinstance(date_str, str) or not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00")).astimezone(UTC)
    except (ValueError, TypeError):
        pass
    for fmt in ScraperConfig.DATE_FORMATS:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)
        except (ValueError, TypeError):
            continue
    logger.warning(f"Failed to parse date '{date_str}' with any known format.")
    return None


def parse_published_date(date_str: str) -> datetime:
    """Parse a date string, falling back to ``now(UTC)`` when unparseable.

    Used for Post construction where a sort key must always exist. For the
    in/out-of-window decision use ``try_parse_published_date`` so undated posts
    are dropped rather than dated to the present moment.
    """
    return try_parse_published_date(date_str) or datetime.now(UTC)


class PostFetcher(Protocol):
    source_url: str

    def fetch(self, start_date: datetime, end_date: datetime) -> list[Post]: ...


class RssFetcher:
    def __init__(self, rss_url: str):
        self.rss_url = rss_url
        self.source_url = rss_url

    def fetch(self, start_date: datetime, end_date: datetime) -> list[Post]:
        logger.info(f"Fetching posts from RSS feed: '{self.rss_url}'")
        posts = []
        try:
            with force_ipv4():
                feed = feedparser.parse(
                    self.rss_url,
                    request_headers={"Accept": "application/xml, application/rss+xml"},
                )

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

                for entry in feed.entries:
                    # Fail closed: an entry whose date cannot be parsed is
                    # excluded rather than dated to "now" and let through.
                    published_date = try_parse_published_date(
                        getattr(entry, "published", "")
                    )
                    if published_date is not None and is_date_in_range(
                        published_date, start_date, end_date
                    ):
                        posts.append(Post.from_entry(entry))

        except SourceFetchError:
            raise
        except Exception as e:
            raise SourceFetchError(
                f"Failed to fetch or process feed '{self.rss_url}': {e}"
            ) from e

        return posts


class BasePageScraper:
    def __init__(self, page_url: str, source: SourceType):
        self.page_url = page_url
        self.source = source
        self.source_url = page_url

    def _fetch_page(self) -> BeautifulSoup:
        if response := _make_robust_request(self.page_url):
            return BeautifulSoup(response.text, "html.parser")
        raise SourceFetchError(
            f"All request attempts failed for '{self.page_url}' "
            f"(likely anti-bot block, datacenter-IP block, or network error)"
        )

    def fetch(self, start_date: datetime, end_date: datetime) -> list[Post]:
        raise NotImplementedError


class GenericPageScraper(BasePageScraper):
    ITEM_SELECTOR: ClassVar[str] = ""

    def fetch(self, start_date: datetime, end_date: datetime) -> list[Post]:
        soup = self._fetch_page()
        posts = []
        for item in soup.select(self.ITEM_SELECTOR):
            try:
                if not (entry_data := self._parse_item(item)):
                    continue
                published_value = entry_data.get("published", "")
                published_str = (
                    published_value if isinstance(published_value, str) else ""
                )
                pub_date = try_parse_published_date(published_str)
                if pub_date is not None and is_date_in_range(
                    pub_date, start_date, end_date
                ):
                    posts.append(Post.from_entry(entry_data))
            except Exception as e:
                logger.error(f"Error processing item from {self.page_url}: {e}")
        return posts

    def _parse_item(self, item: Tag) -> feedparser.FeedParserDict | None:
        raise NotImplementedError


class AnthropicBlogScraper(BasePageScraper):
    DATE_PATTERN: ClassVar[re.Pattern] = re.compile(
        r"\b([A-Z][a-z]{2,8} \d{1,2}, \d{4})\b"
    )

    def fetch(self, start_date: datetime, end_date: datetime) -> list[Post]:
        logger.info(
            f"Anthropic: Fetching posts from '{start_date.date()}' to '{end_date.date()}'"
        )

        soup = self._fetch_page()

        posts = []
        seen_urls = set()

        for link in soup.find_all("a", href=re.compile(r"/engineering/")):
            post = self._extract_post_from_link(link, start_date, end_date)
            if post and post.link not in seen_urls:
                posts.append(post)
                seen_urls.add(post.link)

        return posts

    def _extract_post_from_link(
        self, link: Tag, start_date: datetime, end_date: datetime
    ) -> Post | None:
        try:
            href = link.get("href")
            if not href:
                return None

            if isinstance(href, list):
                href = href[0] if href else None
            if not href:
                return None

            title = self._extract_title(link)
            if not title:
                return None

            date_text = self._find_date_near_element(link)
            if not date_text:
                return None

            pub_date = try_parse_published_date(date_text)
            if pub_date is None or not is_date_in_range(pub_date, start_date, end_date):
                return None

            return Post.from_entry(
                feedparser.FeedParserDict(
                    title=title.strip(),
                    link=urljoin(self.page_url, href),
                    published=pub_date.isoformat(),
                    source=self.source,
                )
            )

        except Exception as e:
            logger.error(f"Error processing link: {e}")
            return None

    @staticmethod
    def _extract_title(link: Tag) -> str:
        heading_elements = link.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
        if heading_elements:
            for heading in heading_elements:
                text = heading.get_text(strip=True)
                if 10 <= len(text) <= 120 and not text.endswith("."):
                    return text
            return heading_elements[0].get_text(strip=True)

        title_selectors = [
            '[class*="title"]',
            '[class*="heading"]',
            '[class*="headline"]',
        ]

        for selector in title_selectors:
            title_elements = link.select(selector)
            for elem in title_elements:
                text = elem.get_text(strip=True)
                if 10 <= len(text) <= 120 and not text.endswith("."):
                    return text

        full_text = link.get_text(strip=True)

        lines = [line.strip() for line in full_text.split("\n") if line.strip()]
        for line in lines:
            if (
                10 <= len(line) <= 120
                and not line.endswith(".")
                and not line.startswith("This is")
                and not line.startswith("A ")
                and "blog post" not in line.lower()
                and "technical report" not in line.lower()
            ):
                return line

        if lines and 5 <= len(lines[0]) <= 120:
            return lines[0]

        return full_text[:120].rsplit(" ", 1)[0] + (
            "..." if len(full_text) > 120 else ""
        )

    def _find_date_near_element(self, element: Tag) -> str | None:
        current = element
        for _ in range(5):
            if not current:
                break
            if match := self.DATE_PATTERN.search(str(current)):
                return match.group(1)
            current = current.parent
        return None


class GoogleBlogScraper(GenericPageScraper):
    ITEM_SELECTOR: ClassVar[str] = "a.glue-card"

    def _parse_item(self, item: Tag) -> feedparser.FeedParserDict | None:
        date_elem = item.select_one("p.glue-label")
        title_elem = item.select_one("span.headline-5")
        href = item.get("href")

        if not (date_elem and title_elem and href):
            return None

        return feedparser.FeedParserDict(
            title=title_elem.text.strip(),
            link=urljoin(self.page_url, str(href)),
            published=date_elem.text.strip(),
            source=self.source,
        )


class LinkedInBlogScraper(GenericPageScraper):
    ITEM_SELECTOR: ClassVar[str] = "li.post-list__item"

    def _parse_item(self, item: Tag) -> feedparser.FeedParserDict | None:
        date_elem = item.select_one("p.grid-post__date")
        title_link_elem = item.select_one("a.grid-post__link")

        if not (date_elem and title_link_elem):
            return None

        href = title_link_elem.get("href")
        if not href:
            return None

        tags = []
        if topic_elem := item.select_one("a.t-14.t-bold"):
            tags = [topic_elem.text.strip()]

        return feedparser.FeedParserDict(
            title=title_link_elem.text.strip(),
            link=str(href),
            published=date_elem.text.strip(),
            source=self.source,
            tags=[{"term": tag} for tag in tags],
        )


class MetaAIBlogScraper(BasePageScraper):
    DATE_PATTERNS: ClassVar[list[re.Pattern]] = [
        re.compile(r"\b([A-Z][a-z]{2,8} \d{1,2}, \d{4})\b"),
        re.compile(r"\b([A-Z][a-z]{2} \d{1,2}, \d{4})\b"),
        re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b"),
        re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"),
    ]
    FILTER_TITLES: ClassVar[set[str]] = {
        "...",
        "back",
        "blog",
        "continue reading",
        "featured",
        "home",
        "learn more",
        "more",
        "news",
        "next",
        "prev",
        "previous",
        "read more",
        "see more",
        "the latest",
        "view all",
    }

    def fetch(self, start_date: datetime, end_date: datetime) -> list[Post]:
        logger.info(
            f"Meta AI: Fetching posts from '{start_date.date()}' to '{end_date.date()}'"
        )

        soup = self._fetch_page()

        posts = []
        seen_urls = set()
        blog_links = soup.find_all("a", href=re.compile(r"/blog/[^/]+/?$"))
        logger.info(f"Meta AI: Found {len(blog_links)} potential blog links")

        for i, link in enumerate(blog_links):
            try:
                href = link.get("href", "")
                if not href or href in seen_urls:
                    continue

                if href.endswith("/blog/") or "?page=" in href:
                    continue

                seen_urls.add(href)

                title = self._extract_title(link)
                if not title or self._should_filter_title(title):
                    logger.debug(f"Meta AI: Filtered out '{title}' (item {i})")
                    continue

                date_text = self._find_date_near_element(link)
                if not date_text:
                    logger.debug(f"Meta AI: No date found for '{title}' (item {i})")
                    continue

                pub_date = try_parse_published_date(date_text)
                if pub_date is None or not is_date_in_range(
                    pub_date, start_date, end_date
                ):
                    logger.debug(f"Meta AI: '{title}' date unparseable or out of range")
                    continue

                # urljoin handles both relative ("/blog/x") and absolute hrefs
                # against the scraper's own page_url, so the host is never
                # hardcoded here (single source of truth: the configured URL).
                full_url = urljoin(self.page_url, href)

                post = Post.from_entry(
                    feedparser.FeedParserDict(
                        title=title.strip(),
                        link=full_url,
                        published=pub_date.isoformat(),
                        source=self.source,
                    )
                )
                posts.append(post)
                logger.info(f"Meta AI: Added '{title}' ({pub_date.date()})")

            except Exception as e:
                logger.error(f"Meta AI: Error processing link {i}: {e}")

        unique_posts = []
        seen_links = set()
        for post in posts:
            if str(post.link) not in seen_links:
                unique_posts.append(post)
                seen_links.add(str(post.link))

        logger.info(f"Meta AI: Found {len(unique_posts)} unique posts in date range")
        return unique_posts

    def _extract_title(self, link: Tag) -> str:
        title = link.get_text(strip=True)

        if not title or len(title) < 3 or title in self.FILTER_TITLES:
            aria_label = link.get("aria-label", "")
            if aria_label and isinstance(aria_label, str) and "Read " in aria_label:
                title = aria_label.replace("Read ", "").strip()
            else:
                parent = link.parent
                for _ in range(3):
                    if parent:
                        title_elem = parent.select_one(
                            'h1, h2, h3, h4, h5, h6, .title, [class*="title"]'
                        )
                        if title_elem:
                            title = title_elem.get_text(strip=True)
                            break
                        parent = parent.parent
                    else:
                        break

        return title

    def _should_filter_title(self, title: str) -> bool:
        if not title:
            return True

        title_lower = title.lower().strip()

        if len(title_lower) < 5:
            return True

        if title_lower in self.FILTER_TITLES:
            return True

        return title_lower.replace(".", "").replace(" ", "") == ""

    def _find_date_near_element(self, element: Tag) -> str:
        current = element
        for _ in range(5):
            if current is None:
                break

            current_text = str(current)
            for pattern in self.DATE_PATTERNS:
                if match := pattern.search(current_text):
                    return match.group(1)

            if hasattr(current, "next_siblings"):
                for sibling in current.next_siblings:
                    if isinstance(sibling, Tag):
                        sibling_text = str(sibling)
                        for pattern in self.DATE_PATTERNS:
                            if match := pattern.search(sibling_text):
                                return match.group(1)

            current = current.parent

        return ""


class QwenBlogScraper(GenericPageScraper):
    ITEM_SELECTOR: ClassVar[str] = "a[href*='/blog/']"

    def _parse_item(self, item: Tag) -> feedparser.FeedParserDict | None:
        href = item.get("href")
        if not href:
            return None

        if isinstance(href, list):
            if not href:
                return None
            href = href[0]

        if href.endswith("/blog/") or href == "/blog" or "/page/" in href:
            return None

        title = item.get_text(strip=True)
        if not title or len(title) < 5:
            parent = item.parent
            for _ in range(3):
                if parent and (heading := parent.select_one("h1, h2, h3, h4, h5, h6")):
                    title = heading.get_text(strip=True)
                    break
                parent = parent.parent if parent else None

        if not title or len(title) < 5:
            return None

        title_lower = title.lower().replace("\xa0", " ").strip()
        nav_keywords = [
            "«",
            "»",
            "continue",
            "more",
            "next",
            "page",
            "prev",
            "previous",
        ]
        if any(keyword in title_lower for keyword in nav_keywords):
            return None

        title = title.replace("| Qwen", "").replace("- Qwen", "").strip()

        page_text = str(item.parent) if item.parent else ""
        date_match = re.search(r"\b([A-Z][a-z]+ \d{1,2}, \d{4})\b", page_text)
        if not date_match:
            # Fail closed: without a parseable date we cannot place the post in a
            # weekly window, so drop it rather than stamping it with "now" (which
            # would force every undated Qwen post into the current digest).
            return None

        return feedparser.FeedParserDict(
            title=title.strip(),
            link=urljoin(self.page_url, href) if href.startswith("/") else href,
            published=date_match.group(1),
            source=self.source,
        )


class XAIBlogScraper(BasePageScraper):
    def fetch(self, start_date: datetime, end_date: datetime) -> list[Post]:
        logger.info(
            f"XAI: Fetching posts from '{start_date.date()}' to '{end_date.date()}'"
        )

        soup = self._fetch_page()

        posts = []
        seen_urls = set()

        news_links = soup.find_all("a", href=re.compile(r"/news/[^/]+/?$"))

        for link in news_links:
            try:
                href = link.get("href")
                if not href or href in seen_urls or href.endswith("/news/"):
                    continue

                seen_urls.add(href)

                title = self._extract_title(link)
                if not title or len(title) < 5:
                    continue

                date_text = self._find_date_near_element(link)
                if not date_text:
                    continue

                pub_date = try_parse_published_date(date_text)
                if pub_date is None or not is_date_in_range(
                    pub_date, start_date, end_date
                ):
                    continue

                # urljoin resolves relative/absolute hrefs against page_url, so
                # the host stays a single source of truth (the configured URL).
                full_url = urljoin(self.page_url, href)

                post = Post.from_entry(
                    feedparser.FeedParserDict(
                        title=title.strip(),
                        link=full_url,
                        published=pub_date.isoformat(),
                        source="xai",
                    )
                )
                posts.append(post)
                logger.info(f"XAI: Added '{title}' ({pub_date.date()})")

            except Exception as e:
                logger.error(f"XAI: Error processing link: {e}")

        return posts

    @staticmethod
    def _extract_title(link: Tag) -> str:
        title = link.get_text(strip=True)

        if not title or len(title) < 5:
            parent = link.parent
            for _ in range(3):
                if parent:
                    heading = parent.select_one("h1, h2, h3, h4, h5, h6")
                    if heading:
                        title = heading.get_text(strip=True)
                        break
                    parent = parent.parent
                else:
                    break

        return title

    @staticmethod
    def _find_date_near_element(element: Tag) -> str:
        date_patterns = [
            re.compile(r"\b([A-Z][a-z]+ \d{1,2}, \d{4})\b"),
            re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b"),
            re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"),
        ]

        current = element
        for _ in range(4):
            if current:
                element_text = str(current)
                for pattern in date_patterns:
                    if match := pattern.search(element_text):
                        return match.group(1)
                current = current.parent
            else:
                break

        return ""


class ScraperRegistry:
    _SCRAPER_MAPPING: ClassVar[dict[str, tuple[type[BasePageScraper], SourceType]]] = {
        AppConstants.External.ANTHROPIC_ENGINEERING.value: (
            AnthropicBlogScraper,
            "anthropic",
        ),
        AppConstants.External.GOOGLE_RESEARCH.value: (GoogleBlogScraper, "google"),
        AppConstants.External.LINKEDIN_ENGINEERING.value: (
            LinkedInBlogScraper,
            "linkedin",
        ),
        AppConstants.External.META_AI.value: (MetaAIBlogScraper, "meta"),
        AppConstants.External.QWEN.value: (QwenBlogScraper, "qwen"),
        AppConstants.External.XAI.value: (XAIBlogScraper, "xai"),
        # NOTE: add new scrapers here
    }

    @classmethod
    def get_fetcher(cls, url: str) -> PostFetcher:
        for url_pattern, (scraper_class, source) in cls._SCRAPER_MAPPING.items():
            if url_pattern in url:
                logger.info("Using %s for URL: '%s'", scraper_class.__name__, url)
                return scraper_class(page_url=url, source=source)
        return RssFetcher(url)

    @classmethod
    def create_fetchers(cls, urls: list[str]) -> list[PostFetcher]:
        return [cls.get_fetcher(url) for url in urls]


class SourceStatus(str, Enum):
    """Health status of a single crawl source for one run."""

    OK = "ok"  # fetched successfully and produced posts in range
    EMPTY = "empty"  # fetched successfully but no posts in the date window
    FAILED = "failed"  # the fetch itself errored (network, parse, anti-bot)


@dataclass
class SourceHealth:
    url: str
    fetcher: str
    status: SourceStatus
    post_count: int = 0
    error: str | None = None

    @property
    def is_failure(self) -> bool:
        return self.status is SourceStatus.FAILED


@dataclass
class CrawlReport:
    """Aggregate health of all sources for a single crawl run."""

    sources: list[SourceHealth] = field(default_factory=list)

    @property
    def failed(self) -> list[SourceHealth]:
        return [s for s in self.sources if s.status is SourceStatus.FAILED]

    @property
    def empty(self) -> list[SourceHealth]:
        return [s for s in self.sources if s.status is SourceStatus.EMPTY]

    @property
    def ok(self) -> list[SourceHealth]:
        return [s for s in self.sources if s.status is SourceStatus.OK]

    @property
    def total_posts(self) -> int:
        return sum(s.post_count for s in self.sources)

    def summary_line(self) -> str:
        return (
            f"{len(self.ok)} ok, {len(self.empty)} empty, "
            f"{len(self.failed)} failed ({self.total_posts} posts)"
        )

    def format_alert(self) -> str:
        """Human-readable health report for SNS/email notification."""
        lines = [f"Crawl source health: {self.summary_line()}", ""]
        if self.failed:
            lines.append("FAILED sources (need attention):")
            lines.extend(
                f"  - {s.url} [{s.fetcher}]: {s.error or 'unknown error'}"
                for s in self.failed
            )
            lines.append("")
        if self.empty:
            lines.append("Empty sources (no posts in window — may be stale):")
            lines.extend(f"  - {s.url} [{s.fetcher}]" for s in self.empty)
            lines.append("")
        lines.append("Healthy sources:")
        lines.extend(f"  - {s.url}: {s.post_count} posts" for s in self.ok)
        return "\n".join(lines)


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
        for fetcher in self.fetchers:
            url = getattr(fetcher, "source_url", type(fetcher).__name__)
            fetcher_name = type(fetcher).__name__
            try:
                fetched = fetcher.fetch(start_date, end_date)
                new_posts = [p for p in fetched if p.link not in seen_links]
                for post in new_posts:
                    all_posts.append(post)
                    seen_links.add(post.link)
                status = SourceStatus.OK if fetched else SourceStatus.EMPTY
                self.report.sources.append(
                    SourceHealth(
                        url=url,
                        fetcher=fetcher_name,
                        status=status,
                        post_count=len(fetched),
                    )
                )
                if not fetched:
                    logger.warning("Source produced no posts in range: '%s'", url)
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
        return sorted(all_posts, key=lambda p: p.published_date, reverse=True)

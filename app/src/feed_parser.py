# NOTE: If there is no RSS feed available, you need to create a custom scraper here

import re
import socket
from contextlib import contextmanager
from datetime import datetime, timezone
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
    REQUEST_HEADERS_OPTIONS: ClassVar[list[dict[str, str]]] = [
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,ko;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
        },
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
        },
        {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
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
        "developer.nvidia.com": "nvidia",
        "openai.com": "openai",
        "blog.palantir.com": "palantir",
        "pinterest-engineering": "pinterest",
        "qwenlm.github.io": "qwen",
        "x.ai": "xai",
        # NOTE: add new sources here
    }


def _make_robust_request(url: str) -> requests.Response | None:
    session = requests.Session()
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()

    cached_index = HeaderCache.get_cached_header_index(domain)
    if cached_index is not None:
        headers = ScraperConfig.REQUEST_HEADERS_OPTIONS[cached_index]
        try:
            session.headers.update(headers)
            response = session.get(
                url,
                timeout=ScraperConfig.REQUEST_TIMEOUT,
                allow_redirects=True,
                verify=True,
            )
            response.raise_for_status()
            logger.info(
                f"Successfully fetched '{url}' with cached headers (index {cached_index})"
            )
            return response
        except RequestException as e:
            logger.warning(
                f"Cached headers (index {cached_index}) failed for '{url}': {e}"
            )

    for i, headers in enumerate(ScraperConfig.REQUEST_HEADERS_OPTIONS):
        if i == cached_index:
            continue
        try:
            session.headers.update(headers)
            response = session.get(
                url,
                timeout=ScraperConfig.REQUEST_TIMEOUT,
                allow_redirects=True,
                verify=True,
            )
            response.raise_for_status()
            HeaderCache.cache_header_index(domain, i)
            logger.info(f"Successfully fetched '{url}' with headers set {i + 1}")
            return response
        except RequestException as e:
            logger.warning(f"Headers set {i + 1} failed for '{url}': {e}")
    logger.error(f"All header attempts failed for '{url}'")
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
        unique_tags = sorted(list(set(tag for tag in v if isinstance(tag, str))))
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
        if len(content) < ScraperConfig.MIN_CONTENT_LENGTH and (
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
            if path_parts:
                source = ScraperConfig.SOURCE_MAPPING.get(path_parts[0].lstrip("@"))
                if source is not None:
                    return source
        return ScraperConfig.SOURCE_MAPPING.get(domain, "unknown")

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
        target_date.astimezone(timezone.utc)
        if target_date.tzinfo
        else target_date.replace(tzinfo=timezone.utc)
    )
    return (
        start_date.astimezone(timezone.utc)
        <= target_utc
        <= end_date.astimezone(timezone.utc)
    )


def parse_published_date(date_str: str) -> datetime:
    if not isinstance(date_str, str) or not date_str:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00")).astimezone(
            timezone.utc
        )
    except (ValueError, TypeError):
        pass
    for fmt in ScraperConfig.DATE_FORMATS:
        try:
            dt = datetime.strptime(date_str, fmt)
            return (
                dt.astimezone(timezone.utc)
                if dt.tzinfo
                else dt.replace(tzinfo=timezone.utc)
            )
        except (ValueError, TypeError):
            continue
    logger.warning(f"Failed to parse date '{date_str}' with any known format.")
    return datetime.now(timezone.utc)


class PostFetcher(Protocol):
    def fetch(self, start_date: datetime, end_date: datetime) -> list[Post]:
        ...


class RssFetcher:
    def __init__(self, rss_url: str):
        self.rss_url = rss_url

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
                    if not is_encoding_error:
                        return []

                for entry in feed.entries:
                    published_date = parse_published_date(
                        getattr(entry, "published", "")
                    )
                    if is_date_in_range(published_date, start_date, end_date):
                        posts.append(Post.from_entry(entry))

        except Exception as e:
            logger.error(f"Failed to fetch or process feed '{self.rss_url}': {e}")
            return []

        return posts


class BasePageScraper:
    def __init__(self, page_url: str, source: SourceType):
        self.page_url = page_url
        self.source = source

    def _fetch_page(self) -> BeautifulSoup | None:
        if response := _make_robust_request(self.page_url):
            return BeautifulSoup(response.text, "html.parser")
        return None

    def fetch(self, start_date: datetime, end_date: datetime) -> list[Post]:
        raise NotImplementedError


class GenericPageScraper(BasePageScraper):
    ITEM_SELECTOR: ClassVar[str] = ""

    def fetch(self, start_date: datetime, end_date: datetime) -> list[Post]:
        soup = self._fetch_page()
        if not soup:
            return []
        posts = []
        for item in soup.select(self.ITEM_SELECTOR):
            try:
                if not (entry_data := self._parse_item(item)):
                    continue
                published_value = entry_data.get("published", "")
                published_str = (
                    published_value if isinstance(published_value, str) else ""
                )
                pub_date = parse_published_date(published_str)
                if is_date_in_range(pub_date, start_date, end_date):
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
        if not soup:
            return []

        posts = []
        seen_urls = set()

        for link in soup.find_all("a", href=re.compile(r"/engineering/")):
            if post := self._extract_post_from_link(link, start_date, end_date):
                if post.link not in seen_urls:
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

            pub_date = parse_published_date(date_text)
            if not is_date_in_range(pub_date, start_date, end_date):
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
        if not soup:
            logger.error("Meta AI: Failed to fetch main page")
            return []

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

                pub_date = parse_published_date(date_text)
                if not is_date_in_range(pub_date, start_date, end_date):
                    logger.debug(
                        f"Meta AI: '{title}' date {pub_date.date()} outside range"
                    )
                    continue

                full_url = (
                    f"https://ai.meta.com{href}" if href.startswith("/") else href
                )

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

        if title_lower.replace(".", "").replace(" ", "") == "":
            return True

        return False

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

        date_text = ""
        page_text = str(item.parent) if item.parent else ""
        if date_match := re.search(r"\b([A-Z][a-z]+ \d{1,2}, \d{4})\b", page_text):
            date_text = date_match.group(1)

        return feedparser.FeedParserDict(
            title=title.strip(),
            link=urljoin(self.page_url, href) if href.startswith("/") else href,
            published=date_text or datetime.now(timezone.utc).isoformat(),
            source=self.source,
        )


class XAIBlogScraper(BasePageScraper):
    def fetch(self, start_date: datetime, end_date: datetime) -> list[Post]:
        logger.info(
            f"XAI: Fetching posts from '{start_date.date()}' to '{end_date.date()}'"
        )

        soup = self._fetch_page()
        if not soup:
            return []

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

                pub_date = parse_published_date(date_text)
                if not is_date_in_range(pub_date, start_date, end_date):
                    continue

                full_url = f"https://x.ai{href}" if href.startswith("/") else href

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
    _SCRAPER_MAPPING: dict[str, tuple[type[BasePageScraper], SourceType]] = {
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


class PostCollector:
    def __init__(self, fetchers: list[PostFetcher]):
        self.fetchers = fetchers

    @classmethod
    def from_urls(cls, urls: list[str]) -> "PostCollector":
        return cls(ScraperRegistry.create_fetchers(urls))

    def collect_posts(self, start_date: datetime, end_date: datetime) -> list[Post]:
        logger.info(
            f"Collecting posts from '{start_date.date()}' to '{end_date.date()}'"
        )
        all_posts = []
        seen_links = set()
        for fetcher in self.fetchers:
            try:
                for post in fetcher.fetch(start_date, end_date):
                    if post.link not in seen_links:
                        all_posts.append(post)
                        seen_links.add(post.link)
            except Exception as e:
                logger.error(f"Fetcher '{type(fetcher).__name__}' failed: {e}")
        return sorted(all_posts, key=lambda p: p.published_date, reverse=True)

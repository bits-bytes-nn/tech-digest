# NOTE: If there is no RSS feed available, you need to create a custom scraper here

import ipaddress
import re
import socket
import time
from collections.abc import Iterator
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

from .constants import DEFAULT_TAGS, MAX_TAGS, AppConstants
from .logger import logger

# Cap on redirects we will follow per request. Bounds redirect-chain abuse and
# keeps a misbehaving source from looping us.
MAX_REDIRECTS: int = 5

# Only these URL schemes may appear in a rendered link (article link, resource
# links). The summary/link fields ultimately originate from scraped/feed
# content and are rendered into the outbound email, so an unchecked scheme like
# ``javascript:`` or ``data:`` would be a phishing / injection vector. Shared
# by the summarizer's link sanitizer and the Post/Article link validators so
# there is a single allow-list.
SAFE_URL_SCHEMES: frozenset[str] = frozenset({"http", "https", "mailto"})


def is_safe_url(url: str) -> bool:
    """True only for absolute URLs whose scheme is explicitly allow-listed.

    Rejects ``javascript:``, ``data:``, ``vbscript:``, scheme-relative
    ``//evil`` and bare/relative strings — none of which belong in a trusted
    outbound link.
    """
    try:
        scheme = urlparse(url.strip()).scheme.lower()
    except (ValueError, AttributeError):
        return False
    return scheme in SAFE_URL_SCHEMES


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """True if ``ip`` falls in an SSRF-sensitive range (private / loopback /
    link-local / reserved / multicast / unspecified) — notably the cloud
    metadata endpoint 169.254.169.254."""
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _is_blocked_host(host: str) -> bool:
    """True if ``host`` targets an SSRF-sensitive address (private / loopback /
    link-local / reserved range — notably the cloud metadata endpoint
    169.254.169.254).

    Two layers:
      1. IP-literal check (cheap): canonical dotted-quad / bracketed-IPv6, plus
         the decimal / hex / octal integer forms of an IPv4 address (e.g.
         ``2130706433``, ``0x7f000001``, ``017700000001`` all == 127.0.0.1) via
         ``inet_aton`` — the OS resolver accepts those, so without this they
         would slip past a plain ``ip_address`` parse.
      2. DNS resolution: a *hostname* is resolved and every returned A/AAAA
         address is checked, so a name that resolves to an internal address
         (e.g. attacker-controlled DNS pointing at 169.254.169.254) is blocked
         too. This is validated again on every redirect hop in ``_try_request``.

    This is not a full DNS-rebinding defense (the address that ``requests``
    ultimately connects to could differ from the one resolved here), but it
    closes the practical gap — a source that links or redirects to an internal
    hostname — well beyond the previous IP-literal-only guard.
    """
    if not host:
        return True
    try:
        return _is_blocked_ip(ipaddress.ip_address(host))
    except ValueError:
        pass
    # Not canonical IP text. Try the integer IPv4 encodings inet_aton accepts
    # (decimal/hex/octal) before treating it as a name to resolve.
    try:
        packed = socket.inet_aton(host)
    except OSError:
        pass
    else:
        return _is_blocked_ip(ipaddress.ip_address(packed))
    # A genuine hostname: resolve it and block if ANY resolved address is
    # sensitive. Resolution failure is treated as not-blocked here (the request
    # will simply fail to connect); only a positive hit on an internal address
    # blocks it.
    return _host_resolves_to_blocked(host)


def _host_resolves_to_blocked(host: str) -> bool:
    """Resolve ``host`` and return True if any resolved IP is SSRF-sensitive."""
    try:
        infos = socket.getaddrinfo(host, None)
    except (OSError, UnicodeError):
        return False
    for info in infos:
        sockaddr = info[4]
        try:
            ip = ipaddress.ip_address(sockaddr[0])
        except ValueError:
            continue
        if _is_blocked_ip(ip):
            return True
    return False


# Every member must be a VALUE in ``ScraperConfig.SOURCE_MAPPING`` (or
# "unknown"), because that map is the only thing that produces a source — see
# ``Post._determine_source``. "deepmind" used to be listed here and was
# unreachable: ``deepmind.google`` maps to "google". ``test_feed_parser`` pins
# the two together now.
SourceType: TypeAlias = Literal[
    "airbnb",
    "amazon",
    "anthropic",
    "aws",
    "eugene_yan",
    "google",
    "huggingface",
    "kakao",
    "linkedin",
    "phil_schmid",
    "meta",
    "microsoft",
    "ncsoft",
    "netflix",
    "nvidia",
    "openai",
    "palantir",
    "pinterest",
    "qwen",
    "sebastian_raschka",
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
    # Visible-text length below which a feed entry is considered a teaser and we
    # go scrape the full article page. Distinct from ``scraping.min_content_length``
    # (the configurable ~600-char gate that decides whether a post is substantive
    # enough to summarize at all) — the two were previously both called
    # "min content length", which invited confusing one for the other.
    FULL_SCRAPE_TEXT_THRESHOLD: ClassVar[int] = 3000
    MARKDOWN_IMAGE_PATTERN: ClassVar[re.Pattern] = re.compile(r"!\[.*?]\((.*?)\)")
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
        # Independent authors. These three are configured feeds, but were absent
        # from this map, so all of them resolved to "unknown". That cost more than
        # a generic logo: `max_per_source` counts by source, so three different
        # authors were capped as if they were one publication, and any newly added
        # unmapped feed would join the same bucket. Underscores become spaces in
        # ``Article.source_label``, so these render as proper names.
        "eugeneyan.com": "eugene_yan",
        "sebastianraschka.com": "sebastian_raschka",
        "magazine.sebastianraschka.com": "sebastian_raschka",
        "philschmid.de": "phil_schmid",
        # NOTE: add new sources here
    }


class SourceFetchError(Exception):
    """Raised when a source cannot be fetched at all (network error, anti-bot
    block, HTTP error) — as opposed to fetching successfully but finding no
    posts. The collector uses this to mark a source FAILED vs. EMPTY."""


def visible_text(html: str) -> str:
    """The readable prose in an HTML string, markup stripped.

    Single source of truth for "what a reader actually sees", used by the
    scrape decision, the content-sufficiency gate and the relevance filter so
    all three agree on what counts as content.
    """
    if not html:
        return ""
    try:
        return BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)
    except Exception:
        return html


def _visible_text_length(html: str) -> int:
    """Number of visible-text characters in an HTML string (markup stripped)."""
    return len(visible_text(html)) if html else 0


# Elements that carry site furniture rather than article content. ``noscript`` is
# deliberately ABSENT: lazy-loading pages put the real <img src> there and leave a
# ``data:image/svg+xml`` placeholder in the visible markup, so dropping it would
# throw away exactly the figures the summary is supposed to include (verified on
# NVIDIA's blog, where 7 of 15 images live in noscript).
NON_CONTENT_TAGS: frozenset[str] = frozenset(
    {"script", "style", "svg", "iframe", "form", "nav", "footer", "template"}
)
# Attributes worth keeping. Everything else — class hooks, data-*, inline styles,
# tracking ids — is markup the model has no use for. ``class`` survives because
# the summarization prompt asks for ``<pre><code class="highlight">`` blocks.
CONTENT_ATTRIBUTES: frozenset[str] = frozenset(
    {"src", "href", "alt", "srcset", "colspan", "rowspan", "class"}
)


def content_html(html: str) -> str:
    """``html`` with site furniture and decorative attributes removed.

    The summarizer needs HTML, not prose: code blocks, tables and image URLs all
    live in the markup, which is why the filter's visible-text shortcut cannot be
    reused here. But most of that markup is not the article — it is navigation,
    scripts, inline styles and class hooks.

    Measured over one issue's three articles with the Bedrock token counter,
    dropping it cuts summarization input from 85,947 to 53,954 tokens (-37%) while
    leaving every ``<pre>``, ``<table>`` and article ``<img>`` in place. Less noise
    around the article is also one less thing for the model to mistake for
    content — a related-posts carousel reads a lot like a list of findings.
    """
    if not html:
        return ""
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(list(NON_CONTENT_TAGS)):
            tag.decompose()
        for tag in soup.find_all(True):
            tag.attrs = {
                key: value
                for key, value in tag.attrs.items()
                if key in CONTENT_ATTRIBUTES
            }
        return str(soup)
    except Exception as e:
        # Never let input shaping cost us the post: fall back to the raw HTML,
        # which is what every run before this used.
        logger.warning("Could not clean content HTML, using it as-is: %s", e)
        return html


def _get_following_redirects(
    session: requests.Session, url: str, headers: dict[str, str]
) -> requests.Response | None:
    """GET ``url``, following up to ``MAX_REDIRECTS`` redirects manually and
    SSRF-validating every hop's host before connecting. Returns the final
    response, or None if any hop targets a blocked host or the chain is too
    long. Raises RequestException (via session.get) so the caller's retry
    logic still applies to transient errors.

    ``headers`` is passed per-request rather than written onto the session, so
    each header set in the fallback ladder is sent exactly as declared.
    """
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        host = urlparse(current).hostname or ""
        if _is_blocked_host(host):
            logger.warning("Refusing request to blocked host: '%s'", current)
            return None
        response = session.get(
            current,
            headers=headers,
            timeout=ScraperConfig.REQUEST_TIMEOUT,
            allow_redirects=False,
            verify=True,
        )
        if response.is_redirect and response.headers.get("location"):
            current = urljoin(current, response.headers["location"])
            continue
        return response
    logger.warning("Too many redirects for '%s'; refusing.", url)
    return None


def _try_request(
    session: requests.Session, url: str, headers: dict[str, str]
) -> requests.Response | None:
    """Single GET attempt with one retry for transient errors (timeout / 5xx /
    429). Returns the response on success, or None if this header set fails.

    ``headers`` is applied per-request. It used to be written onto the shared
    session with ``session.headers.update``, which made the sets in the fallback
    ladder cumulative instead of alternative: after the Chrome set (12 headers)
    failed, the Safari set (5 headers) was merged on top of it, so the retry went
    out with a macOS Safari ``User-Agent`` alongside Chrome's
    ``Sec-Ch-Ua: "Google Chrome";v="131"`` and ``Sec-Ch-Ua-Platform: "Windows"``.
    Safari sends no client hints at all, so that combination cannot occur in a
    real browser — and an impossible fingerprint is precisely what the anti-bot
    filters this ladder exists to get past are looking for. Every attempt after
    the first was therefore *less* likely to succeed than the first.
    """
    session.max_redirects = MAX_REDIRECTS
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            # Follow redirects MANUALLY so every hop's host is SSRF-validated
            # before we connect to it — allow_redirects=True would connect to an
            # internal redirect target before we ever saw its URL. Each Location
            # is re-checked with _is_blocked_host (which now also resolves
            # hostnames), closing the "source redirects to the metadata endpoint"
            # gap for intermediate hops, not just the final landing URL.
            response = _get_following_redirects(session, url, headers)
            if response is None:
                return None
            response.raise_for_status()
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
    # Single-sentence lede produced alongside the summary; rendered above the
    # article body so a multi-article digest can be skimmed.
    one_liner: str = ""
    tags: list[str] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)
    score: float = 0.0

    @field_validator("tags", mode="before")
    @classmethod
    def validate_tags(cls, v):
        if not v:
            return list(DEFAULT_TAGS)
        # Preserve the source ordering (tags are emitted most-significant-first)
        # while de-duplicating, then cap. dict.fromkeys keeps first-seen order;
        # sorting here would discard relevance and drop important trailing tags.
        unique_tags = list(dict.fromkeys(tag for tag in v if isinstance(tag, str)))
        return unique_tags[:MAX_TAGS]

    @classmethod
    def from_entry(cls, entry: feedparser.FeedParserDict) -> "Post":
        """Build a Post from a feed entry or a scraper-built ``FeedParserDict``.

        ``source`` is always DERIVED from the link via ``SOURCE_MAPPING``, never
        read off the entry: that map is the single source of truth, and a
        per-scraper source would be a second copy of it that could drift. The
        index-page scrapers used to pass ``source=`` here and it was silently
        ignored — one of them had drifted to a hardcoded literal without anyone
        noticing, which is what a duplicate nobody reads looks like.
        """
        link_str = getattr(entry, "link", "")
        content = cls._extract_content_from_entry(entry, link_str)
        return cls(
            title=getattr(entry, "title", "No Title").strip(),
            link=link_str,
            published_date=parse_published_date(getattr(entry, "published", "")),
            source=cls._determine_source(link_str),
            content=content,
            # ``summary`` is deliberately NOT seeded from the feed's teaser.
            # It is the slot the LLM summary lands in, and downstream code uses
            # "summary is non-empty" as the signal that summarization SUCCEEDED
            # (Summarizer.process_posts, Article.summary min_length=1). Seeding
            # it with the RSS teaser made that signal always true, so a post
            # whose summarization failed shipped its raw feed teaser as the
            # article body while ALSO being reported as filtered-out.
            images=cls._extract_images(content, link_str),
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
        feed_text_len = _visible_text_length(content)
        # Only adopt the scrape if it actually has MORE readable prose than the
        # feed content. A JS-rendered/consent-walled page can yield a thin body
        # that, if it blindly replaced a decent feed teaser, would push the post
        # below the downstream content gate and silently drop a healthy post.
        if (
            feed_text_len < ScraperConfig.FULL_SCRAPE_TEXT_THRESHOLD
            and (scraped_content := cls._scrape_full_content(link))
            and _visible_text_length(scraped_content) > feed_text_len
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
            # Resolve markdown image URLs against base_url too (same as <img
            # src> above), so relative paths like ![](/img/x.png) aren't dropped
            # by the http filter below.
            image_urls.update(
                urljoin(base_url, m)
                for m in ScraperConfig.MARKDOWN_IMAGE_PATTERN.findall(content_html)
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
    #: Items this fetcher last offered BEFORE date filtering. Set by ``fetch``; see
    #: ``SourceHealth.candidates`` for why the collector records it.
    candidates: int | None

    def fetch(self, start_date: datetime, end_date: datetime) -> list[Post]: ...


class RssFetcher:
    def __init__(self, rss_url: str):
        self.rss_url = rss_url
        self.source_url = rss_url
        self.candidates: int | None = None

    def fetch(self, start_date: datetime, end_date: datetime) -> list[Post]:
        logger.info(f"Fetching posts from RSS feed: '{self.rss_url}'")
        # Reset first: a stale count from a previous call would describe the wrong
        # fetch if this collector is reused.
        self.candidates = None
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

                # Entries the feed offered, before the date window. A feed that
                # parses cleanly but carries none at all has moved or broken; a
                # quiet blog still lists its back catalogue.
                self.candidates = len(feed.entries)

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
    # Regex matching the hrefs of THIS site's post links. Index-page scrapers set
    # it so date attribution can tell "my post's date" from "a neighbour's date".
    LINK_PATTERN: ClassVar[re.Pattern | None] = None
    # Date formats to look for near a post link, most specific first.
    DATE_PATTERNS: ClassVar[tuple[re.Pattern, ...]] = ()
    # How far up the tree to look before giving up.
    DATE_SEARCH_DEPTH: ClassVar[int] = 5
    # How far up the tree to look for a heading when the card has none inside.
    TITLE_ANCESTOR_DEPTH: ClassVar[int] = 3
    # A candidate longer than this is not a title. Index pages that nest a whole
    # card inside the <a> concatenate heading, date and teaser into one string;
    # preferring the heading is the real defense and this is only a backstop, so
    # the bound is generous — the longest title ever published here is 103 chars.
    TITLE_MAX_CHARS: ClassVar[int] = 250
    # Words that make up navigation chrome ("Read more", "Next Page"). Used to
    # reject a link whose text contains NOTHING BUT these — see ``is_nav_text``.
    NAV_WORDS: ClassVar[frozenset[str]] = frozenset(
        {
            "all",
            "archive",
            "back",
            "blog",
            "continue",
            "featured",
            "first",
            "home",
            "index",
            "last",
            "latest",
            "learn",
            "more",
            "new",
            "newer",
            "news",
            "next",
            "older",
            "page",
            "pages",
            "post",
            "posts",
            "prev",
            "previous",
            "read",
            "reading",
            "see",
            "start",
            "view",
        }
    )
    # Function words: they neither identify chrome nor count as subject matter,
    # so "The Latest" is still chrome while "Read more About Llama" is not.
    TITLE_STOPWORDS: ClassVar[frozenset[str]] = frozenset(
        {"a", "an", "and", "at", "by", "for", "in", "of", "on", "the", "to", "with"}
    )

    def is_nav_text(self, text: str) -> bool:
        """True when ``text`` is navigation chrome rather than a post title.

        The rule is "nothing here names a subject": tokenize, then reject only if
        NO token is substantive (outside ``NAV_WORDS`` and ``TITLE_STOPWORDS``).

        Rejecting on the mere PRESENCE of a nav word — which the Qwen scraper did
        before this moved to the base class — throws away real posts, because
        those words also occur in real titles: "Next Generation Reasoning Models",
        "More Efficient Attention", "Page-Level Retrieval" were all dropped. An
        earlier version matched substrings rather than words, which additionally
        lost "Preventing..." (contains "prev") and "PageRank" (contains "page").
        """
        lowered = text.replace("\xa0", " ").casefold()
        # Pagination arrows carry no word characters, so they need their own test.
        if "«" in lowered or "»" in lowered:
            return True
        words = re.findall(r"[^\W\d_]+", lowered, flags=re.UNICODE)
        if not words:
            return True
        return not any(
            word not in self.NAV_WORDS and word not in self.TITLE_STOPWORDS
            for word in words
        )

    def _extract_title(self, link: Tag) -> str:
        """The post title for an index-page link, or "" if none is credible.

        Returning "" rather than a best guess is deliberate: every caller skips
        the link, which is the same fail-closed stance the date gate takes.

        The three index scrapers each had their own version of this, and they
        disagreed on both strategy and thresholds. The x.ai one was fixed to
        prefer the heading inside the card after it emitted titles like
        "Grok 4.6Aug 12, 2026Introducing Grok 4.6Aug 12, 2026Introducing..."; the
        Meta one still read the card's full text first and so still had that bug,
        and the Anthropic one filtered candidates with English content tests
        (``startswith("This is")``, ``"blog post" not in ...``) that meant nothing
        on any other site. One implementation, ordered by how reliably each signal
        names the post, is both shorter and less wrong.
        """
        for candidate in self._title_candidates(link):
            title = " ".join(candidate.split())
            if not title or len(title) > self.TITLE_MAX_CHARS:
                continue
            if self.is_nav_text(title):
                continue
            return title
        return ""

    def _title_candidates(self, link: Tag) -> Iterator[str]:
        """Title candidates for ``link``, most trustworthy first."""
        # A heading inside the card: the site's own markup saying "this is the
        # title", and the only signal that survives a card with a nested teaser.
        for heading in link.select("h1, h2, h3, h4, h5, h6"):
            yield heading.get_text(strip=True)
        # An element the site labels as the title by class name.
        for element in link.select(
            '[class*="title"], [class*="heading"], [class*="headline"]'
        ):
            yield element.get_text(strip=True)
        # The link's own text, before the accessible name — see below for why.
        yield link.get_text(strip=True)
        # A heading just outside the link, for markup that puts the <a> inside
        # the heading's sibling rather than the other way round.
        parent = link.parent
        for _ in range(self.TITLE_ANCESTOR_DEPTH):
            if parent is None:
                break
            if heading := parent.select_one("h1, h2, h3, h4, h5, h6"):
                yield heading.get_text(strip=True)
            parent = parent.parent
        # The accessible name, last. It is the only signal an image-only card
        # link has, but it is written for screen readers and so normally carries a
        # call-to-action in front of the title, which is why the visible text is
        # preferred when there is any.
        for attribute in ("aria-label", "title"):
            value = link.get(attribute)
            if isinstance(value, str):
                yield self._strip_call_to_action(value)

    # Leading call-to-action in an accessible name, e.g. Meta's "Read <title>"
    # and Qwen's "post link to <title>". Matched only at the START, so unlike a
    # content blacklist it cannot reject or truncate a real title — "Reading
    # Comprehension in LLMs" keeps its first word because "reading" is not
    # followed by more text here, it IS the text.
    TITLE_CALL_TO_ACTION: ClassVar[re.Pattern] = re.compile(
        r"^\s*(?:read|view|see|open|go\s+to|learn\s+more\s+about|"
        r"(?:blog\s+)?post\s+link\s+to)\s+(?=\S)",
        re.IGNORECASE,
    )

    def _strip_call_to_action(self, accessible_name: str) -> str:
        return self.TITLE_CALL_TO_ACTION.sub("", accessible_name, count=1)

    def __init__(self, page_url: str):
        self.page_url = page_url
        self.source_url = page_url
        self.candidates: int | None = None

    @property
    def link_pattern(self) -> re.Pattern:
        """``LINK_PATTERN``, asserted present.

        The base class leaves it optional because feed-based fetchers have no
        index page to scan; every index-page scraper must set it.
        """
        if self.LINK_PATTERN is None:
            raise NotImplementedError(
                f"{type(self).__name__} must set LINK_PATTERN to find its posts"
            )
        return self.LINK_PATTERN

    def _find_date_near_element(self, element: Tag) -> str | None:
        """The publication date belonging to ``element``, or None.

        Walks from the link outward. The subtle part is what to do once the walk
        reaches an ancestor holding SEVERAL posts: a plain regex over that
        ancestor returns the first date in document order, which belongs to
        whichever post happens to come first. An undated post therefore inherited
        its neighbour's date and slipped into the weekly window with a date the
        site never gave it — defeating the fail-closed date gate that
        ``try_parse_published_date`` exists to provide.

        So at every level the OTHER post links are removed before searching. A
        date that survives is either inside our own link or in shared chrome (a
        "Published May 30, 2026" section heading legitimately covers every post
        under it), and both of those are correctly ours. A date that only existed
        inside a sibling post is not ours, and we keep looking or give up.
        """
        current: Tag | None = element
        for _ in range(self.DATE_SEARCH_DEPTH):
            if current is None:
                break
            if found := self._search_dates(self._without_other_posts(current, element)):
                return found
            current = current.parent
        return None

    def _without_other_posts(self, scope: Tag, keep: Tag) -> str:
        """``scope``'s HTML with every post link other than ``keep`` removed."""
        if self.LINK_PATTERN is None or scope is keep:
            return str(scope)
        # Work on a copy: the live tree is reused for the remaining posts.
        clone = BeautifulSoup(str(scope), "html.parser")
        keep_html = str(keep)
        removed_self = False
        for link in clone.find_all("a", href=self.LINK_PATTERN):
            if not removed_self and str(link) == keep_html:
                removed_self = True  # this is us; leave our own date in place
                continue
            link.decompose()
        return str(clone)

    def _search_dates(self, html: str) -> str | None:
        for pattern in self.DATE_PATTERNS:
            if match := pattern.search(html):
                return match.group(1)
        return None

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
        items = soup.select(self.ITEM_SELECTOR)
        # Zero matches means the selector no longer finds this site's cards, which
        # is indistinguishable from "no posts this week" without this count.
        self.candidates = len(items)
        for item in items:
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
    LINK_PATTERN: ClassVar[re.Pattern | None] = re.compile(r"/engineering/")
    DATE_PATTERNS: ClassVar[tuple[re.Pattern, ...]] = (
        re.compile(r"\b([A-Z][a-z]{2,8} \d{1,2}, \d{4})\b"),
    )

    def fetch(self, start_date: datetime, end_date: datetime) -> list[Post]:
        logger.info(
            f"Anthropic: Fetching posts from '{start_date.date()}' to '{end_date.date()}'"
        )

        soup = self._fetch_page()

        posts = []
        seen_urls = set()

        links = soup.find_all("a", href=self.link_pattern)
        self.candidates = len(links)

        for link in links:
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
                )
            )

        except Exception as e:
            logger.error(f"Error processing link: {e}")
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
            # Resolve relative hrefs against the page URL, matching every other
            # scraper. A bare "/blog/engineering/x" would otherwise become the
            # Post link verbatim, resolving source to "unknown" and making the
            # full-content scrape refuse an empty host. urljoin is a no-op on
            # already-absolute URLs.
            link=urljoin(self.page_url, str(href)),
            published=date_elem.text.strip(),
            tags=[{"term": tag} for tag in tags],
        )


class MetaAIBlogScraper(BasePageScraper):
    LINK_PATTERN: ClassVar[re.Pattern | None] = re.compile(r"/blog/[^/]+/?$")
    DATE_PATTERNS: ClassVar[tuple[re.Pattern, ...]] = (
        re.compile(r"\b([A-Z][a-z]{2,8} \d{1,2}, \d{4})\b"),
        re.compile(r"\b([A-Z][a-z]{2} \d{1,2}, \d{4})\b"),
        re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b"),
        re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"),
    )

    def fetch(self, start_date: datetime, end_date: datetime) -> list[Post]:
        logger.info(
            f"Meta AI: Fetching posts from '{start_date.date()}' to '{end_date.date()}'"
        )

        soup = self._fetch_page()

        posts = []
        # Deduplicate on the RESOLVED url, not the raw href: an index page can
        # link the same post both relatively ("/blog/x") and absolutely, which a
        # raw-href set treats as two posts. This used to be a raw-href set
        # followed by a second, resolved-url pass over the finished list — two
        # passes to express one rule.
        seen_urls: set[str] = set()
        blog_links = soup.find_all("a", href=self.link_pattern)
        # Already logged; now also reported, so a selector that stops matching
        # shows up in the crawl-health alert rather than only in the run log.
        self.candidates = len(blog_links)
        logger.info(f"Meta AI: Found {len(blog_links)} potential blog links")

        for i, link in enumerate(blog_links):
            try:
                href = link.get("href", "")
                if not href:
                    continue

                if href.endswith("/blog/") or "?page=" in href:
                    continue

                # urljoin handles both relative ("/blog/x") and absolute hrefs
                # against the scraper's own page_url, so the host is never
                # hardcoded here (single source of truth: the configured URL).
                full_url = urljoin(self.page_url, str(href))
                if full_url in seen_urls:
                    continue
                seen_urls.add(full_url)

                title = self._extract_title(link)
                if not title:
                    logger.debug(f"Meta AI: no usable title for link {i}")
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

                post = Post.from_entry(
                    feedparser.FeedParserDict(
                        title=title.strip(),
                        link=full_url,
                        published=pub_date.isoformat(),
                    )
                )
                posts.append(post)
                logger.info(f"Meta AI: Added '{title}' ({pub_date.date()})")

            except Exception as e:
                logger.error(f"Meta AI: Error processing link {i}: {e}")

        logger.info(f"Meta AI: Found {len(posts)} unique posts in date range")
        return posts


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

        # Shared extractor: heading first, then labelled elements, then the
        # link's own text — and it applies the shared nav-chrome test, so the
        # per-scraper nav word list this used to carry is gone.
        title = self._extract_title(item)
        if not title:
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
        )


class XAIBlogScraper(BasePageScraper):
    LINK_PATTERN: ClassVar[re.Pattern | None] = re.compile(r"/news/[^/]+/?$")
    DATE_PATTERNS: ClassVar[tuple[re.Pattern, ...]] = (
        re.compile(r"\b([A-Z][a-z]+ \d{1,2}, \d{4})\b"),
        re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b"),
        re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"),
    )
    DATE_SEARCH_DEPTH: ClassVar[int] = 4

    def fetch(self, start_date: datetime, end_date: datetime) -> list[Post]:
        logger.info(
            f"XAI: Fetching posts from '{start_date.date()}' to '{end_date.date()}'"
        )

        soup = self._fetch_page()

        posts = []
        seen_urls = set()

        news_links = soup.find_all("a", href=self.link_pattern)
        self.candidates = len(news_links)

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
                    )
                )
                posts.append(post)
                logger.info(f"XAI: Added '{title}' ({pub_date.date()})")

            except Exception as e:
                logger.error(f"XAI: Error processing link: {e}")

        return posts


class ScraperRegistry:
    # URL fragment -> scraper. No source is recorded alongside the class: the
    # source comes from ``SOURCE_MAPPING`` via the post's link, so pairing one
    # here would be a second copy of that map. The entries used to be
    # ``(class, source)`` tuples whose source was threaded down and then
    # discarded by ``Post.from_entry``.
    _SCRAPER_MAPPING: ClassVar[dict[str, type[BasePageScraper]]] = {
        AppConstants.External.ANTHROPIC_ENGINEERING.value: AnthropicBlogScraper,
        AppConstants.External.GOOGLE_RESEARCH.value: GoogleBlogScraper,
        AppConstants.External.LINKEDIN_ENGINEERING.value: LinkedInBlogScraper,
        AppConstants.External.META_AI.value: MetaAIBlogScraper,
        AppConstants.External.QWEN.value: QwenBlogScraper,
        AppConstants.External.XAI.value: XAIBlogScraper,
        # NOTE: add new scrapers here
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
    #: Posts this source contributed to the digest, after cross-source dedup.
    post_count: int = 0
    #: Posts it returned that a previous source had already provided.
    duplicates: int = 0
    #: Items the source offered BEFORE the date-window filter — feed entries for
    #: a feed, index-page links matching the scraper's LINK_PATTERN for a scraper.
    #:
    #: This is what separates the two reasons a source can be EMPTY, which the
    #: report used to render identically: a blog that published nothing this week
    #: offers its back catalogue and the window filters it out, while a moved feed
    #: or a scraper whose selector no longer matches offers NOTHING to filter. One
    #: is normal and one needs a person. Both showed up as one indistinguishable
    #: line among the ~12 empty sources of a typical run, so a broken crawler could
    #: sit there for weeks.
    #:
    #: ``None`` means the fetcher did not report a count — unknown, NOT zero.
    candidates: int | None = None
    error: str | None = None


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
    def likely_broken(self) -> list[SourceHealth]:
        """EMPTY sources that offered NOTHING for the date window to filter.

        A feed that moved, or a scraper whose selector stopped matching: the fetch
        itself succeeded, so the source is not FAILED, but there was no material at
        any date. Distinct from a blog that was simply quiet, which offers its back
        catalogue and has the window filter it out.

        Deliberately NOT a fourth ``SourceStatus``. "Fetched fine, produced no
        posts in range" is an accurate status and stays one; what is new is the
        evidence that separates the two reasons for it, so the inference lives here
        instead of being baked into a state every caller has to handle.

        ``candidates is None`` (a fetcher that reports no count) is excluded —
        unknown is not zero, and guessing here would page on the unknown case.
        """
        return [s for s in self.empty if s.candidates == 0]

    @property
    def total_posts(self) -> int:
        return sum(s.post_count for s in self.sources)

    def summary_line(self) -> str:
        broken = len(self.likely_broken)
        empty = f"{len(self.empty)} empty"
        if broken:
            empty += f" ({broken} with nothing to filter)"
        return (
            f"{len(self.ok)} ok, {empty}, "
            f"{len(self.failed)} failed ({self.total_posts} posts)"
        )

    @staticmethod
    def _empty_reason(source: SourceHealth) -> str:
        """Why this source produced nothing, in the reader's terms."""
        if source.candidates is None:
            return "no count reported"
        if source.candidates == 0:
            return "offered 0 items — LIKELY BROKEN (moved feed or stale selector)"
        return f"offered {source.candidates} item(s), none in the date window"

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
            lines.append("Empty sources (no posts in window):")
            lines.extend(
                f"  - {s.url} [{s.fetcher}]: {self._empty_reason(s)}"
                for s in self.empty
            )
            lines.append("")
        lines.append("Healthy sources:")
        lines.extend(
            f"  - {s.url}: {s.post_count} posts"
            + (f" ({s.duplicates} deduped)" if s.duplicates else "")
            for s in self.ok
        )
        return "\n".join(lines)


def normalize_title_key(title: str) -> str:
    """Canonical key for exact-duplicate detection across sources.

    Several configured feeds carry the SAME announcement under different URLs
    (deepmind.google and research.google both map to source ``google``; an AWS
    post is also syndicated to Medium). Link-only dedup let those through as two
    identical cards in one digest.

    This is deliberately an EXACT match after normalization — case, whitespace
    and punctuation are folded, nothing else. No fuzzy/similarity scoring, so it
    can never merge two genuinely different articles.
    """
    folded = re.sub(r"[^\w\s]", " ", title.casefold())
    return " ".join(folded.split())


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
                        # Posts this source CONTRIBUTED, not posts it returned.
                        # Reporting the fetch count let a source whose every post
                        # was a syndicated duplicate of an earlier one appear as
                        # "3 posts" while adding nothing to the digest, so the
                        # health line overstated how much material the crawl
                        # actually had. ``duplicates`` keeps the difference
                        # visible instead of hiding it.
                        post_count=contributed,
                        duplicates=len(fetched) - contributed,
                    )
                )
                if not fetched:
                    logger.warning("Source produced no posts in range: '%s'", url)
                elif not contributed:
                    logger.warning(
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
        return sorted(all_posts, key=lambda p: p.published_date, reverse=True)

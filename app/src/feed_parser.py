# NOTE: If there is no RSS feed available, you need to create a custom scraper here

import ipaddress
import re
import socket
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from html import unescape
from typing import ClassVar, Protocol, TypeAlias
from urllib.parse import urljoin, urlparse

import feedparser
import requests
from bs4 import BeautifulSoup, Tag
from feedparser.exceptions import CharacterEncodingOverride
from pydantic import BaseModel, Field, field_validator
from requests.exceptions import RequestException

from .constants import DEFAULT_TAGS, MAX_TAGS
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


# A value of ``ScraperConfig.SOURCE_MAPPING``, or "unknown". The map is the only
# thing that produces a source, so it is the list; a Literal restating it was a
# second copy that needed a test to keep the two in step.
SourceType: TypeAlias = str


class ScraperConfig:
    # Measured across every configured host: only these two ever held the
    # article; anything else falls back to <body> via the share rule below.
    CONTENT_SELECTORS: ClassVar[list[str]] = ["article", "main"]
    # Share of a page's visible text a CONTENT_SELECTORS match must hold before it
    # is preferred over the whole <body>. See ``Post._scrape_full_content``.
    CONTENT_CONTAINER_MIN_SHARE: ClassVar[float] = 0.5
    # Tried after ``datetime.fromisoformat``, which already covers every ISO
    # shape: RFC 822 for feeds, "May 29, 2026" for index pages.
    DATE_FORMATS: ClassVar[tuple[str, ...]] = (
        "%a, %d %b %Y %H:%M:%S GMT",
        "%a, %d %b %Y %H:%M:%S %z",
        "%b %d, %Y",
        "%B %d, %Y",
    )
    # Visible-text length below which a feed entry is a teaser and the article
    # page is fetched instead. Not ``scraping.min_content_length``, the ~600-char
    # gate on whether a post is worth summarizing at all.
    FULL_SCRAPE_TEXT_THRESHOLD: ClassVar[int] = 3000
    MARKDOWN_IMAGE_PATTERN: ClassVar[re.Pattern] = re.compile(r"!\[.*?]\((.*?)\)")
    # A full, realistic browser header set. The Sec-Fetch-* / Sec-Ch-Ua and
    # Upgrade-Insecure-Requests headers reduce 403s from anti-bot filters that
    # reject bare User-Agent requests. One set: over 90 days of Batch logs, no
    # second or third set ever rescued a request this one had lost.
    REQUEST_HEADERS: ClassVar[dict[str, str]] = {
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
    }
    REQUEST_TIMEOUT: ClassVar[int] = 30
    SOURCE_MAPPING: ClassVar[dict[str, SourceType]] = {
        "airbnb-engineering": "airbnb",
        "amazon.science": "amazon",
        "aws.amazon.com": "aws",
        "deepmind.google": "google",
        "research.google": "google",
        "huggingface.co": "huggingface",
        "linkedin.com": "linkedin",
        "engineering.fb.com": "meta",
        "microsoft.com": "microsoft",
        "netflixtechblog.com": "netflix",
        "netflix-techblog": "netflix",  # Medium publication slug
        "developer.nvidia.com": "nvidia",
        "openai.com": "openai",
        "blog.palantir.com": "palantir",
        "palantir": "palantir",  # Medium @-handle
        "pinterest-engineering": "pinterest",
        "pinterest_engineering": "pinterest",  # Medium @-handle
        "qwenlm.github.io": "qwen",
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
    """The readable prose in an HTML string, site furniture and markup stripped.

    Single source of truth for "what a reader actually sees", used by the
    scrape decision, the content-sufficiency gate and the relevance filter so
    all three agree on what counts as content — and agree with the summarizer,
    which receives ``content_html`` and so never sees ``NON_CONTENT_TAGS``.

    Counting those tags would let a JavaScript-rendered page's menus and script
    data clear the gate while the summarizer receives little more than a title.
    """
    if not html:
        return ""
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(list(NON_CONTENT_TAGS)):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)
    except Exception:
        return html


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
        # Never let input shaping cost the post: fall back to the raw HTML.
        logger.warning("Could not clean content HTML, using it as-is: %s", e)
        return html


def parse_page(response: requests.Response) -> BeautifulSoup:
    """Parse a fetched page, decoding it with the charset the page declares.

    Without a charset in Content-Type, ``requests`` assumes ISO-8859-1 for text/*,
    and ``response.text`` turned every UTF-8 page from deepmind.google and
    sebastianraschka.com into mojibake ("Xiaomiâs") before the model saw it. Given
    bytes, BeautifulSoup reads the page's own <meta charset> instead.
    """
    if "charset=" in response.headers.get("content-type", "").lower():
        return BeautifulSoup(response.text, "html.parser")
    return BeautifulSoup(response.content, "html.parser")


def _get_following_redirects(
    session: requests.Session, url: str, headers: dict[str, str]
) -> requests.Response | None:
    """GET ``url``, following up to ``MAX_REDIRECTS`` redirects manually and
    SSRF-validating every hop's host before connecting. Returns the final
    response, or None if any hop targets a blocked host or the chain is too
    long. Raises RequestException (via session.get) so the caller's retry
    logic still applies to transient errors.

    ``headers`` is passed per request rather than written onto the session.
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
    """GET with up to two retries for transient errors (timeout / 5xx / 429).
    Returns the response on success, or None on failure.

    ``headers`` is passed per request, never written onto the session, so what
    is sent is exactly what the caller declared.
    """
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
    """GET an article or index page with browser headers, or None if it fails.

    SSRF-checked up front and at every redirect hop; transient errors retried
    (see ``_try_request``).
    """
    if _is_blocked_host(urlparse(url).hostname or ""):
        logger.warning("Refusing request to blocked host: '%s'", url)
        return None
    if response := _try_request(requests.Session(), url, ScraperConfig.REQUEST_HEADERS):
        return response
    logger.error("Request failed for '%s'", url)
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
        per-scraper source would be a second copy of it that could drift.
        """
        link_str = getattr(entry, "link", "")
        content = cls._extract_content_from_entry(entry, link_str)
        return cls(
            # Some feeds escape the title twice (Amazon Science sends
            # "don&amp;#8217;t"), so feedparser hands back "don&#8217;t" and the
            # card showed the entity literally. Decoding is idempotent on a
            # title that was already plain text.
            title=unescape(getattr(entry, "title", "No Title")).strip(),
            link=link_str,
            published_date=parse_published_date(entry_date_text(entry)),
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
        feed_text_len = len(visible_text(content))
        # Only adopt the scrape if it actually has MORE readable prose than the
        # feed content. A JS-rendered/consent-walled page can yield a thin body
        # that, if it blindly replaced a decent feed teaser, would push the post
        # below the downstream content gate and silently drop a healthy post.
        if (
            feed_text_len < ScraperConfig.FULL_SCRAPE_TEXT_THRESHOLD
            and (scraped_content := cls._scrape_full_content(link))
            and len(visible_text(scraped_content)) > feed_text_len
        ):
            content = scraped_content
        return content

    @staticmethod
    def _scrape_full_content(url: str) -> str:
        """The article's HTML from its own page, or "" if it cannot be fetched.

        Picks the candidate container that actually HOLDS the prose, rather than
        the first selector that happens to match. The old version returned
        ``select_one(selector)`` for the earliest matching selector without looking
        inside it, which made the ``<body>`` fallback unreachable whenever any
        selector matched — even an empty one.

        Measured on Hugging Face's blog, where ``<article>`` is a near-empty
        wrapper: it returned 82 visible characters while the same page carried
        21,669 (and 93 against 64,950 on another post). Those posts then died at
        the content-sufficiency gate, so a healthy source silently contributed
        nothing — 21 of 26 collected posts were dropped that way in one run.
        """
        if not url or not (response := _make_robust_request(url)):
            return ""
        try:
            soup = parse_page(response)
            body = soup.find("body")
            # get_text on the parsed element, not visible_text on its string: the
            # latter would re-parse every candidate.
            body_length = len(body.get_text(separator=" ", strip=True)) if body else 0
            best: Tag | None = None
            best_length = 0
            for selector in ScraperConfig.CONTENT_SELECTORS:
                for element in soup.select(selector):
                    length = len(element.get_text(separator=" ", strip=True))
                    if length > best_length:
                        best, best_length = element, length
            # A narrower container is preferable when it keeps the article, because
            # it excludes nav/footer chrome. It is only worth trusting if it holds
            # most of the page's prose: the HF wrapper held 0.4%, while a site whose
            # <article> really is the article holds 60-90%. Half is nowhere near
            # either case, so the cut needs no tuning to separate them.
            if best is not None and best_length >= body_length * (
                ScraperConfig.CONTENT_CONTAINER_MIN_SHARE
            ):
                return str(best)
            return str(body or "")
        except Exception as e:
            logger.error(f"Error parsing content for '{url}': {e}")
            return ""

    @staticmethod
    def _determine_source(url: str) -> SourceType:
        if not url:
            return "unknown"
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower().removeprefix("www.")
        if domain == "medium.com":
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
        return len(visible_text(self.content))

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
    """``start <= target < end``: half-open, so a post dated exactly on a
    boundary belongs to one weekly window, not two (see ``get_date_range``)."""
    target_utc = (
        target_date.astimezone(UTC)
        if target_date.tzinfo
        else target_date.replace(tzinfo=UTC)
    )
    return start_date.astimezone(UTC) <= target_utc < end_date.astimezone(UTC)


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
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        # An offset-less ISO date is UTC, as in the strptime branch below.
        # ``astimezone`` would read it as the container's local time instead, and
        # the Batch image sets TZ=Asia/Seoul: "2026-10-04" became 10-03 15:00 UTC
        # and fell out of a window starting on 10-04.
        return dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)
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


def entry_date_text(entry: feedparser.FeedParserDict) -> str:
    """The publication-date text of a feed entry, or "" if it carries none.

    Reads ``published`` first and falls back to ``updated``. Atom makes
    ``<published>`` OPTIONAL while ``<updated>`` is REQUIRED, and feedparser maps
    the two to distinct attributes — so a perfectly valid Atom feed that carries
    only ``<updated>`` has no ``entry.published`` at all. Reading only
    ``published`` meant every entry of such a feed failed the fail-closed date gate
    and was dropped, permanently and silently: the source still reported
    candidates, so it read as a merely quiet blog rather than a broken one (§7's
    discriminator cannot see this case either).

    None of the configured feeds is shaped that way today — all 19 were checked,
    and every entry of every one supplies a parseable ``published`` — so this
    closes a trap on the "add new sources here" path rather than recovering posts
    already lost.

    ``published`` wins when both exist. ``updated`` is a MODIFICATION time, so
    preferring it would let an old post edited this week re-enter the weekly
    window. When ``published`` is absent, ``updated`` is the only date the feed
    offers, and using it beats dropping the post.
    """
    return getattr(entry, "published", "") or getattr(entry, "updated", "")


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
    FEED_HEADERS: ClassVar[dict[str, str]] = {
        "User-Agent": feedparser.USER_AGENT,
        "Accept": "application/xml, application/rss+xml",
    }

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

                skipped = 0
                for entry, published_date in zip(feed.entries, dates, strict=True):
                    if published_date is None or not is_date_in_range(
                        published_date, start_date, end_date
                    ):
                        continue
                    # One malformed entry (e.g. a link urlparse rejects) must
                    # cost that entry, not every valid post in the feed.
                    try:
                        posts.append(Post.from_entry(entry))
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

    def _fetch_feed(self) -> feedparser.FeedParserDict:
        """Download the feed through the request layer, then parse the bytes.

        ``feedparser.parse(url)`` fetches with urllib and sets no socket timeout,
        so a feed host that accepted the connection and then stalled held the job
        until the 3-hour Batch timeout — and Batch then retried it twice. The
        request layer adds the timeout, the transient-error retry and the per-hop
        SSRF check every other fetch already had. The headers are the ones
        feedparser sent itself, so feed hosts see the same client as before.
        """
        response = _try_request(requests.Session(), self.rss_url, self.FEED_HEADERS)
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
        if response := _make_robust_request(self.page_url):
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
                    posts.append(Post.from_entry(entry_data))
            except Exception as e:
                logger.error(f"Error processing item from {self.page_url}: {e}")
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
        if not (response := _make_robust_request(url)):
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
    #: Items with a readable date the source offered BEFORE the date window.
    #: It separates the two reasons a source can be EMPTY: a quiet blog offers a
    #: back catalogue the window filters out; a moved feed or a broken scraper
    #: offers nothing. Only the second needs a person (``likely_broken``).
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
        if self.report.likely_broken:
            # Named here, not only in the SNS alert. The alert carries the URLs but
            # is suppressed for expected-flaky sources and depends on a live SNS
            # subscription, so without this the run log knew the COUNT and not which
            # sources — which is exactly the question the first real run raised.
            logger.warning(
                "%d source(s) offered nothing for the date window to filter — a "
                "moved feed or a stale selector rather than a quiet blog: %s",
                len(self.report.likely_broken),
                [s.url for s in self.report.likely_broken],
            )
        return sorted(all_posts, key=lambda p: p.published_date, reverse=True)

"""The article a crawl produces, and the URL and date rules it is held to."""

import re
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import TypeAlias
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, field_validator

from app.domain.constants import DEFAULT_TAGS, MAX_TAGS
from app.domain.content import visible_text
from app.logger import logger

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


# A value of ``SOURCE_MAPPING``, or "unknown". The map is the only
# thing that produces a source, so it is the list.
SourceType: TypeAlias = str

SOURCE_MAPPING: dict[str, SourceType] = {
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
    "blog.palantir.com": "palantir",
    "palantir": "palantir",  # Medium @-handle
    "palantir.medium.com": "palantir",  # the same feed links here too
    "pinterest-engineering": "pinterest",
    "pinterest_engineering": "pinterest",  # Medium @-handle
    "qwenlm.github.io": "qwen",
    # Independent authors. Each needs its own entry: ``max_per_source``
    # counts by source, so unmapped feeds share one "unknown" cap.
    # Underscores become spaces in ``Article.source_label``.
    "eugeneyan.com": "eugene_yan",
    "sebastianraschka.com": "sebastian_raschka",
    "magazine.sebastianraschka.com": "sebastian_raschka",
    "philschmid.de": "phil_schmid",
    # NOTE: add new sources here
}

# Tried after ``datetime.fromisoformat``, which already covers every ISO
# shape: RFC 822 for feeds, "May 29, 2026" for index pages.
DATE_FORMATS: tuple[str, ...] = (
    "%a, %d %b %Y %H:%M:%S GMT",
    "%a, %d %b %Y %H:%M:%S %z",
    "%b %d, %Y",
    "%B %d, %Y",
)

MARKDOWN_IMAGE_PATTERN: re.Pattern = re.compile(r"!\[.*?]\((.*?)\)")


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
    # The feed carried only a teaser and the article page could not be read
    # (blocked, failed or unparseable). Crawl health counts these per source; it
    # is run state, not part of the saved post.
    page_unreadable: bool = Field(default=False, exclude=True)

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
                source = SOURCE_MAPPING.get(handle)
                if source is not None:
                    return source
        return SOURCE_MAPPING.get(domain, "unknown")

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
            # ``src`` is single-valued, so str() only narrows the stub's type.
            image_urls: set[str] = {
                urljoin(base_url, str(img["src"]))
                for img in soup.find_all("img")
                if img.get("src")
            }
            # Resolve markdown image URLs against base_url too (same as <img
            # src> above), so relative paths like ![](/img/x.png) aren't dropped
            # by the http filter below.
            image_urls.update(
                urljoin(base_url, m)
                for m in MARKDOWN_IMAGE_PATTERN.findall(content_html)
            )
            return [url for url in image_urls if url.startswith("http")]
        except Exception as e:
            logger.warning(f"Failed to extract images from '{base_url}': {e}")
            return []


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
    try:
        # RFC 822, the RSS date format, including the named zones ("PDT", "EST")
        # that strptime's %z rejects.
        dt = parsedate_to_datetime(date_str)
        return dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)
    except (ValueError, TypeError, IndexError):
        pass
    for fmt in DATE_FORMATS:
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


def get_date_range(
    end_date_str: str | None, days_back: int
) -> tuple[datetime, datetime]:
    """The half-open window ``[start, end)`` an issue dated ``end`` covers.

    ``end`` is midnight UTC at the START of the issue date (today when no date is
    given), and the issue covers the ``days_back`` full days before it. Gates
    compare with ``start <= t < end`` (see ``is_date_in_range``), so the windows
    of consecutive weekly runs abut exactly, whatever hour the job runs at.

    Ending the window at the run hour or at 23:59 instead would leave posts
    published on a run day after the job ran in no issue at all.
    """
    if end_date_str:
        day = datetime.strptime(end_date_str, "%Y-%m-%d").replace(tzinfo=UTC)
    else:
        day = datetime.now(UTC)
    end_date = day.replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=days_back)
    return start_date, end_date


def normalize_title_key(title: str) -> str:
    """Canonical key for exact-duplicate detection across sources.

    Several configured feeds carry the SAME announcement under different URLs
    (deepmind.google and research.google both map to source ``google``; an AWS
    post is also syndicated to Medium), which link-only dedup would show as two
    identical cards.

    This is deliberately an EXACT match after normalization — case, whitespace
    and punctuation are folded, nothing else. No fuzzy/similarity scoring, so it
    can never merge two genuinely different articles.
    """
    folded = re.sub(r"[^\w\s]", " ", title.casefold())
    return " ".join(folded.split())

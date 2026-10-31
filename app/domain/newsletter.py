"""The issue as data: its articles, chrome and reader-facing labels."""

import re
from datetime import UTC, datetime
from typing import Any

from bs4 import BeautifulSoup
from markupsafe import escape
from pydantic import BaseModel, Field, computed_field, field_validator

from app.domain.constants import Language
from app.domain.post import is_safe_url, try_parse_published_date

# Gmail (and several other clients) truncate a message body larger than this,
# showing a "view entire message" link and hiding everything past the cut —
# including the trailing article cards and the footer. Measured against real
# published issues, every 5-article digest exceeded it, so the build warns
# loudly rather than letting the tail silently disappear from readers' inboxes.
GMAIL_CLIP_BYTES: int = 102_400

# Reading speed used for the per-card "N min read" estimate, per language.
# Korean is counted in characters and English in words; both are deliberately
# conservative for dense technical prose.
# Source keys whose display name is not their title case (logo alt text).
SOURCE_LABELS: dict[str, str] = {"aws": "AWS"}
READING_SPEED: dict[Language, tuple[str, int]] = {
    Language.KO: ("chars", 450),
    Language.EN: ("words", 200),
}

# Reader-facing chrome, per language. Previously hardcoded English in the
# templates, which left a Korean digest with "Published on" / "View More" /
# "Additional resources for reference" labels around Korean content.
UI_LABELS: dict[Language, dict[str, str]] = {
    Language.KO: {
        "published_on": "발행",
        "view_more": "원문 보기",
        "resources": "함께 볼 자료",
        "reading_time": "읽는 시간 {minutes}분",
        "relevance": "관련도 점수",
    },
    Language.EN: {
        "published_on": "Published on",
        "view_more": "Read the original",
        "resources": "Further reading",
        "reading_time": "{minutes} min read",
        "relevance": "Relevance score",
    },
}

_WHITESPACE_RUN_WITH_NEWLINE = re.compile(r"[ \t]*\r?\n[ \t\r\n]*")
_HORIZONTAL_WHITESPACE_RUN = re.compile(r"[ \t]{2,}")
_PRE_BLOCK = re.compile(r"(<pre\b[^>]*>.*?</pre>)", re.DOTALL | re.IGNORECASE)


def collapse_html_whitespace(html: str) -> str:
    """Shrink the template's pretty-printing whitespace without changing layout.

    Table-based email markup is deeply indented, which costs ~10% of the message
    size in pure indentation — bytes that count against the clip threshold above.

    Every whitespace run is collapsed to a SINGLE space rather than removed.
    Removing it would be a few percent smaller but could delete a meaningful
    inter-word space between two inline tags in the model-generated prose
    (``<em>A</em>\\n<em>B</em>``), so the safe collapse is used instead.
    ``<pre>`` blocks are left byte-for-byte intact, since whitespace there IS
    the content.
    """
    parts = _PRE_BLOCK.split(html)
    for i, part in enumerate(parts):
        if i % 2:  # odd indices are the captured <pre>...</pre> blocks
            continue
        part = _WHITESPACE_RUN_WITH_NEWLINE.sub(" ", part)
        parts[i] = _HORIZONTAL_WHITESPACE_RUN.sub(" ", part)
    return "".join(parts)


def estimate_reading_minutes(html: str, language: Language) -> int:
    """Minutes to read an article card's rendered summary (minimum 1)."""
    text = BeautifulSoup(html or "", "html.parser").get_text(separator=" ", strip=True)
    unit, per_minute = READING_SPEED.get(language, READING_SPEED[Language.EN])
    amount = len(text) if unit == "chars" else len(text.split())
    return max(1, round(amount / per_minute)) if amount else 1


def normalize_date(v: str) -> str:
    """A date string as ``YYYY-MM-DD`` in UTC, falling back to today (UTC).

    The same parser the crawler gates on, so a date that got a post into the
    issue always renders; UTC because the Batch image runs with TZ=Asia/Seoul.
    """
    parsed = try_parse_published_date(v if isinstance(v, str) else str(v))
    return (parsed or datetime.now(UTC)).strftime("%Y-%m-%d")


class Article(BaseModel):
    title: str = Field(min_length=1)
    link: str
    published_date: str
    thumbnail: str
    summary: str = Field(min_length=1)
    # Single-sentence lede shown above the body so the digest can be skimmed.
    # Rendered through Jinja autoescape (not ``| safe``), so it needs no
    # sanitization here.
    one_liner: str = ""
    reading_minutes: int = Field(default=0, ge=0)
    source: str = "unknown"
    tags: list[str] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    _normalize_date = field_validator("published_date", mode="before")(normalize_date)

    @field_validator("link", mode="before")
    @classmethod
    def _validate_link_scheme(cls, v: Any) -> str:
        # article.link is rendered into an href in the outbound email. It
        # originates from scraped/feed content, so a javascript:/data: scheme
        # would be a phishing/injection vector in clients that don't strip it.
        # Blank an unsafe scheme rather than raise, so one bad link doesn't drop
        # the whole article (the template then leaves out the CTA).
        link = str(v) if v is not None else ""
        return link if is_safe_url(link) else ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def source_label(self) -> str:
        """Human-friendly source name for alt text / accessibility. Exposed as a
        computed field so it appears in ``model_dump()`` and the templates can
        use it directly (single source of truth instead of duplicating the
        replace/title filter chain in each template).

        The logo's alt text is the only place a screen reader learns the
        source. A key of three letters or fewer is an acronym ("aws" -> "AWS",
        which "Aws" read aloud as a word); longer keys are title-cased.
        """
        if self.source == "unknown":
            return ""
        return SOURCE_LABELS.get(self.source) or self.source.replace("_", " ").title()


class Footer(BaseModel):
    title: str


class Header(BaseModel):
    title: str = Field(min_length=1)
    description: str
    thumbnail: str
    publish_date: str
    _normalize_publish_date = field_validator("publish_date", mode="before")(
        normalize_date
    )


class Section(BaseModel):
    introduction: str

    @field_validator("introduction", mode="before")
    @classmethod
    def _escape_introduction(cls, v: Any) -> str:
        # The greeting is LLM-generated plain text rendered into the template
        # with Jinja `| safe`, so it is a trust boundary just like the summary
        # HTML (see summary._sanitize_html). The greeting prompt asks for
        # plain text, so HTML-escaping is a no-op for legitimate output while
        # neutralizing any injected <script>/markup echoed from scraped titles.
        return str(escape(v)) if v is not None else ""


class NewsletterData(BaseModel):
    header: Header
    section: Section
    articles: list[Article]
    footer: Footer
    # The template read ``data.language`` to set <html lang="...">, but the field
    # never existed, so every issue — including English ones — silently fell back
    # to the ``| default('ko')`` filter. Screen readers and client-side
    # translation both key off that attribute.
    language: Language = Language.KO

    @computed_field  # type: ignore[prop-decorator]
    @property
    def labels(self) -> dict[str, str]:
        """Reader-facing chrome in the issue's language (single source of truth
        for both the newsletter and the standalone-article templates)."""
        return UI_LABELS.get(self.language, UI_LABELS[Language.EN])


class BuildConfiguration(BaseModel):
    stage: str = "dev"
    # UTC, like the rest of the pipeline. A naive now() here read as Asia/Seoul
    # in the Batch image, so the default could name tomorrow's date.
    date_suffix: str = Field(
        default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%d")
    )
    language: Language = Language.KO
    # Used when the stage config leaves a chrome field unset. The intro has no
    # default: main always supplies the greeting or its fixed fallback.
    header_title: str = "Weekly AI Tech Blog Digest"
    header_description: str = (
        "A curated digest of the most impactful AI developments, research and "
        "engineering from leading tech companies and research institutions."
    )
    header_thumbnail: str = "peccy.png"
    first_section_intro: str
    footer_title: str = (
        "Thank you for reading! Stay tuned for next week's AI insights and discoveries."
    )
    save_individual_articles: bool = False
    convert_to_images: bool = False

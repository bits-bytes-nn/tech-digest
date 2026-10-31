"""What makes a model-written summary safe and fit to ship, and which posts ship.

The summary HTML is rendered with Jinja ``| safe``, so everything here that
cleans it is a trust boundary. No model call happens in this module: the
summarizer service hands it the model's answer.
"""

import re
from html import escape, unescape
from typing import Any, ClassVar, Literal
from urllib.parse import quote

import markdown
from bs4 import (
    BeautifulSoup,
    CData,
    Comment,
    Declaration,
    Doctype,
    NavigableString,
    ProcessingInstruction,
)
from pydantic import BaseModel, Field, field_validator

from app.domain.constants import (
    DEFAULT_TAGS,
    MAX_TAGS,
    FilteringCriteria,
    LanguageModelId,
)
from app.domain.post import Post, is_safe_url

# The terminal-colon rule lives in ``quality_metrics`` so the corrector below and
# the rubric that scores it share one pattern and cannot disagree.
from app.domain.quality_metrics import KO_TERMINAL_COLON
from app.logger import logger


class SummarizerConfig:
    # Not codehilite (its markup depends on whether Pygments is installed, and
    # the Batch image does not ship it), toc (the sanitizer strips the
    # heading ids), footnotes (its in-page "#fn:" links fail the scheme check,
    # leaving <a> tags without an href) or def_list (<dl> is not allow-listed,
    # so a term and its definition were unwrapped onto one line of bare text).
    # attr_list stays only so a "{: .x}" suffix is consumed rather than shown;
    # the sanitizer drops the attributes it sets.
    MARKDOWN_EXTENSIONS: ClassVar[list[str]] = [
        "attr_list",
        "fenced_code",
        "nl2br",
        "sane_lists",
        "tables",
    ]
    # The width a summary image may take: the 800px column less the card gutter
    # (20px a side) and the card padding (24px a side) in main_contents.html.
    # Outlook for Windows ignores max-width, so without a width attribute a
    # 2,000px chart pushed the whole column past the window.
    SUMMARY_IMAGE_WIDTH: ClassVar[int] = 712
    # HTML sanitization allow-list for the model-generated summary, which is
    # rendered with Jinja ``| safe`` (autoescape off). Only these structural and
    # inline tags survive; everything else (notably <script>, <style>, <iframe>,
    # event-handler attributes, javascript: URLs) is stripped. The tag set
    # mirrors what the summarization prompt is instructed to emit.
    ALLOWED_HTML_TAGS: ClassVar[frozenset[str]] = frozenset(
        {
            "a",
            "b",
            "blockquote",
            "br",
            "code",
            "col",
            "colgroup",
            "del",
            "div",
            "em",
            "figcaption",
            "figure",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "hr",
            "i",
            "img",
            "li",
            "ol",
            "p",
            "pre",
            "span",
            "strong",
            "sub",
            "sup",
            "table",
            "tbody",
            "td",
            "tfoot",
            "th",
            "thead",
            "tr",
            "ul",
        }
    )
    # Attributes kept per tag; all others are dropped (this removes on*= handlers,
    # style, etc.). href/src additionally pass a scheme check below, and the
    # sanitizer then sets its own class on tables and size on images.
    ALLOWED_HTML_ATTRS: ClassVar[dict[str, frozenset[str]]] = {
        "a": frozenset({"href", "title"}),
        "img": frozenset({"src", "alt", "title"}),
        "td": frozenset({"colspan", "rowspan"}),
        "th": frozenset({"colspan", "rowspan"}),
    }


def _normalize_tags(tags_input: Any) -> list[str]:
    tags = []
    if isinstance(tags_input, str):
        tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()]
    elif isinstance(tags_input, list):
        for item in tags_input:
            if isinstance(item, str):
                tags.extend([tag.strip() for tag in item.split(",") if tag.strip()])
    return tags


_BACKTICK_SPAN = re.compile(r"`([^`\n]+)`")


def _backticks_to_code(summary: str) -> str:
    """Turn Markdown inline code left inside HTML into ``<code>``.

    Markdown is not processed inside raw HTML blocks, so a backtick span the
    model wrote inside a ``<p>`` would reach the reader as literal backticks.
    """
    if "`" not in summary:
        return summary
    soup = BeautifulSoup(summary, "html.parser")
    for text in soup.find_all(string=_BACKTICK_SPAN):
        if text.find_parent(["pre", "code"]):
            continue
        pieces = _BACKTICK_SPAN.split(str(text))
        for i, piece in enumerate(pieces):
            if i % 2:
                code = soup.new_tag("code")
                code.string = piece
                text.insert_before(code)
            elif piece:
                text.insert_before(NavigableString(piece))
        text.extract()
    return str(soup)


def _postprocess_summary(summary: str) -> str:
    """Deterministic corrections to the model's summary HTML.

    The Korean terminal colon (see ``KO_TERMINAL_COLON``) and Markdown backtick
    spans, both unambiguously wrong and cheap to detect. Links are not
    rewritten: a host-alias table cannot know a site's URL layout
    (``magazine.sebastianraschka.com/p/x`` resolves where
    ``sebastianraschka.com/p/x`` 404s).
    """
    return _backticks_to_code(KO_TERMINAL_COLON.sub(r"\1.", summary))


def _safe_anchor(url: str, text: str) -> str:
    """Build an <a> tag with both href and text HTML-escaped."""
    return f'<a href="{escape(url.strip(), quote=True)}">{escape(text.strip())}</a>'


def _normalize_urls(urls_input: Any) -> list[str]:
    # Each entry becomes a sanitized <a> tag; entries whose URL is missing or
    # uses a non-allow-listed scheme are dropped rather than rendered.
    if isinstance(urls_input, str):
        markdown_links = re.findall(r"\[(.*?)]\((.*?)\)", urls_input)
        if markdown_links:
            return [
                _safe_anchor(url, desc)
                for desc, url in markdown_links
                if is_safe_url(url)
            ]
        raw_items = [url.strip() for url in urls_input.split(",") if url.strip()]
    elif isinstance(urls_input, list):
        raw_items = [item.strip() for item in urls_input if isinstance(item, str)]
    else:
        return []

    sanitized: list[str] = []
    for item in raw_items:
        # An item may already be an <a> tag (the prompt asks for HTML links) or a
        # bare URL. Parse out href/text, validate the scheme, re-emit escaped.
        if "<a" in item.lower():
            anchor = BeautifulSoup(item, "html.parser").find("a")
            # bs4's .get may return a list for multi-valued attrs; coerce to str.
            href = str(anchor.get("href", "")) if anchor else ""
            text = anchor.get_text(strip=True) if anchor else ""
            if is_safe_url(href):
                sanitized.append(_safe_anchor(href, text or href))
        elif is_safe_url(item):
            sanitized.append(_safe_anchor(item, item))
    return sanitized


# Prefixes of the filtered-out reasons that mean a model call FAILED, as opposed
# to a verdict on the post. main's "Model Failures" alert matches on these, so
# they are defined once here and every failure reason is built from them, so a
# reworded message cannot silently stop the alert.
FILTER_FAILED = "Filtering failed"
FILTER_UNPARSEABLE = "Filter response unparseable"
SUMMARY_FAILED = "Summarization failed"
SUMMARY_INVALID = "Summary validation failed"
MODEL_FAILURE_PREFIXES: tuple[str, ...] = (
    FILTER_FAILED,
    FILTER_UNPARSEABLE,
    SUMMARY_FAILED,
    SUMMARY_INVALID,
)


def _sanitize_html(html: str) -> str:
    """Strip disallowed tags/attributes from model-generated summary HTML.

    The summary is rendered with Jinja ``| safe``, so it is a trust boundary:
    the LLM (and, transitively, scraped source content it may echo) must not be
    able to inject <script>, event handlers, or javascript:/data: URLs into the
    outbound email. Unknown tags are unwrapped (their text is kept); disallowed
    attributes and unsafe href/src schemes are removed.
    """
    soup = BeautifulSoup(html, "html.parser")
    # Comments are not tags, so the loop below never saw them — and Outlook
    # renders whatever sits inside a conditional comment (<!--[if mso]>...),
    # links, forms and tracking images included. Nothing the reader should see
    # lives in one, so every such node goes.
    for node in soup.find_all(
        string=lambda text: isinstance(
            text, Comment | CData | Declaration | Doctype | ProcessingInstruction
        )
    ):
        node.extract()
    for tag in list(soup.find_all(True)):
        name = tag.name.lower()
        if name not in SummarizerConfig.ALLOWED_HTML_TAGS:
            # Drop script/style content entirely; otherwise keep inner text.
            if name in ("script", "style"):
                tag.decompose()
            else:
                tag.unwrap()
            continue
        allowed = SummarizerConfig.ALLOWED_HTML_ATTRS.get(name, frozenset())
        for attr in list(tag.attrs):
            if attr.lower() not in allowed:
                del tag[attr]
            elif attr.lower() in ("href", "src") and not is_safe_url(
                str(tag.get(attr, ""))
            ):
                # Allow-listed attribute but unsafe scheme — drop the whole tag's
                # link rather than leave a javascript:/data: payload.
                del tag[attr]
        # Force the presentation class on EVERY table, regardless of how the
        # model emitted it (bare ``<table>`` from markdown, or a ``<table ...>``
        # with attributes in raw HTML).
        if name == "table":
            tag["class"] = "table-bordered"
        # Every image gets the same size: the width attribute for Outlook, and a
        # fluid style so narrower screens scale it down.
        elif name == "img":
            if not tag.get("src"):
                # Its src failed the scheme check: an empty 712px box otherwise.
                tag.decompose()
                continue
            width = SummarizerConfig.SUMMARY_IMAGE_WIDTH
            tag["width"] = str(width)
            tag["style"] = f"width:100%;max-width:{width}px;height:auto"
    return str(soup)


_SPACES = re.compile(r"\s")


def _strip_unsourced_images(
    summary_html: str, source_html: str, known_images: list[str]
) -> tuple[str, list[str]]:
    """Remove ``<img>`` tags the source post does not actually contain.

    The model is told to embed only images from the article, but an LLM can
    invent a plausible CDN path — which renders as a broken image (or a stray
    alt string) in the delivered email. Provenance is checked by EXACT evidence,
    never by pattern-guessing the URL: the src must either be one of the images
    our own parser extracted from the post (``Post.images``, already resolved to
    absolute URLs) or appear verbatim in the post HTML the model was given.

    The source is compared with its HTML entities decoded as well as raw: a
    ``srcset`` carries ``...png?w=1200&amp;q=90``, the model correctly writes the
    URL with a plain ``&``, and a raw-only comparison dropped images the article
    really contains (three in one OpenAI post).

    A src that differs from one of ``known_images`` only in which space
    characters it uses is that image, and is restored to the known URL: models
    copy the narrow no-break space in macOS screenshot names ("1.27.14\u202fPM.png")
    as a plain space, which names a file the CDN does not have.

    Returns ``(cleaned_html, dropped_srcs)``.
    """
    if "<img" not in summary_html.lower():
        return summary_html, []
    allowed = set(known_images)
    by_spacing = {_SPACES.sub(" ", url): url for url in known_images}
    decoded_source = unescape(source_html)
    soup = BeautifulSoup(summary_html, "html.parser")
    dropped: list[str] = []
    changed = False
    for img in list(soup.find_all("img")):
        src = str(img.get("src") or "").strip()
        if src and (src in allowed or src in source_html or src in decoded_source):
            continue
        if known := by_spacing.get(_SPACES.sub(" ", src)):
            img["src"] = known
            changed = True
            continue
        dropped.append(src or "(no src)")
        # The whole figure goes: a caption left behind describes nothing.
        (img.find_parent("figure") or img).decompose()
    if not (dropped or changed):
        return summary_html, []
    return str(soup), dropped


# RFC 3986 reserved and unreserved characters plus "%", so an already-encoded
# URL passes through unchanged and only what a URL cannot carry is encoded.
_URL_SAFE_CHARS = ":/?#[]@!$&'()*+,;=%-._~"


def _encode_image_srcs(summary_html: str) -> str:
    """Percent-encode the characters a URL cannot carry raw in ``<img src>``.

    Source URLs are used as published, and some contain them: macOS names
    screenshots "...at 1.27.14\u202fPM.png", so AWS blog images carry a narrow
    no-break space. Left raw, email clients and image proxies do not reliably
    encode it themselves and the image renders broken.
    """
    if "<img" not in summary_html.lower():
        return summary_html
    soup = BeautifulSoup(summary_html, "html.parser")
    changed = False
    for img in soup.find_all("img"):
        src = str(img.get("src") or "")
        encoded = quote(src, safe=_URL_SAFE_CHARS)
        if encoded != src:
            img["src"] = encoded
            changed = True
    return str(soup) if changed else summary_html


# Mail clients show an image at most ~800px wide, so a 2x rendition is enough;
# larger files only slow the message.
MAX_IMAGE_CANDIDATE_WIDTH = 1600


def _image_candidates(src: str, source_html: str) -> list[str]:
    """Other renditions of the image at ``src`` in the article, best first.

    The article's own markup lists them: ``srcset``/``data-srcset`` with widths,
    ``data-src``, and the ``<source>`` siblings of a ``<picture>``. A rendition no
    wider than ``MAX_IMAGE_CANDIDATE_WIDTH`` comes first, widest first; then the
    larger ones, then those of unknown width.
    """
    soup = BeautifulSoup(source_html, "html.parser")
    found: dict[str, int] = {}
    for img in soup.find_all("img"):
        picture = img.find_parent("picture")
        tags = [img, *(picture.find_all("source") if picture else [])]
        urls: dict[str, int] = {}
        for tag in tags:
            for name in ("srcset", "data-srcset"):
                for part in str(tag.get(name) or "").split(","):
                    bits = part.split()
                    if bits:
                        width = bits[1][:-1] if len(bits) > 1 else ""
                        urls[bits[0]] = int(width) if width.isdigit() else 0
            for name in ("src", "data-src"):
                if value := str(tag.get(name) or ""):
                    urls.setdefault(value, 0)
        if src in urls:
            found.update(urls)
    found.pop(src, None)

    def rank(item: tuple[str, int]) -> tuple[int, int]:
        width = item[1]
        if not width:
            return (2, 0)
        return (0, -width) if width <= MAX_IMAGE_CANDIDATE_WIDTH else (1, width)

    return [url for url, _ in sorted(found.items(), key=rank) if is_safe_url(url)]


def _unsourced_urls(urls: list[str], source_html: str) -> list[str]:
    """Reference links whose href does not appear verbatim in the source post.

    Reported (not removed): unlike an image, a slightly-rewritten but real link
    is still useful to the reader, so dropping it would cost more than it saves.
    Surfacing the count makes prompt drift visible in the run log instead of
    silently shipping links the article never contained.
    """
    unsourced = []
    decoded_source = unescape(source_html)
    for anchor in urls:
        href = BeautifulSoup(anchor, "html.parser").find("a")
        url = str(href.get("href", "")) if href else ""
        if url and url not in source_html and url not in decoded_source:
            unsourced.append(url)
    return unsourced


class SummaryOutput(BaseModel):
    summary: str = Field(min_length=1)
    # A single-sentence lede rendered above the sections so the reader can
    # triage a multi-article digest without reading every card.
    one_liner: str = ""
    tags: list[str] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)

    @field_validator("one_liner", mode="before")
    @classmethod
    def _clean_one_liner(cls, v: Any) -> str:
        """Reduce the lede to a single line of plain text.

        It is rendered through Jinja's autoescape (NOT ``| safe``), so markup
        here would show up as literal text — strip any the model emitted, and
        collapse newlines so it cannot break the single-line layout.
        """
        if not isinstance(v, str):
            return ""
        text = " ".join(
            BeautifulSoup(v, "html.parser").get_text(separator=" ", strip=True).split()
        )
        # The lede is a sentence; models drop the period about one time in three.
        return f"{text}." if text and text[-1] not in ".!?" else text

    @field_validator("summary", mode="before")
    def _convert_markdown_to_html(cls, v: str) -> str:
        if not isinstance(v, str):
            return v
        try:
            html = markdown.markdown(
                v,
                extensions=SummarizerConfig.MARKDOWN_EXTENSIONS,
                output_format="html5",
            )
            return _sanitize_html(html)
        except Exception as e:
            # Never hand back the raw model output: it is rendered with | safe,
            # and this was the one path from model text to the email that
            # skipped the sanitizer.
            logger.error("Error converting markdown to HTML: %s", e)
            return _sanitize_html(escape(v))

    @field_validator("tags", mode="before")
    def _validate_tags(cls, v: Any) -> list[str]:
        tags = _normalize_tags(v)
        # Keep the model's ordering (it lists tags most-relevant-first) while
        # de-duplicating; alphabetical sorting would drop important tags purely
        # because they sort late. dict.fromkeys preserves first-seen order.
        unique_tags = list(dict.fromkeys(tags))
        return unique_tags[:MAX_TAGS] if unique_tags else list(DEFAULT_TAGS)

    @field_validator("urls", mode="before")
    def _validate_urls(cls, v: Any) -> list[str]:
        return _normalize_urls(v)


class SummarizationOptions(BaseModel):
    """The settings the ``summarization`` config section and the Summarizer share.

    Declared once: the config model (``app.configs.Summarization``) extends this with
    the greeting model, and ``SummarizerSettings`` with what only the run knows.
    """

    use_filtering: bool = True
    filtering_criteria: FilteringCriteria = FilteringCriteria.ALL
    included_topics: list[str] = Field(default_factory=list)
    excluded_topics: list[str] = Field(default_factory=list)
    filtering_model_id: LanguageModelId
    filtering_enable_thinking: bool = False
    summarization_model_id: LanguageModelId
    summarization_enable_thinking: bool = False
    max_concurrency: int = Field(default=10, ge=1)
    min_score: float = Field(default=0.7, ge=0.0, le=1.0)
    max_posts: int | None = Field(default=None, ge=1)
    # Cap on articles from one source, applied after ranking; freed slots are
    # backfilled in rank order, so it changes which posts ship, never how many.
    # Leave unset for a single-vendor digest, where concentration is the point.
    max_per_source: int | None = Field(default=None, ge=1)
    # The filter scores prose, so it gets visible text rather than HTML (§10).
    filter_on_visible_text: bool = True
    # The summarizer needs markup (code, tables, images) but not site furniture.
    summarize_on_cleaned_html: bool = True
    # Thinking budgets for budget-thinking models (Claude 4.x and older) only;
    # adaptive-thinking models ignore them and use thinking_effort. None = the
    # factory default, so only an explicitly set budget is reported as ignored.
    filtering_thinking_budget_tokens: int | None = Field(default=None, ge=1024)
    summarization_thinking_budget_tokens: int | None = Field(default=None, ge=1024)
    # Depth for adaptive-thinking models (output_config.effort) and GPT models
    # (reasoning.effort). None = the model's default. Sent only while thinking is
    # on, except on models that cannot stop thinking (§8).
    thinking_effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    # Cap on summary output tokens; None = the model's maximum.
    summarization_max_tokens: int | None = Field(default=None, ge=256)


def gate_by_content_length(
    posts: list[Post], minimum: int
) -> tuple[list[Post], list[tuple[Post, str]]]:
    """Split ``posts`` into those with enough visible text to summarize well and
    the dropped ones, each with the reason the run report shows.

    A thin post would only produce an empty or invented summary.
    """
    substantive: list[Post] = []
    dropped: list[tuple[Post, str]] = []
    for post in posts:
        length = post.text_length()
        if length >= minimum:
            substantive.append(post)
        else:
            reason = (
                f"Insufficient content: {length} visible chars "
                f"(minimum {minimum}). Skipped before summarization."
            )
            dropped.append((post, reason))
            logger.info(
                "Post '%s' dropped: %d visible chars < %d minimum.",
                post.title,
                length,
                minimum,
            )
    logger.info(
        "Content-length gate: %d/%d posts have sufficient content.",
        len(substantive),
        len(posts),
    )
    return substantive, dropped


def select_for_digest(
    posts: list[Post], max_posts: int | None, max_per_source: int | None
) -> list[Post]:
    """Rank by relevance and pick the issue's line-up.

    Ordering is ``(score desc, published_date desc)``: the rubric's few anchors
    produce frequent ties, and recency breaks them deterministically so
    the same inputs always yield the same issue.

    When ``max_per_source`` is set, the cap is applied in a first pass and any
    remaining slots are then BACKFILLED from the leftovers in rank order. A
    cap that shrank the digest would trade a real loss (fewer articles) for a
    cosmetic gain, so the cap only ever changes *which* posts fill the slots,
    never *how many*. Leaving it unset keeps pure score ranking — the right
    default for a single-vendor digest (``filtering_criteria: amazon``), where
    source concentration is the point rather than a defect.
    """
    ranked = sorted(posts, key=lambda p: (p.score, p.published_date), reverse=True)
    limit = max_posts or len(ranked)
    cap = max_per_source

    if cap:
        per_source: dict[str, int] = {}
        selected: list[Post] = []
        overflow: list[Post] = []
        for post in ranked:
            if len(selected) >= limit:
                break
            if per_source.get(post.source, 0) < cap:
                per_source[post.source] = per_source.get(post.source, 0) + 1
                selected.append(post)
            else:
                overflow.append(post)
        if len(selected) < limit and overflow:
            backfilled = overflow[: limit - len(selected)]
            logger.info(
                "Source cap (%d/source) left %d slot(s) unfilled; "
                "backfilling in rank order: %s",
                cap,
                len(backfilled),
                [p.title for p in backfilled],
            )
            selected.extend(backfilled)
            # Re-rank so the backfilled posts sit in relevance order rather
            # than being appended after lower-scoring capped picks.
            selected.sort(key=lambda p: (p.score, p.published_date), reverse=True)
        ranked = selected
    elif len(ranked) > limit:
        ranked = ranked[:limit]

    if len(posts) > len(ranked):
        logger.info(
            "Selected top %d of %d posts for the digest%s.",
            len(ranked),
            len(posts),
            f" (max {cap} per source)" if cap else "",
        )
    logger.info(
        "Digest line-up (relevance order): %s",
        [f"{p.source}:{p.title} ({p.score:.2f})" for p in ranked],
    )
    return ranked

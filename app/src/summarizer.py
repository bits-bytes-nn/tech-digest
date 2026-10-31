import re
import time
from collections.abc import Callable
from html import escape, unescape
from typing import Any, ClassVar, Literal
from urllib.parse import quote

import boto3
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
from langchain_aws import ChatBedrockConverse
from langchain_classic.output_parsers import OutputFixingParser
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, field_validator

from .constants import (
    DEFAULT_TAGS,
    MAX_TAGS,
    FilteringCriteria,
    Language,
    LanguageModelId,
)
from .eval_metrics import is_anchor
from .feed_parser import (
    Post,
    content_html,
    is_email_renderable_image,
    is_safe_url,
    visible_text,
)
from .logger import logger
from .model_factory import BedrockLanguageModelFactory
from .prompts.prompts import FilteringPrompt, SummarizationPrompt

# The terminal-colon rule lives in ``quality_metrics`` so the corrector below and
# the rubric that scores it share one pattern and cannot disagree.
from .quality_metrics import KO_LENGTH_BAND, KO_TERMINAL_COLON, is_korean
from .utils import (
    BatchProcessor,
    HTMLTagOutputParser,
    measure_execution_time,
)


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
        # with attributes in raw HTML). The old ``"<table>" -> classed`` string
        # replace silently missed attributed tables, leaving them unstyled — so
        # a metrics table rendered borderless/cramped in email clients.
        if name == "table":
            tag["class"] = "table-bordered"
        # Every image gets the same size: the width attribute for Outlook, and a
        # fluid style so narrower screens scale it down.
        elif name == "img":
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
# Total time one summary may spend checking its images. Past it, the images not
# yet proven to load are dropped rather than shipped unchecked.
IMAGE_PROBE_BUDGET_SECONDS = 120


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


def _drop_unrenderable_images(
    summary_html: str,
    source_html: str,
    budget_seconds: float = IMAGE_PROBE_BUDGET_SECONDS,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[str, list[str]]:
    """Keep only images a mail client will display.

    Outlook for Windows shows a WebP image's alt text instead, so the alt text
    and the caption ran together as one odd sentence ("Massive Text Embedding
    Benchmark (Code)코드 검색 부문 ... 차트입니다"). An image that fails is
    swapped for another rendition the article lists (NVIDIA serves WebP as the
    main file and PNG in its srcset); with none, its figure goes. Returns
    ``(cleaned_html, dropped_srcs)``.
    """
    if "<img" not in summary_html.lower():
        return summary_html, []
    deadline = clock() + budget_seconds
    soup = BeautifulSoup(summary_html, "html.parser")
    dropped: list[str] = []
    changed = False
    for img in list(soup.find_all("img")):
        src = str(img.get("src") or "")
        chosen = None
        for candidate in [src, *_image_candidates(src, source_html)]:
            if clock() >= deadline:
                break
            if is_email_renderable_image(candidate):
                chosen = candidate
                break
        if chosen is None:
            dropped.append(src)
            (img.find_parent("figure") or img).decompose()
        elif chosen != src:
            img["src"] = chosen
            changed = True
    return (str(soup), dropped) if (dropped or changed) else (summary_html, [])


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

    Declared once: the config model (``configs.Summarization``) extends this with
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


class SummarizerSettings(SummarizationOptions):
    """Everything the Summarizer needs, as one validated object.

    The composition root builds it from config
    (``model_validate(cfg.summarization.model_dump() | {...})``), adding the
    values only the run knows: the issue's language and the content gate.
    """

    # The output-repair model. Not a config key; scripts/put_inference_profiles.py
    # reads this default so the profile it provisions matches the model used.
    fixing_model_id: LanguageModelId = LanguageModelId.CLAUDE_V5_5_HAIKU
    language: Language = Language.KO
    min_content_length: int = Field(default=600, ge=0)


class Summarizer:
    def __init__(
        self,
        boto_session: boto3.Session,
        settings: SummarizerSettings,
    ) -> None:
        self.boto_session = boto_session
        self.settings = settings
        self.language = settings.language
        self.filtered_out_posts: list[tuple[Post, str]] = []
        self.batch_processor = BatchProcessor(
            max_concurrency=settings.max_concurrency,
            batch_size=settings.max_concurrency,
        )
        self.llm_factory = BedrockLanguageModelFactory(boto_session=self.boto_session)
        self.filter = self._create_filter(
            settings.filtering_model_id,
            settings.filtering_enable_thinking,
            thinking_budget_tokens=settings.filtering_thinking_budget_tokens,
        )
        self.summarizer = self._create_summarizer(
            settings.summarization_model_id,
            settings.fixing_model_id,
            settings.summarization_enable_thinking,
            thinking_budget_tokens=settings.summarization_thinking_budget_tokens,
            max_tokens=settings.summarization_max_tokens,
        )

    def _create_filter(
        self,
        model_id: LanguageModelId,
        use_thinking: bool = False,
        thinking_budget_tokens: int | None = None,
    ) -> Runnable:
        model_info = self.llm_factory.get_model_info(model_id)
        if not model_info:
            raise ValueError(f"Invalid model ID: '{model_id}'")
        if use_thinking and model_info.supports_thinking:
            # Extended thinking forces temperature=1.0 (Anthropic API rule), so
            # the 0.00-1.00 rubric scores become non-deterministic: the same
            # post may fall either side of min_score across runs, making the
            # score-ranked top-N cut unstable. Acceptable if you want deeper
            # reasoning, but be aware inclusion is no longer reproducible.
            # Models without sampling controls are non-deterministic with or
            # without this switch (and Opus 5.5 and Fable 5.1 think either way), so
            # the temperature claim would be false for them.
            if model_info.supports_sampling_params:
                logger.warning(
                    "Filtering has thinking enabled: scores are sampled at "
                    "temperature=1.0 and are NOT deterministic across runs."
                )
            else:
                logger.warning(
                    "Filtering has thinking enabled on a model without sampling "
                    "controls: scores are NOT deterministic across runs."
                )
        filtering_llm = self.llm_factory.get_model(
            model_id=model_id,
            temperature=0.0,
            stage="filtering",
            enable_thinking=use_thinking,
            thinking_budget_tokens=thinking_budget_tokens
            or self.llm_factory.DEFAULT_THINKING_BUDGET_TOKENS,
            effort=self.settings.thinking_effort,
        )
        # The large, static filtering rubric is sent on every post — cache it as
        # a system prefix so only the per-post content is billed at full rate.
        # The cache marker differs by backend, so tell the prompt which one runs.
        prompt = FilteringPrompt.for_criteria(
            self.settings.filtering_criteria
        ).get_prompt(
            enable_prompt_cache=model_info.supports_prompt_caching,
            use_converse=isinstance(filtering_llm, ChatBedrockConverse),
        )
        return (
            prompt
            | filtering_llm
            | HTMLTagOutputParser(
                tag_names=FilteringPrompt.output_variables,
                text_tags=FilteringPrompt.output_variables,
            )
        )

    def _cache_can_hit(self) -> bool:
        """Whether caching the summarization prefix can ever pay for itself.

        A cache entry is only written when a response completes, so every request
        in a concurrent wave starts against a cold cache. The summarization batch
        is bounded by ``max_posts`` and runs ``max_concurrency`` at a time, so when
        the whole batch fits in one wave *every* call misses — and the digest is
        weekly, far beyond the entry lifetime, so nothing survives to the next run
        either. Writing costs 1.25x the normal input rate, which made caching a
        pure surcharge in that configuration.

        This is measured, not predicted: across 24 real summarization calls the
        usage logger recorded ``cache_read=0`` on every one, with a
        ``cache_write`` on every one. Filtering is not gated by this: it scores
        every collected post in several waves, so later waves can hit (on models
        that accept a cache point at all).
        """
        max_posts = self.settings.max_posts
        # None means "no cap", so the batch is however many posts survived
        # filtering — routinely more than one wave, and caching helps.
        return max_posts is None or max_posts > self.settings.max_concurrency

    def _create_summarizer(
        self,
        model_id: LanguageModelId,
        fixing_model_id: LanguageModelId,
        use_thinking: bool = False,
        thinking_budget_tokens: int | None = None,
        max_tokens: int | None = None,
    ):
        model_info = self.llm_factory.get_model_info(model_id)
        if not model_info:
            raise ValueError(f"Invalid model ID: '{model_id}'")
        fixing_llm = self.llm_factory.get_model(
            model_id=LanguageModelId(fixing_model_id),
            temperature=0.0,
            stage="output-fixing",
        )
        # required_tags=["summary"] makes the parser raise on empty/malformed
        # output so OutputFixingParser actually invokes the fixing model to
        # repair it — otherwise a missing summary would silently pass through.
        output_parser = OutputFixingParser.from_llm(
            llm=fixing_llm,
            parser=HTMLTagOutputParser(
                tag_names=SummarizationPrompt.output_variables,
                required_tags=["summary"],
                text_tags=self.SUMMARY_TEXT_TAGS,
            ),
        )
        summarization_llm = self.llm_factory.get_model(
            model_id=model_id,
            temperature=0.0,
            stage="summarization",
            enable_thinking=use_thinking,
            thinking_budget_tokens=thinking_budget_tokens
            or self.llm_factory.DEFAULT_THINKING_BUDGET_TOKENS,
            max_tokens=max_tokens,
            effort=self.settings.thinking_effort,
        )
        # Cache the static analysis instructions as a system prefix — but only
        # when a later call can actually READ what an earlier one wrote.
        prompt = SummarizationPrompt.for_language(self.language).get_prompt(
            enable_prompt_cache=model_info.supports_prompt_caching
            and self._cache_can_hit(),
            use_converse=isinstance(summarization_llm, ChatBedrockConverse),
        )
        return prompt | summarization_llm | output_parser

    @measure_execution_time
    def process_posts(self, posts: list[Post]) -> list[Post]:
        substantive_posts = self._gate_by_content_length(posts)
        if not substantive_posts:
            logger.warning("No posts had sufficient content to summarize.")
            return []
        if self.settings.use_filtering:
            posts_to_process = self._filter_posts(substantive_posts)
            if not posts_to_process:
                logger.warning("No posts remained after filtering.")
                return []
        else:
            posts_to_process = substantive_posts
        # Rank and apply max_posts BEFORE summarizing, so we only pay to
        # summarize the posts that will actually ship (summarization is the
        # expensive LLM step).
        return self._summarize_lineup(posts_to_process)

    # How many times a failed summary may be replaced by the next-ranked post.
    # Bounded so a systematic failure (a rejected request shape) costs a few
    # extra calls, not one per eligible post.
    MAX_BACKFILL_ROUNDS: ClassVar[int] = 2

    def _summarize_lineup(self, eligible: list[Post]) -> list[Post]:
        """Summarize the issue's line-up, replacing posts whose summary failed.

        Selection happens BEFORE summarizing so only shipped posts are paid for.
        But a post whose summary then failed simply vanished: the issue went out
        a card short even when other posts had passed the filter, and the post
        lost was usually the top-ranked one. Each round re-selects from the
        eligible posts minus the failures — so the per-source cap still holds —
        and summarizes only the posts not yet done.
        """
        pool = list(eligible)
        done: dict[str, Post] = {}
        lineup = self._select_for_digest(pool)
        for round_number in range(self.MAX_BACKFILL_ROUNDS + 1):
            pending = [p for p in lineup if p.link not in done]
            if not pending:
                break
            for post in self._summarize_posts(pending):
                done[post.link] = post
            failed = {p.link for p in pending if p.link not in done}
            if not failed:
                break
            pool = [p for p in pool if p.link not in failed]
            if round_number == self.MAX_BACKFILL_ROUNDS:
                logger.warning(
                    "%d summary failure(s) left after %d backfill round(s); "
                    "the issue ships short.",
                    len(failed),
                    self.MAX_BACKFILL_ROUNDS,
                )
                break
            lineup = self._select_for_digest(pool)
            logger.warning(
                "%d summary failure(s); refilling the line-up from the next-ranked "
                "eligible posts.",
                len(failed),
            )
        return [p for p in lineup if p.link in done]

    def _select_for_digest(self, posts: list[Post]) -> list[Post]:
        return self.select_for_digest(
            posts, self.settings.max_posts, self.settings.max_per_source
        )

    @staticmethod
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

    def _gate_by_content_length(self, posts: list[Post]) -> list[Post]:
        """Drop posts whose visible text is too thin to summarize well.

        A thin post would only produce an empty or invented summary. Dropped
        posts are recorded in ``filtered_out_posts`` so the run report shows them.
        """
        minimum = self.settings.min_content_length
        substantive: list[Post] = []
        for post in posts:
            length = post.text_length()
            if length >= minimum:
                substantive.append(post)
            else:
                reason = (
                    f"Insufficient content: {length} visible chars "
                    f"(minimum {minimum}). Skipped before summarization."
                )
                self.filtered_out_posts.append((post, reason))
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
        return substantive

    def _filter_posts(self, posts: list[Post]) -> list[Post]:
        valid_posts = [p for p in posts if p.content]
        # Reachable only with min_content_length=0 (the gate above otherwise
        # requires real text). Record rather than drop, so a post can never
        # vanish from BOTH the surviving list and the filtered-out report.
        for post in posts:
            if not post.content:
                self.filtered_out_posts.append(
                    (post, "No content available to evaluate.")
                )
                logger.warning("Post '%s' has no content; skipping.", post.title)
        if not valid_posts:
            return []

        included = ", ".join(self.settings.included_topics)
        excluded = ", ".join(self.settings.excluded_topics)
        strip_markup = self.settings.filter_on_visible_text

        def prepare_inputs(items: list[Post]) -> list[dict[str, Any]]:
            return [
                {
                    # The filter scores topic relevance and quality, both of which
                    # live in the prose — markup carries none of it. Measured on a
                    # real run, raw HTML was 8x the visible text (one page: 77,417
                    # chars of HTML around 5,669 chars of article), so sending HTML
                    # meant ~87% of filtering input tokens were nav bars, inline
                    # styles and scripts. Summarization still receives the full
                    # HTML, which it needs for code blocks, tables and image URLs.
                    "post": visible_text(post.content)
                    if strip_markup
                    else post.content,
                    "original_title": post.title,
                    # Provenance is evidence the text may not state: an Amazon
                    # Science post rarely names Amazon, and scored as off-topic.
                    "source_url": str(post.link),
                    "included_topics": included,
                    "excluded_topics": excluded,
                }
                for post in items
            ]

        responses = self.batch_processor.execute_with_fallback(
            items_to_process=valid_posts,
            prepare_inputs_func=prepare_inputs,
            batch_func=self.filter.batch,
            sequential_func=self.filter.invoke,
            task_name="filtering",
        )
        filtered_posts = []
        # Lengths are guaranteed equal (both batch and sequential-fallback paths
        # preserve positional alignment), so strict=True catches any regression.
        for post, response in zip(valid_posts, responses, strict=True):
            if response is None:
                # The model call failed for this post even after fallback.
                self.filtered_out_posts.append(
                    (post, f"{FILTER_FAILED} (no model response).")
                )
                logger.warning("No filter response for '%s'; skipping.", post.title)
                continue
            if not str(response.get("score") or "").strip():
                # No score at all is a failed call, not a verdict. Defaulting it
                # to 0.0 filed a refusal, a truncated answer or a malformed one
                # under "low relevance", indistinguishable from a real low score
                # — so a model that declined every post read as a quiet week.
                logger.error(
                    "Filter response for '%s' has no score; treating it as a "
                    "failed call. Response: %s",
                    post.title,
                    response,
                )
                self.filtered_out_posts.append(
                    (post, f"{FILTER_FAILED} (no score in model response).")
                )
                continue
            try:
                score = float(response["score"])
                # Out of [0,1] is a failed call, not a verdict: clamping the
                # model's "8.5" (meant as 0.85) to 1.0 would rank it first.
                # A value off the rubric's anchors is only logged.
                if not 0.0 <= score <= 1.0:
                    logger.error(
                        "Filter score %.3f for '%s' is outside [0,1]; treating it "
                        "as a failed call.",
                        score,
                        post.title,
                    )
                    self.filtered_out_posts.append(
                        (post, f"{FILTER_FAILED} (score {score} out of range).")
                    )
                    continue
                if not is_anchor(score):
                    logger.info(
                        "Filter score %.3f for '%s' is not one of the rubric's anchors.",
                        score,
                        post.title,
                    )
                reason = response.get("reason", "No reason provided.")
                post.score = score
                if score >= self.settings.min_score:
                    filtered_posts.append(post)
                    logger.info(
                        "Post '%s' passed filter with score %.2f.", post.title, score
                    )
                else:
                    self.filtered_out_posts.append((post, reason))
                    logger.info(
                        "Post '%s' filtered out with score %.2f. Reason: %s",
                        post.title,
                        score,
                        reason,
                    )
            except (ValueError, TypeError) as e:
                logger.error(
                    "Could not parse filter response for '%s': %s. Response: %s",
                    post.title,
                    e,
                    response,
                )
                # Record it so the post is accounted for in the run report
                # rather than silently vanishing from both lists.
                self.filtered_out_posts.append((post, f"{FILTER_UNPARSEABLE}: {e}"))
        return filtered_posts

    def _summarize_posts(self, posts: list[Post]) -> list[Post]:
        """Summarize ``posts`` in place and return ONLY the ones that succeeded.

        The success set is returned explicitly rather than inferred afterwards
        from ``post.summary`` being non-empty. That inference was subtly wrong:
        any pre-existing text in the field (the RSS teaser, before it stopped
        being seeded — see ``Post.from_entry``) made a FAILED summarization look
        successful, so the post shipped its teaser as the article body while also
        appearing in the filtered-out report.
        """
        if not posts:
            return []

        def prepare_inputs(items: list[Post]) -> list[dict[str, Any]]:
            # Site furniture is stripped but the markup is kept: the summary
            # needs the code blocks, tables and image URLs that live in it.
            return [
                {
                    "post": content_html(post.content)
                    if self.settings.summarize_on_cleaned_html
                    else post.content
                }
                for post in items
            ]

        summaries = self.batch_processor.execute_with_fallback(
            items_to_process=posts,
            prepare_inputs_func=prepare_inputs,
            batch_func=self.summarizer.batch,
            sequential_func=self.summarizer.invoke,
            task_name="summarization",
        )
        succeeded: list[Post] = []
        for post, summary_data in zip(posts, summaries, strict=True):
            if summary_data is None:
                logger.warning(
                    "No summary produced for '%s' (model call failed).", post.title
                )
                # Record so the post is accounted for in the run report and, if
                # every summary fails, the empty-digest alert fires instead of a
                # silent empty send. Mirrors _filter_posts' failure handling.
                self.filtered_out_posts.append(
                    (post, f"{SUMMARY_FAILED} (no model response).")
                )
                continue
            try:
                summary_output = SummaryOutput.model_validate(summary_data)
                finalized = self._finalize_summary(post, summary_output.summary)
                if problem := self._unusable_summary(finalized):
                    logger.warning("Summary for '%s' rejected: %s", post.title, problem)
                    self.filtered_out_posts.append(
                        (post, f"{SUMMARY_FAILED} ({problem}).")
                    )
                    continue
                post.summary = finalized
                post.one_liner = summary_output.one_liner
                post.tags = summary_output.tags
                post.urls = summary_output.urls
                if unsourced := _unsourced_urls(post.urls, post.content):
                    logger.warning(
                        "%d reference link(s) for '%s' are not present in the "
                        "source article (kept, but verify): %s",
                        len(unsourced),
                        post.title,
                        unsourced,
                    )
                succeeded.append(post)
            except Exception as e:
                logger.error(
                    "Failed to parse summary for post '%s': %s. Data: %s",
                    post.title,
                    e,
                    summary_data,
                )
                self.filtered_out_posts.append((post, f"{SUMMARY_INVALID}: {e}"))
        return succeeded

    # A real summary is thousands of characters (KO band 4,000-6,000, EN
    # 1,300-2,000 words). A refusal, a "too short to summarize" reply or a
    # repair that wrapped either of them in tags is a sentence or two. The floor
    # sits far below the band so a terse but genuine summary still ships.
    # Summary tags read as plain text. Not one_liner: _clean_one_liner already
    # decodes it once, and a second decode turned "List&lt;String&gt;" into
    # "List" on the card.
    SUMMARY_TEXT_TAGS: ClassVar[list[str]] = ["tags"]
    MIN_SUMMARY_SHARE: ClassVar[float] = 0.3
    EN_MIN_WORDS: ClassVar[int] = 400

    def _unusable_summary(self, summary_html: str) -> str | None:
        """Why this summary must not ship, or None if it may.

        The output parser only checks that a ``<summary>`` tag exists. Without
        this, a refused or empty answer that the repair model wrapped in tags
        was published as an article card.
        """
        text = visible_text(summary_html)
        if self.language is Language.KO:
            minimum = int(KO_LENGTH_BAND[0] * self.MIN_SUMMARY_SHARE)
            if len(text) < minimum:
                return f"{len(text)} visible chars, below {minimum}"
            if not is_korean(text):
                return "not written in Korean"
        elif len(text.split()) < self.EN_MIN_WORDS:
            return f"{len(text.split())} words, below {self.EN_MIN_WORDS}"
        return None

    @staticmethod
    def _finalize_summary(post: Post, summary_html: str) -> str:
        cleaned, dropped = _strip_unsourced_images(
            summary_html, post.content, post.images
        )
        if dropped:
            logger.warning(
                "Dropped %d image(s) from the summary of '%s' — not found in the "
                "source article (would have rendered broken): %s",
                len(dropped),
                post.title,
                dropped,
            )
        cleaned, unrenderable = _drop_unrenderable_images(cleaned, post.content)
        if unrenderable:
            logger.warning(
                "Dropped %d image(s) from the summary of '%s' — not a type mail "
                "clients display, or not loadable: %s",
                len(unrenderable),
                post.title,
                unrenderable,
            )
        return _postprocess_summary(_encode_image_srcs(cleaned))

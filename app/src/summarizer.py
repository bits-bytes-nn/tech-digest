import re
from html import escape
from typing import Any, ClassVar

import boto3
import markdown
from bs4 import BeautifulSoup
from langchain_aws import ChatBedrockConverse
from langchain_classic.output_parsers import OutputFixingParser
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, field_validator

from .constants import FilteringCriteria, Language, LanguageModelId
from .eval_metrics import is_on_grid
from .feed_parser import Post, is_safe_url
from .logger import logger
from .model_factory import BedrockLanguageModelFactory
from .prompts.prompts import FilteringPrompt, SummarizationPrompt
from .utils import (
    BatchProcessor,
    HTMLTagOutputParser,
    measure_execution_time,
)


class SummarizerConfig:
    CONVERT_MARKDOWN_TO_HTML: ClassVar[bool] = True
    DEFAULT_TAGS: ClassVar[list[str]] = ["uncategorized"]
    # NOTE: <table> is class-injected in _sanitize_html (BeautifulSoup catches
    # every table regardless of attributes; a literal "<table>" string match
    # here silently missed attributed tables and left them unstyled).
    HTML_REPLACEMENTS: ClassVar[dict[str, str]] = {
        "<code>": '<code class="highlight">',
        "<pre>": '<pre class="pre-scrollable">',
    }
    MARKDOWN_EXTENSIONS: ClassVar[list[str]] = [
        "attr_list",
        "codehilite",
        "def_list",
        "fenced_code",
        "footnotes",
        "nl2br",
        "sane_lists",
        "tables",
        "toc",
    ]
    MAX_TAGS: ClassVar[int] = 5
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
    # style, etc.). href/src additionally pass a scheme check below.
    ALLOWED_HTML_ATTRS: ClassVar[dict[str, frozenset[str]]] = {
        "a": frozenset({"href", "title"}),
        "img": frozenset({"src", "alt", "title"}),
        "code": frozenset({"class"}),
        "pre": frozenset({"class"}),
        "table": frozenset({"class"}),
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


# Korean sentence-ending verb syllables the model sometimes terminates with a
# colon instead of a period ("...합니다:" -> "...합니다."). Korean prose does not
# end a sentence with a colon, so a colon right after one of these syllables is
# always a mistake — but ONLY when it actually ends the sentence.
_KO_SENTENCE_END_SYLLABLES = "다요죠"

# Match a sentence-ending syllable + colon ONLY at a real clause boundary: the
# colon must be followed by markup (</p>, </li>, <br>, ...) or end-of-string.
# This is the key difference from a blind ``"다:" -> "다."`` replace, which also
# corrupts legitimate list-introducing colons (e.g. "결과는 다음과 같습니다: 첫째").
# Post-processing runs AFTER markdown->HTML, so a terminal colon is always
# followed by a closing tag; an introducing colon is followed by text.
#
# The trailing negative lookahead preserves a colon that introduces a BLOCK
# list: markdown renders "...다음과 같습니다:" before a bulleted list as
# "<p>...습니다:</p>\n<ul>" (or "...습니다:<br>\n<ul>" under nl2br), where the
# colon IS at a boundary yet legitimately introduces the list — so skip it when
# a <ul>/<ol> follows past the intervening closing tags / <br>.
_KO_TERMINAL_COLON = re.compile(
    rf"([{_KO_SENTENCE_END_SYLLABLES}]):(?=\s*(?:<|$))"
    r"(?!\s*(?:(?:</[^>]+>|<br\s*/?>)\s*)*<[uo]l\b)"
)

# Host aliases: a source occasionally surfaces under a CDN/subdomain variant
# that should be normalized to its canonical host in rendered links. These are
# genuine per-host facts (not heuristics), so they live in an explicit map.
_HOST_ALIASES: dict[str, str] = {
    "magazine.sebastianraschka": "sebastianraschka",
}


def _postprocess_summary(summary: str) -> str:
    summary = _KO_TERMINAL_COLON.sub(r"\1.", summary)
    for variant, canonical in _HOST_ALIASES.items():
        summary = summary.replace(variant, canonical)
    return summary


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


def _sanitize_html(html: str) -> str:
    """Strip disallowed tags/attributes from model-generated summary HTML.

    The summary is rendered with Jinja ``| safe``, so it is a trust boundary:
    the LLM (and, transitively, scraped source content it may echo) must not be
    able to inject <script>, event handlers, or javascript:/data: URLs into the
    outbound email. Unknown tags are unwrapped (their text is kept); disallowed
    attributes and unsafe href/src schemes are removed.
    """
    soup = BeautifulSoup(html, "html.parser")
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
            tag["class"] = "table table-bordered"
    return str(soup)


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

    Returns ``(cleaned_html, dropped_srcs)``.
    """
    if "<img" not in summary_html.lower():
        return summary_html, []
    allowed = set(known_images)
    soup = BeautifulSoup(summary_html, "html.parser")
    dropped: list[str] = []
    for img in list(soup.find_all("img")):
        src = str(img.get("src") or "").strip()
        if src and (src in allowed or src in source_html):
            continue
        dropped.append(src or "(no src)")
        img.decompose()
    if not dropped:
        return summary_html, []
    return str(soup), dropped


def _unsourced_urls(urls: list[str], source_html: str) -> list[str]:
    """Reference links whose href does not appear verbatim in the source post.

    Reported (not removed): unlike an image, a slightly-rewritten but real link
    is still useful to the reader, so dropping it would cost more than it saves.
    Surfacing the count makes prompt drift visible in the run log instead of
    silently shipping links the article never contained.
    """
    unsourced = []
    for anchor in urls:
        href = BeautifulSoup(anchor, "html.parser").find("a")
        url = str(href.get("href", "")) if href else ""
        if url and url not in source_html:
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
        text = BeautifulSoup(v, "html.parser").get_text(separator=" ", strip=True)
        return " ".join(text.split())

    @field_validator("summary", mode="before")
    def _convert_markdown_to_html(cls, v: str) -> str:
        if not SummarizerConfig.CONVERT_MARKDOWN_TO_HTML or not isinstance(v, str):
            return v
        try:
            html = markdown.markdown(
                v,
                extensions=SummarizerConfig.MARKDOWN_EXTENSIONS,
                output_format="html5",
            )
            for old, new in SummarizerConfig.HTML_REPLACEMENTS.items():
                html = html.replace(old, new)
            # Sanitize AFTER the class-injecting replacements so the allow-listed
            # class attributes (code/pre/table) survive the attribute filter.
            return _sanitize_html(html)
        except Exception as e:
            logger.error("Error converting markdown to HTML: %s", e)
            return v

    @field_validator("tags", mode="before")
    def _validate_tags(cls, v: Any) -> list[str]:
        tags = _normalize_tags(v)
        # Keep the model's ordering (it lists tags most-relevant-first) while
        # de-duplicating; alphabetical sorting would drop important tags purely
        # because they sort late. dict.fromkeys preserves first-seen order.
        unique_tags = list(dict.fromkeys(tags))
        return (
            unique_tags[: SummarizerConfig.MAX_TAGS]
            if unique_tags
            else SummarizerConfig.DEFAULT_TAGS
        )

    @field_validator("urls", mode="before")
    def _validate_urls(cls, v: Any) -> list[str]:
        return _normalize_urls(v)


class SummarizerSettings(BaseModel):
    """Everything the Summarizer needs to run, as one validated object.

    Replaces a 14-positional-parameter constructor plus four more arguments on
    ``process_posts``. Field names deliberately mirror the ``summarization``
    config section so the composition root can build this straight from config
    (``SummarizerSettings.model_validate(cfg.summarization.model_dump() | ...)``)
    without restating every key — and without ``src`` importing ``configs``,
    which would invert the dependency direction.
    """

    filtering_model_id: LanguageModelId
    summarization_model_id: LanguageModelId
    fixing_model_id: LanguageModelId | None = None
    filtering_criteria: FilteringCriteria = FilteringCriteria.ALL
    filtering_enable_thinking: bool = False
    summarization_enable_thinking: bool = False
    language: Language = Language.KO
    use_filtering: bool = True
    included_topics: list[str] = Field(default_factory=list)
    excluded_topics: list[str] = Field(default_factory=list)
    max_concurrency: int = Field(default=10, ge=1)
    min_score: float = Field(default=0.7, ge=0.0, le=1.0)
    min_content_length: int = Field(default=600, ge=0)
    max_posts: int | None = Field(default=None, ge=1)
    max_per_source: int | None = Field(default=None, ge=1)
    filtering_thinking_budget_tokens: int = Field(default=4096, ge=1024)
    summarization_thinking_budget_tokens: int = Field(default=8192, ge=1024)
    summarization_max_tokens: int | None = Field(default=None, ge=256)
    thinking_effort: str | None = None


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
            logger.warning(
                "Filtering has thinking enabled: scores are sampled at "
                "temperature=1.0 and are NOT deterministic across runs."
            )
        filtering_llm = self.llm_factory.get_model(
            model_id=model_id,
            temperature=0.0,
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
            | HTMLTagOutputParser(tag_names=FilteringPrompt.output_variables)
        )

    def _create_summarizer(
        self,
        model_id: LanguageModelId,
        fixing_model_id: LanguageModelId | None,
        use_thinking: bool = False,
        thinking_budget_tokens: int | None = None,
        max_tokens: int | None = None,
    ):
        model_info = self.llm_factory.get_model_info(model_id)
        if not model_info:
            raise ValueError(f"Invalid model ID: '{model_id}'")
        fixing_model_id = fixing_model_id or LanguageModelId.CLAUDE_V4_5_HAIKU
        fixing_llm = self.llm_factory.get_model(
            model_id=LanguageModelId(fixing_model_id), temperature=0.0
        )
        # required_tags=["summary"] makes the parser raise on empty/malformed
        # output so OutputFixingParser actually invokes the fixing model to
        # repair it — otherwise a missing summary would silently pass through.
        output_parser = OutputFixingParser.from_llm(
            llm=fixing_llm,
            parser=HTMLTagOutputParser(
                tag_names=SummarizationPrompt.output_variables,
                required_tags=["summary"],
            ),
        )
        summarization_llm = self.llm_factory.get_model(
            model_id=model_id,
            temperature=0.0,
            enable_thinking=use_thinking,
            thinking_budget_tokens=thinking_budget_tokens
            or self.llm_factory.DEFAULT_THINKING_BUDGET_TOKENS,
            max_tokens=max_tokens,
            effort=self.settings.thinking_effort,
        )
        # Cache the static analysis instructions as a system prefix; only the
        # per-post body varies and is billed at the full input rate.
        prompt = SummarizationPrompt.for_language(self.language).get_prompt(
            enable_prompt_cache=model_info.supports_prompt_caching,
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
        posts_to_process = self._select_for_digest(posts_to_process)
        summarized_posts = self._summarize_posts(posts_to_process)
        dropped = len(posts_to_process) - len(summarized_posts)
        if dropped:
            logger.warning(
                "%d/%d posts had no usable summary and were dropped.",
                dropped,
                len(posts_to_process),
            )
        return summarized_posts

    def _select_for_digest(self, posts: list[Post]) -> list[Post]:
        """Rank by relevance and pick the issue's line-up.

        Ordering is ``(score desc, published_date desc)``: the coarse 0.05 anchor
        grid produces frequent ties, and recency breaks them deterministically so
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
        max_posts = self.settings.max_posts
        limit = max_posts if max_posts and max_posts > 0 else len(ranked)
        cap = self.settings.max_per_source

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

        This is the fix for the recurring "article too short to write about"
        failure: previously such posts reached the summarizer and produced an
        empty/garbage summary. Dropped posts are recorded in
        ``filtered_out_posts`` so they appear in the run notification.
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

        def prepare_inputs(items: list[Post]) -> list[dict[str, Any]]:
            return [
                {
                    "post": post.content,
                    "original_title": post.title,
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
                    (post, "Filtering failed (no model response).")
                )
                logger.warning("No filter response for '%s'; skipping.", post.title)
                continue
            try:
                raw_score = float(response.get("score", 0.0))
                # Clamp to [0,1]: the model occasionally emits an out-of-range
                # value (e.g. "8.5" for an intended 0.85), which would both pass
                # the min_score gate and sort to the top of the rank, evicting
                # genuinely better posts. Off-grid values are logged (not
                # rejected — the rubric allows a ±0.05 step) for observability.
                score = min(1.0, max(0.0, raw_score))
                if score != raw_score:
                    logger.warning(
                        "Filter score %.3f for '%s' out of [0,1]; clamped to %.2f.",
                        raw_score,
                        post.title,
                        score,
                    )
                elif not is_on_grid(score):
                    logger.info(
                        "Filter score %.3f for '%s' is off the 0.05 anchor grid.",
                        score,
                        post.title,
                    )
                reason = response.get("reason", "No reason provided.")
                post.score = score
                if title := response.get("title"):
                    post.title = title
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
                self.filtered_out_posts.append(
                    (post, f"Filter response unparseable: {e}")
                )
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
            return [{"post": post.content} for post in items]

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
                    (post, "Summarization failed (no model response).")
                )
                continue
            try:
                summary_output = SummaryOutput.model_validate(summary_data)
                post.summary = self._finalize_summary(post, summary_output.summary)
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
                self.filtered_out_posts.append(
                    (post, f"Summary validation failed: {e}")
                )
        return succeeded

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
        return _postprocess_summary(cleaned)

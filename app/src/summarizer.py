import re
from html import escape
from typing import Any, ClassVar
from urllib.parse import urlparse

import boto3
import markdown
from bs4 import BeautifulSoup
from langchain_aws import ChatBedrockConverse
from langchain_classic.output_parsers import OutputFixingParser
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, field_validator

from .constants import FilteringCriteria, Language, LanguageModelId
from .feed_parser import Post
from .logger import logger
from .prompts.prompts import FilteringPrompt, SummarizationPrompt
from .utils import (
    BatchProcessor,
    BedrockLanguageModelFactory,
    HTMLTagOutputParser,
    measure_execution_time,
)


class SummarizerConfig:
    CONVERT_MARKDOWN_TO_HTML: ClassVar[bool] = True
    DEFAULT_TAGS: ClassVar[list[str]] = ["uncategorized"]
    HTML_REPLACEMENTS: ClassVar[dict[str, str]] = {
        "<code>": '<code class="highlight">',
        "<table>": '<table class="table table-bordered">',
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
_KO_TERMINAL_COLON = re.compile(rf"([{_KO_SENTENCE_END_SYLLABLES}]):(?=\s*(?:<|$))")

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


# Only these URL schemes may appear in a rendered <a href>. The summary fields
# are emitted by the LLM and rendered with Jinja's ``| safe`` (autoescape off),
# so an unchecked href like ``javascript:...`` or ``data:...`` would execute /
# phish. Anything not on this list is dropped.
_SAFE_URL_SCHEMES: frozenset[str] = frozenset({"http", "https", "mailto"})


def _is_safe_url(url: str) -> bool:
    """True only for absolute URLs whose scheme is explicitly allow-listed.

    Rejects ``javascript:``, ``data:``, ``vbscript:``, scheme-relative ``//evil``
    and bare/relative strings — none of which belong in a trusted outbound link.
    """
    try:
        scheme = urlparse(url.strip()).scheme.lower()
    except (ValueError, AttributeError):
        return False
    return scheme in _SAFE_URL_SCHEMES


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
                if _is_safe_url(url)
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
            if _is_safe_url(href):
                sanitized.append(_safe_anchor(href, text or href))
        elif _is_safe_url(item):
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
            elif attr.lower() in ("href", "src") and not _is_safe_url(
                str(tag.get(attr, ""))
            ):
                # Allow-listed attribute but unsafe scheme — drop the whole tag's
                # link rather than leave a javascript:/data: payload.
                del tag[attr]
    return str(soup)


class SummaryOutput(BaseModel):
    summary: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)

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


class Summarizer:
    def __init__(
        self,
        boto_session: boto3.Session,
        filtering_model_id: LanguageModelId,
        summarization_model_id: LanguageModelId,
        fixing_model_id: LanguageModelId | None = None,
        filtering_criteria: FilteringCriteria = FilteringCriteria.ALL,
        filtering_enable_thinking: bool = False,
        summarization_enable_thinking: bool = False,
        language: Language = Language.KO,
        max_concurrency: int = 10,
        min_score: float = 0.7,
        min_content_length: int = 600,
        filtering_thinking_budget_tokens: int = 4096,
        summarization_thinking_budget_tokens: int = 8192,
        summarization_max_tokens: int | None = None,
    ) -> None:
        self.boto_session = boto_session
        self.filtering_criteria = filtering_criteria
        self.language = language
        self.min_score = min_score
        self.min_content_length = min_content_length
        self.filtered_out_posts: list[tuple[Post, str]] = []
        self.batch_processor = BatchProcessor(
            max_concurrency=max_concurrency, batch_size=max_concurrency
        )
        self.llm_factory = BedrockLanguageModelFactory(boto_session=self.boto_session)
        self.filter = self._create_filter(
            filtering_model_id,
            filtering_enable_thinking,
            thinking_budget_tokens=filtering_thinking_budget_tokens,
        )
        self.summarizer = self._create_summarizer(
            summarization_model_id,
            fixing_model_id,
            summarization_enable_thinking,
            thinking_budget_tokens=summarization_thinking_budget_tokens,
            max_tokens=summarization_max_tokens,
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
        )
        # The large, static filtering rubric is sent on every post — cache it as
        # a system prefix so only the per-post content is billed at full rate.
        # The cache marker differs by backend, so tell the prompt which one runs.
        prompt = FilteringPrompt.for_criteria(self.filtering_criteria).get_prompt(
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
        )
        # Cache the static analysis instructions as a system prefix; only the
        # per-post body varies and is billed at the full input rate.
        prompt = SummarizationPrompt.for_language(self.language).get_prompt(
            enable_prompt_cache=model_info.supports_prompt_caching,
            use_converse=isinstance(summarization_llm, ChatBedrockConverse),
        )
        return prompt | summarization_llm | output_parser

    @measure_execution_time
    def process_posts(
        self,
        posts: list[Post],
        use_filtering: bool,
        included_topics: list[str],
        excluded_topics: list[str],
        max_posts: int | None = None,
    ) -> list[Post]:
        substantive_posts = self._gate_by_content_length(posts)
        if not substantive_posts:
            logger.warning("No posts had sufficient content to summarize.")
            return []
        if use_filtering:
            posts_to_process = self._filter_posts(
                substantive_posts, included_topics, excluded_topics
            )
            if not posts_to_process:
                logger.warning("No posts remained after filtering.")
                return []
        else:
            posts_to_process = substantive_posts
        # Rank by score and apply max_posts BEFORE summarizing, so we only pay
        # to summarize the posts that will actually ship (summarization is the
        # expensive LLM step). Secondary sort by recency breaks the frequent
        # ties among the coarse anchor scores deterministically.
        posts_to_process.sort(
            key=lambda p: (getattr(p, "score", 0.0), p.published_date),
            reverse=True,
        )
        if max_posts and max_posts > 0 and len(posts_to_process) > max_posts:
            logger.info(
                "Limiting to top %d of %d posts before summarization.",
                max_posts,
                len(posts_to_process),
            )
            posts_to_process = posts_to_process[:max_posts]
        self._summarize_posts(posts_to_process)
        return posts_to_process

    def _gate_by_content_length(self, posts: list[Post]) -> list[Post]:
        """Drop posts whose visible text is too thin to summarize well.

        This is the fix for the recurring "article too short to write about"
        failure: previously such posts reached the summarizer and produced an
        empty/garbage summary. Dropped posts are recorded in
        ``filtered_out_posts`` so they appear in the run notification.
        """
        substantive: list[Post] = []
        for post in posts:
            length = post.text_length()
            if length >= self.min_content_length:
                substantive.append(post)
            else:
                reason = (
                    f"Insufficient content: {length} visible chars "
                    f"(minimum {self.min_content_length}). Skipped before "
                    f"summarization."
                )
                self.filtered_out_posts.append((post, reason))
                logger.info(
                    "Post '%s' dropped: %d visible chars < %d minimum.",
                    post.title,
                    length,
                    self.min_content_length,
                )
        logger.info(
            "Content-length gate: %d/%d posts have sufficient content.",
            len(substantive),
            len(posts),
        )
        return substantive

    def _filter_posts(
        self, posts: list[Post], included_topics: list[str], excluded_topics: list[str]
    ) -> list[Post]:
        valid_posts = [p for p in posts if p.content]
        if not valid_posts:
            return []

        def prepare_inputs(items: list[Post]) -> list[dict[str, Any]]:
            return [
                {
                    "post": post.content,
                    "original_title": post.title,
                    "included_topics": ", ".join(included_topics),
                    "excluded_topics": ", ".join(excluded_topics),
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
                score = float(response.get("score", 0.0))
                reason = response.get("reason", "No reason provided.")
                post.score = score
                if title := response.get("title"):
                    post.title = title
                if score >= self.min_score:
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

    def _summarize_posts(self, posts: list[Post]):
        if not posts:
            return

        def prepare_inputs(items: list[Post]) -> list[dict[str, Any]]:
            return [{"post": post.content} for post in items]

        summaries = self.batch_processor.execute_with_fallback(
            items_to_process=posts,
            prepare_inputs_func=prepare_inputs,
            batch_func=self.summarizer.batch,
            sequential_func=self.summarizer.invoke,
            task_name="summarization",
        )
        for post, summary_data in zip(posts, summaries, strict=True):
            if summary_data is None:
                logger.warning(
                    "No summary produced for '%s' (model call failed).", post.title
                )
                continue
            try:
                summary_output = SummaryOutput.model_validate(summary_data)
                post.summary = _postprocess_summary(summary_output.summary)
                post.tags = summary_output.tags
                post.urls = summary_output.urls
            except Exception as e:
                logger.error(
                    "Failed to parse summary for post '%s': %s. Data: %s",
                    post.title,
                    e,
                    summary_data,
                )

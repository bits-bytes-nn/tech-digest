import re
from typing import Any, ClassVar

import boto3
import markdown
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


def _normalize_tags(tags_input: Any) -> list[str]:
    tags = []
    if isinstance(tags_input, str):
        tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()]
    elif isinstance(tags_input, list):
        for item in tags_input:
            if isinstance(item, str):
                tags.extend([tag.strip() for tag in item.split(",") if tag.strip()])
    return tags


# Deterministic, declarative summary fix-ups. Kept as an explicit, documented
# map (rather than chained inline .replace() calls) so each correction states
# its intent and new ones can be added without touching control flow.
_SUMMARY_REPLACEMENTS: dict[str, str] = {
    # The model occasionally ends a Korean sentence with a colon where a period
    # is correct ("...합니다:" -> "...합니다.").
    "다:": "다.",
    "요:": "요.",
    # Normalize a known source-URL variant to its canonical host.
    "magazine.sebastianraschka": "sebastianraschka",
}


def _postprocess_summary(summary: str) -> str:
    for old, new in _SUMMARY_REPLACEMENTS.items():
        summary = summary.replace(old, new)
    return summary


def _normalize_urls(urls_input: Any) -> list[str]:
    urls = []
    if isinstance(urls_input, str):
        markdown_links = re.findall(r"\[(.*?)]\((.*?)\)", urls_input)
        if markdown_links:
            return [f'<a href="{url}">{desc}</a>' for desc, url in markdown_links]
        urls = [url.strip() for url in urls_input.split(",") if url.strip()]
    elif isinstance(urls_input, list):
        for item in urls_input:
            if isinstance(item, str):
                urls.append(item.strip())
    return urls


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
            return html
        except Exception as e:
            logger.error("Error converting markdown to HTML: %s", e)
            return v

    @field_validator("tags", mode="before")
    def _validate_tags(cls, v: Any) -> list[str]:
        tags = _normalize_tags(v)
        unique_tags = sorted(set(tags))
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

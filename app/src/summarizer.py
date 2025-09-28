import re
from typing import Any

import boto3
import markdown
from langchain.output_parsers import OutputFixingParser
from pydantic import BaseModel, Field, field_validator

from .constants import FilteringCriteria, Language, LanguageModelId
from .feed_parser import Post
from .logger import logger
from .prompts.prompts import FilteringPrompt, SummarizationPrompt
from .utils import (
    BedrockLanguageModelFactory,
    BatchProcessor,
    HTMLTagOutputParser,
    measure_execution_time,
)


class SummarizerConfig:
    CONVERT_MARKDOWN_TO_HTML: bool = True
    DEFAULT_TAGS: list[str] = ["uncategorized"]
    HTML_REPLACEMENTS: dict[str, str] = {
        "<code>": '<code class="highlight">',
        "<table>": '<table class="table table-bordered">',
        "<pre>": '<pre class="pre-scrollable">',
    }
    MARKDOWN_EXTENSIONS: list[str] = [
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
    MAX_TAGS: int = 5


def _normalize_tags(tags_input: Any) -> list[str]:
    tags = []
    if isinstance(tags_input, str):
        tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()]
    elif isinstance(tags_input, list):
        for item in tags_input:
            if isinstance(item, str):
                tags.extend([tag.strip() for tag in item.split(",") if tag.strip()])
    return tags


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
        unique_tags = sorted(list(set(tags)))
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
        language: Language = Language.KO,
        max_concurrency: int = 10,
        min_score: float = 0.7,
    ) -> None:
        self.boto_session = boto_session
        self.filtering_criteria = filtering_criteria
        self.language = language
        self.min_score = min_score
        self.filtered_out_posts: list[tuple[Post, str]] = []
        self.batch_processor = BatchProcessor(
            max_concurrency=max_concurrency, batch_size=max_concurrency
        )
        self.llm_factory = BedrockLanguageModelFactory(boto_session=self.boto_session)
        self.filter = self._create_filter(filtering_model_id)
        self.summarizer = self._create_summarizer(
            summarization_model_id, fixing_model_id
        )

    def _create_filter(self, model_id: LanguageModelId):
        model_info = self.llm_factory.get_model_info(model_id)
        if not model_info:
            raise ValueError(f"Invalid model ID: '{model_id}'")
        filtering_llm = self.llm_factory.get_model(model_id=model_id, temperature=0.0)
        return (
            FilteringPrompt.for_criteria(self.filtering_criteria).get_prompt()
            | filtering_llm
            | HTMLTagOutputParser(tag_names=FilteringPrompt.output_variables)
        )

    def _create_summarizer(
        self, model_id: LanguageModelId, fixing_model_id: LanguageModelId | None
    ):
        model_info = self.llm_factory.get_model_info(model_id)
        if not model_info:
            raise ValueError(f"Invalid model ID: '{model_id}'")
        fixing_model_id = fixing_model_id or LanguageModelId.CLAUDE_V3_5_HAIKU
        fixing_llm = self.llm_factory.get_model(
            model_id=LanguageModelId(fixing_model_id), temperature=0.0
        )
        output_parser = OutputFixingParser.from_llm(
            llm=fixing_llm,
            parser=HTMLTagOutputParser(tag_names=SummarizationPrompt.output_variables),
        )
        summarization_llm = self.llm_factory.get_model(
            model_id=model_id, temperature=0.0
        )
        return (
            SummarizationPrompt.for_language(self.language).get_prompt()
            | summarization_llm
            | output_parser
        )

    @measure_execution_time
    def process_posts(
        self,
        posts: list[Post],
        use_filtering: bool,
        included_topics: list[str],
        excluded_topics: list[str],
    ) -> list[Post]:
        if use_filtering:
            posts_to_process = self._filter_posts(
                posts, included_topics, excluded_topics
            )
            if not posts_to_process:
                logger.warning("No posts remained after filtering.")
                return []
        else:
            posts_to_process = [p for p in posts if p.content]
        self._summarize_posts(posts_to_process)
        return posts_to_process

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
        for post, response in zip(valid_posts, responses):
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
        for post, summary_data in zip(posts, summaries):
            try:
                summary_output = SummaryOutput.model_validate(summary_data)
                post.summary = summary_output.summary.replace("다:", "다.").replace(
                    "요:", "요."
                )
                post.tags = summary_output.tags
                post.urls = summary_output.urls
            except Exception as e:
                logger.error(
                    "Failed to parse summary for post '%s': %s. Data: %s",
                    post.title,
                    e,
                    summary_data,
                )

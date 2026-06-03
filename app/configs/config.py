import os
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, EmailStr, Field, model_validator
from pydantic_core import PydanticUndefined

from app.src import EnvVars, FilteringCriteria, LanguageModelId


class BaseModelWithDefaults(BaseModel):
    @model_validator(mode="before")
    def set_defaults_for_none_fields(cls, values: dict[str, Any]) -> dict[str, Any]:
        # Treat an explicit ``null`` in the YAML the same as an omitted key, so a
        # field's declared default (or default_factory) takes effect instead of
        # leaking None into a non-Optional field. Handle BOTH default forms:
        # ``field.default`` for plain defaults and ``field.default_factory`` for
        # factory defaults (e.g. list/dict fields) — the latter was previously
        # ignored, so ``rss_urls: null`` would surface None instead of [].
        if not isinstance(values, dict):
            return values
        for field_name, field in cls.model_fields.items():
            # Only act when the key is present AND explicitly None; an omitted
            # key is left for pydantic to default normally.
            if field_name not in values or values.get(field_name) is not None:
                continue
            if field.default is not None and field.default is not PydanticUndefined:
                values[field_name] = field.default
            elif field.default_factory is not None:
                values[field_name] = field.default_factory()  # type: ignore[call-arg]
        return values


class Resources(BaseModelWithDefaults):
    project_name: str = Field(min_length=1)
    stage: Literal["dev", "prod"] = Field(default="dev")
    profile_name: str | None = Field(default=None)
    default_region_name: str = Field(default="ap-northeast-2")
    bedrock_region_name: str = Field(default="us-west-2")
    s3_bucket_name: str = Field(min_length=1)
    s3_prefix: str | None = Field(default=None)
    vpc_id: str | None = Field(default=None)
    subnet_ids: list[str] | None = Field(default=None)
    lambda_or_batch: Literal["lambda", "batch"] = Field(default="lambda")
    cron_expression: str | None = Field(
        default=None,
        pattern=r"^cron\(([0-9]|[1-5][0-9]) ([0-9]|1[0-9]|2[0-3]) (([1-9]|[12][0-9]|3[01])|\*|\?) (([1-9]|1[0-2])|\*) (([1-7])|\*|\?) (\*|[0-9]{4})\)$",
    )


class Scraping(BaseModelWithDefaults):
    rss_urls: list[str]
    days_back: int = Field(default=7, ge=1)
    # Minimum length of *visible* article text (markup stripped) for a post to
    # be considered substantive enough to summarize. Posts below this are
    # dropped during filtering so they never reach the summarizer, which would
    # otherwise emit an empty/low-quality summary ("article too short").
    #
    # Rationale for 600: an RSS teaser / link-blog stub is typically 1-3 short
    # paragraphs (~300-500 visible chars); a real article with enough substance
    # to summarize meaningfully reliably clears ~600. It is intentionally a
    # CHARACTER count (not tokens) for speed and dependency-freedom.
    #
    # KNOWN BIAS: character counting under-counts information density for CJK
    # languages (Korean/Chinese/Japanese express the same content in far fewer
    # characters than English). Sources like kakao, ncsoft and qwen can thus be
    # gated more aggressively than English sources at the same threshold. If you
    # crawl primarily CJK sources, lower this (e.g. 300-400) per stage.
    min_content_length: int = Field(default=600, ge=0)


class Summarization(BaseModelWithDefaults):
    use_filtering: bool = Field(default=True)
    filtering_criteria: FilteringCriteria
    included_topics: list[str] = Field(default_factory=list)
    excluded_topics: list[str] = Field(default_factory=list)
    filtering_model_id: LanguageModelId
    filtering_enable_thinking: bool = Field(default=False)
    summarization_model_id: LanguageModelId
    summarization_enable_thinking: bool = Field(default=False)
    greeting_model_id: LanguageModelId
    max_concurrency: int = Field(default=10, ge=1)
    min_score: float = Field(default=0.7, ge=0.0, le=1.0)
    max_posts: int | None = Field(default=None, ge=1)
    # Extended-thinking budget (tokens) for the filtering/summarization models
    # when *_enable_thinking is on. The model factory's hardcoded 2048 default
    # is too small to meaningfully reason over a full article; expose it so it
    # can be tuned per stage. Must be < the model's max output tokens.
    filtering_thinking_budget_tokens: int = Field(default=4096, ge=1024)
    summarization_thinking_budget_tokens: int = Field(default=8192, ge=1024)
    # Optional cap on summary output tokens (None = use the model's maximum).
    summarization_max_tokens: int | None = Field(default=None, ge=256)


class Newsletter(BaseModelWithDefaults):
    send_emails: bool = Field(default=False)
    save_articles: bool = Field(default=False)
    convert_to_images: bool = Field(default=False)
    sender: EmailStr | None = Field(default=None)
    header_title: str | None = Field(default=None, min_length=1)
    header_description: str | None = Field(default=None, min_length=1)
    header_thumbnail: str | None = Field(default=None, min_length=1)
    footer_title: str | None = Field(default=None, min_length=1)
    logos: dict[str, str] = Field(default_factory=dict)


class Config(BaseModelWithDefaults):
    resources: Resources = Field(
        default_factory=lambda: Resources(project_name="tech-digest", s3_bucket_name="")
    )
    scraping: Scraping = Field(default_factory=lambda: Scraping(rss_urls=[]))
    summarization: Summarization = Field(
        default_factory=lambda: Summarization(
            filtering_criteria=FilteringCriteria.ALL,
            filtering_model_id=LanguageModelId.CLAUDE_V4_6_SONNET,
            summarization_model_id=LanguageModelId.CLAUDE_V4_6_SONNET,
            greeting_model_id=LanguageModelId.CLAUDE_V4_5_HAIKU,
        )
    )
    newsletter: Newsletter = Field(default_factory=Newsletter)

    @classmethod
    def from_yaml(cls, file_path: str) -> "Config":
        with open(file_path, encoding="utf-8") as file:
            config_data = yaml.safe_load(file)
        return cls(**config_data)

    @classmethod
    def load(cls) -> "Config":
        load_dotenv()
        config_suffix = os.environ.get(EnvVars.CONFIG_FILE_SUFFIX.value, "dev")
        filename, extension = "config", "yaml"
        suffix = f"-{config_suffix}" if config_suffix else ""
        config_file = (
            Path(filename).with_name(f"{filename}{suffix}").with_suffix(f".{extension}")
        )
        config_path = Path(__file__).parent / config_file
        return cls.from_yaml(str(config_path))

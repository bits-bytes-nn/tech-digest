import os
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, EmailStr, Field, model_validator

from app.src import EnvVars, FilteringCriteria, LanguageModelId


class BaseModelWithDefaults(BaseModel):
    @model_validator(mode="before")
    def set_defaults_for_none_fields(cls, values: dict[str, Any]) -> dict[str, Any]:
        for field_name, field in cls.model_fields.items():
            if values.get(field_name) is None and field.default is not None:
                values[field_name] = field.default
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


class Summarization(BaseModelWithDefaults):
    use_filtering: bool = Field(default=True)
    filtering_criteria: FilteringCriteria
    included_topics: list[str] = Field(default_factory=list)
    excluded_topics: list[str] = Field(default_factory=list)
    filtering_model_id: LanguageModelId
    summarization_model_id: LanguageModelId
    greeting_model_id: LanguageModelId
    max_concurrency: int = Field(default=10, ge=1)
    min_score: float = Field(default=0.7, ge=0.0, le=1.0)
    max_posts: int | None = Field(default=None, ge=1)


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
            filtering_model_id=LanguageModelId.CLAUDE_V4_5_SONNET,
            summarization_model_id=LanguageModelId.CLAUDE_V4_5_SONNET,
            greeting_model_id=LanguageModelId.CLAUDE_V3_5_HAIKU,
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

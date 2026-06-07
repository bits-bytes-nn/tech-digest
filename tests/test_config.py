"""Unit tests for configuration loading and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.configs.config import (
    Config,
    Newsletter,
    Resources,
    Scraping,
    Summarization,
)
from app.src.constants import FilteringCriteria, LanguageModelId


class TestResources:
    def test_minimal_valid(self):
        r = Resources(project_name="tech-digest", s3_bucket_name="my-bucket")
        assert r.stage == "dev"
        assert r.default_region_name == "ap-northeast-2"
        assert r.lambda_or_batch == "lambda"

    def test_invalid_stage_rejected(self):
        with pytest.raises(ValidationError):
            Resources(
                project_name="x",
                s3_bucket_name="b",
                stage="staging",  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize(
        "cron,ok",
        [
            ("cron(0 1 ? * 6 *)", True),
            ("cron(0 1 ? * 6 2026)", True),
            # Valid AWS cron forms the old enumerated regex wrongly rejected:
            ("cron(*/30 * * * ? *)", True),  # step
            ("cron(0 1 ? * MON *)", True),  # day-of-week name
            ("cron(0 9,17 ? * 2-6 *)", True),  # list + range
            ("cron(0 1 L * ? *)", True),  # last-day-of-month
            ("not-a-cron", False),
            ("cron(0 1 ? * 6)", False),  # only 5 fields
            ("cron(0 1 ? * 6 * *)", False),  # 7 fields
            ("cron()", False),
        ],
    )
    def test_cron_expression_pattern(self, cron, ok):
        if ok:
            Resources(project_name="x", s3_bucket_name="b", cron_expression=cron)
        else:
            with pytest.raises(ValidationError):
                Resources(project_name="x", s3_bucket_name="b", cron_expression=cron)


class TestScraping:
    def test_days_back_minimum(self):
        with pytest.raises(ValidationError):
            Scraping(rss_urls=[], days_back=0)

    def test_default_days_back(self):
        assert Scraping(rss_urls=["https://x.com/feed"]).days_back == 7


class TestSummarization:
    def test_score_bounds(self):
        with pytest.raises(ValidationError):
            Summarization(
                filtering_criteria=FilteringCriteria.ALL,
                filtering_model_id=LanguageModelId.CLAUDE_V4_6_SONNET,
                summarization_model_id=LanguageModelId.CLAUDE_V4_6_SONNET,
                greeting_model_id=LanguageModelId.CLAUDE_V4_5_HAIKU,
                min_score=1.5,
            )

    def test_max_posts_optional(self):
        s = Summarization(
            filtering_criteria=FilteringCriteria.ALL,
            filtering_model_id=LanguageModelId.CLAUDE_V4_6_SONNET,
            summarization_model_id=LanguageModelId.CLAUDE_V4_6_SONNET,
            greeting_model_id=LanguageModelId.CLAUDE_V4_5_HAIKU,
        )
        assert s.max_posts is None


class TestNewsletter:
    def test_sender_required_when_sending(self):
        # send_emails=True without a sender would make deploy_infra build the SES
        # FromAddress condition / SNS subscription from str(None) == "None".
        with pytest.raises(ValidationError):
            Newsletter(send_emails=True)

    def test_sender_optional_when_not_sending(self):
        n = Newsletter(send_emails=False)
        assert n.sender is None

    def test_sender_accepted_when_sending(self):
        n = Newsletter(send_emails=True, sender="ok@example.com")
        assert str(n.sender) == "ok@example.com"


class TestExplicitNullCoercion:
    """An explicit ``null`` in YAML should fall back to the field's default
    (including default_factory), not leak None into a non-Optional field."""

    def test_explicit_null_factory_dict_field(self):
        # logos has default_factory=dict; explicit None must become {}.
        assert Newsletter(logos=None).logos == {}

    def test_explicit_null_factory_list_fields(self):
        s = Summarization(
            filtering_criteria=FilteringCriteria.ALL,
            filtering_model_id=LanguageModelId.CLAUDE_V4_6_SONNET,
            summarization_model_id=LanguageModelId.CLAUDE_V4_6_SONNET,
            greeting_model_id=LanguageModelId.CLAUDE_V4_5_HAIKU,
            included_topics=None,
            excluded_topics=None,
        )
        assert s.included_topics == []
        assert s.excluded_topics == []

    def test_explicit_null_plain_default(self):
        # days_back has a plain default of 7; explicit None must become 7.
        assert Scraping(rss_urls=["https://x.com/feed"], days_back=None).days_back == 7


class TestConfigFromYaml:
    def test_loads_dev_yaml_if_present(self, tmp_path):
        yaml_content = """
resources:
  project_name: tech-digest
  s3_bucket_name: my-bucket
scraping:
  rss_urls:
    - "https://openai.com/news/rss.xml"
  days_back: 7
summarization:
  filtering_criteria: all
  filtering_model_id: anthropic.claude-sonnet-4-6
  summarization_model_id: anthropic.claude-sonnet-4-6
  greeting_model_id: anthropic.claude-haiku-4-5-20251001-v1:0
newsletter:
  sender: "test@example.com"
"""
        p = tmp_path / "config-test.yaml"
        p.write_text(yaml_content, encoding="utf-8")
        config = Config.from_yaml(str(p))
        assert config.resources.project_name == "tech-digest"
        assert config.summarization.min_score == 0.7
        assert len(config.scraping.rss_urls) == 1

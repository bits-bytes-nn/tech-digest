"""Unit tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.configs.config import (
    Config,
    Newsletter,
    Resources,
    Scraping,
    Summarization,
)
from app.domain.constants import FilteringCriteria, LanguageModelId


class TestResources:
    def test_minimal_valid(self):
        r = Resources(project_name="tech-digest", s3_bucket_name="my-bucket")
        assert r.stage == "dev"
        assert r.default_region_name == "ap-northeast-2"
        assert r.langchain_tracing is False

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


class TestSummarizationSwitchesReachTheSummarizer:
    """A knob documented in the template must survive config load.

    ``filter_on_visible_text`` and ``summarize_on_cleaned_html`` were documented
    in config-template.yaml and design.md as switches with a "set False to revert"
    instruction, but were never declared on the Summarization model — so
    pydantic's ignore-extras dropped them and SummarizerSettings fell back to its
    own default. The defaults matched, so the pipeline looked correct while both
    switches did nothing at all.
    """

    @pytest.mark.parametrize(
        "field", ["filter_on_visible_text", "summarize_on_cleaned_html"]
    )
    def test_switch_is_a_declared_field(self, field):
        assert field in Summarization.model_fields

    @pytest.mark.parametrize(
        "field", ["filter_on_visible_text", "summarize_on_cleaned_html"]
    )
    def test_false_in_yaml_reaches_summarizer_settings(self, field):
        from app.application.summarize import SummarizerSettings

        section = Summarization(
            filtering_criteria=FilteringCriteria.ALL,
            filtering_model_id=LanguageModelId.CLAUDE_V5_SONNET,
            summarization_model_id=LanguageModelId.CLAUDE_V5_SONNET,
            greeting_model_id=LanguageModelId.CLAUDE_V4_5_HAIKU,
            **{field: False},
        )
        settings = SummarizerSettings.model_validate(section.model_dump())
        assert getattr(settings, field) is False

    def test_template_documents_only_switches_that_exist(self):
        """Every summarization key in the shipped template must be a real field."""
        import yaml

        template = yaml.safe_load(
            (
                Path(__file__).resolve().parent.parent
                / "app"
                / "configs"
                / "config-template.yaml"
            ).read_text(encoding="utf-8")
        )
        unknown = set(template["summarization"]) - set(Summarization.model_fields)
        assert not unknown, (
            f"config-template.yaml documents summarization keys the model ignores: "
            f"{sorted(unknown)}"
        )


_VALID_SUMMARIZATION = {
    "filtering_criteria": "all",
    "filtering_model_id": "anthropic.claude-haiku-5-5",
    "summarization_model_id": "anthropic.claude-sonnet-5-5",
    "greeting_model_id": "anthropic.claude-haiku-5-5",
}


class TestNewestModelsAreSelectable:
    """The newest Claude and GPT ids must load from YAML for every stage, and the
    shipped configs must agree with the code defaults."""

    @pytest.mark.parametrize(
        "model_value",
        [
            "anthropic.claude-haiku-5-5",
            "anthropic.claude-sonnet-5-5",
            "anthropic.claude-opus-5-5",
            "anthropic.claude-fable-5-1",
            "openai.gpt-6.1-sol",
            "openai.gpt-6-sol",
            "openai.gpt-6-luna",
            "openai.gpt-6-astra",
        ],
    )
    def test_every_stage_accepts_the_id(self, model_value):
        s = Summarization.model_validate(
            {
                "filtering_criteria": "all",
                "filtering_model_id": model_value,
                "summarization_model_id": model_value,
                "greeting_model_id": model_value,
            }
        )
        assert s.filtering_model_id.value == model_value
        assert s.summarization_model_id.value == model_value
        assert s.greeting_model_id.value == model_value

    @pytest.mark.parametrize("effort", [None, "low", "medium", "high", "xhigh", "max"])
    def test_every_effort_level_loads(self, effort):
        s = Summarization.model_validate(
            {**_VALID_SUMMARIZATION, "thinking_effort": effort}
        )
        assert s.thinking_effort == effort

    def test_an_unknown_effort_is_rejected(self):
        with pytest.raises(ValidationError, match="thinking_effort"):
            Summarization.model_validate(
                {**_VALID_SUMMARIZATION, "thinking_effort": "extreme"}
            )

    @staticmethod
    def _summarization(name: str) -> dict:
        import yaml

        path = Path(__file__).resolve().parent.parent / "app" / "configs" / name
        return yaml.safe_load(path.read_text(encoding="utf-8"))["summarization"]

    def test_the_template_defaults_to_the_chosen_models(self):
        section = self._summarization("config-template.yaml")
        assert section["filtering_model_id"] == LanguageModelId.GPT_6_1_SOL.value
        assert (
            section["summarization_model_id"]
            == LanguageModelId.CLAUDE_V5_5_SONNET.value
        )
        assert section["greeting_model_id"] == LanguageModelId.CLAUDE_V5_5_HAIKU.value

    @pytest.mark.parametrize("name", ["config-ci-batch.yaml"])
    def test_ci_configs_use_the_template_models(self, name):
        template = self._summarization("config-template.yaml")
        section = self._summarization(name)
        for key in (
            "filtering_model_id",
            "summarization_model_id",
            "greeting_model_id",
        ):
            assert section[key] == template[key], (name, key)

    def test_a_config_missing_a_section_is_rejected(self):
        with pytest.raises(ValidationError):
            Config.model_validate(
                {"resources": {"project_name": "p", "s3_bucket_name": "b"}}
            )


class TestCommittedSourcesReachTheirScraper:
    """A URL on a host that has a hand-written scraper must be routed to it.

    The template listed ``https://www.anthropic.com/news`` while the scraper is
    registered for ``/engineering``, so the URL fell through to the RSS fetcher,
    which found no entries in an HTML page — and ``expected_flaky_urls`` then
    muted the resulting "likely broken" alert. Anyone copying the template got a
    dead source that could never report itself.
    """

    @pytest.mark.parametrize(
        "config_file",
        ["config-template.yaml", "config-ci-batch.yaml"],
    )
    def test_no_scraper_host_falls_through_to_rss(self, config_file):
        from urllib.parse import urlparse

        import yaml

        from app.adapters.feeds import RssFetcher, ScraperRegistry

        path = Path(__file__).resolve().parent.parent / "app" / "configs" / config_file
        urls = yaml.safe_load(path.read_text())["scraping"]["rss_urls"]
        scraper_hosts = {
            pattern.split("/", 1)[0] for pattern in ScraperRegistry._SCRAPER_MAPPING
        }
        for url in urls:
            if urlparse(url).netloc in scraper_hosts:
                fetcher = ScraperRegistry.get_fetcher(url)
                assert not isinstance(fetcher, RssFetcher), url


class TestS3PrefixIsNormalisedOnce:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("tech-digest/", "tech-digest"),
            ("/tech-digest", "tech-digest"),
            ("/", None),
            (None, None),
        ],
    )
    def test_slashes_are_stripped(self, raw, expected):
        r = Resources(project_name="p", s3_bucket_name="b", s3_prefix=raw)
        assert r.s3_prefix == expected


def test_a_misspelt_key_is_rejected_not_ignored():
    with pytest.raises(ValidationError, match="min_scor"):
        Summarization.model_validate({**_VALID_SUMMARIZATION, "min_scor": 0.9})

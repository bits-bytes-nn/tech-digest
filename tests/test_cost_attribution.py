"""Bedrock cost attribution.

On-demand `InvokeModel` bills against no taggable resource, so its token spend
cannot carry a cost-allocation tag. An APPLICATION inference profile is taggable,
and invoking through its ARN attributes the usage — so the resolver prefers one
when it exists. Because that changes which ARN every call uses, the fallback
behaviour has to be airtight: cost reporting must never stop a generation.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.src.constants import LanguageModelId
from app.src.model_factory import BedrockCrossRegionModelHelper, TokenUsageLogger

MODEL = LanguageModelId.CLAUDE_V5_SONNET
# _build_cross_region_model_id uses region[:2] outside ap-*, so us-west-2 -> "us."
SYSTEM_PROFILE = f"us.{MODEL.value}"
APP_ARN = "arn:aws:bedrock:us-west-2:111111111111:application-inference-profile/abc123"


@pytest.fixture(autouse=True)
def _clear_caches(monkeypatch):
    BedrockCrossRegionModelHelper._profile_set_cache.clear()
    BedrockCrossRegionModelHelper._application_profile_cache.clear()
    monkeypatch.setenv("PROJECT_NAME", "tech-digest")
    monkeypatch.setenv("STAGE", "dev")
    yield
    BedrockCrossRegionModelHelper._profile_set_cache.clear()
    BedrockCrossRegionModelHelper._application_profile_cache.clear()


class _FakePaginator:
    def __init__(self, pages):
        self._pages = pages

    def paginate(self, **kwargs):
        assert kwargs.get("typeEquals") == "APPLICATION"
        return self._pages


class _FakeBedrock:
    def __init__(self, system_profiles=(), application=(), raise_on_app=None):
        self._system = system_profiles
        self._application = application
        self._raise_on_app = raise_on_app
        self.app_list_calls = 0

    def list_inference_profiles(self, **kwargs):
        return {
            "inferenceProfileSummaries": [
                {"inferenceProfileId": p} for p in self._system
            ]
        }

    def get_paginator(self, name):
        self.app_list_calls += 1
        if self._raise_on_app:
            raise self._raise_on_app
        return _FakePaginator([{"inferenceProfileSummaries": list(self._application)}])


class _FakeSession:
    def __init__(self, bedrock):
        self.bedrock = bedrock

    def client(self, service, region_name=None):
        assert service == "bedrock"
        return self.bedrock


class TestApplicationProfileName:
    def test_deterministic_and_slugified(self, monkeypatch):
        name = BedrockCrossRegionModelHelper.application_profile_name(MODEL)
        # Non-alphanumerics in the model id become single dashes so the name is a
        # legal profile name; the script and the runtime derive it identically.
        assert name == "tech-digest-dev-anthropic-claude-sonnet-5"

    def test_follows_project_and_stage(self, monkeypatch):
        monkeypatch.setenv("PROJECT_NAME", "other")
        monkeypatch.setenv("STAGE", "prod")
        assert BedrockCrossRegionModelHelper.application_profile_name(MODEL).startswith(
            "other-prod-"
        )

    def test_falls_back_to_config_file_suffix_for_stage(self, monkeypatch):
        """Local runs set CONFIG_FILE_SUFFIX but not STAGE."""
        monkeypatch.delenv("STAGE", raising=False)
        monkeypatch.setenv("CONFIG_FILE_SUFFIX", "prod")
        assert "-prod-" in BedrockCrossRegionModelHelper.application_profile_name(MODEL)

    def test_versioned_model_ids_slugify_without_colons(self):
        name = BedrockCrossRegionModelHelper.application_profile_name(
            LanguageModelId.CLAUDE_V4_5_HAIKU
        )
        assert ":" not in name and "." not in name


class TestResolverPrefersApplicationProfile:
    def test_application_profile_arn_used_when_present(self):
        bedrock = _FakeBedrock(
            system_profiles=[SYSTEM_PROFILE],
            application=[
                {
                    "inferenceProfileName": (
                        "tech-digest-dev-anthropic-claude-sonnet-5"
                    ),
                    "inferenceProfileArn": APP_ARN,
                }
            ],
        )
        resolved = BedrockCrossRegionModelHelper.get_cross_region_model_id(
            _FakeSession(bedrock), MODEL, "us-west-2"
        )
        assert resolved == APP_ARN

    def test_falls_back_to_system_profile_when_absent(self):
        """Running without the profiles provisioned must behave exactly as before."""
        bedrock = _FakeBedrock(system_profiles=[SYSTEM_PROFILE], application=[])
        resolved = BedrockCrossRegionModelHelper.get_cross_region_model_id(
            _FakeSession(bedrock), MODEL, "us-west-2"
        )
        assert resolved == SYSTEM_PROFILE

    def test_denied_lookup_never_breaks_resolution(self):
        """A missing bedrock:ListInferenceProfiles permission on APPLICATION
        profiles must not fail the call — cost reporting is not worth an outage."""
        bedrock = _FakeBedrock(
            system_profiles=[SYSTEM_PROFILE],
            raise_on_app=RuntimeError("AccessDeniedException"),
        )
        resolved = BedrockCrossRegionModelHelper.get_cross_region_model_id(
            _FakeSession(bedrock), MODEL, "us-west-2"
        )
        assert resolved == SYSTEM_PROFILE

    def test_profile_for_a_different_project_is_ignored(self):
        bedrock = _FakeBedrock(
            system_profiles=[SYSTEM_PROFILE],
            application=[
                {
                    "inferenceProfileName": "omnisummary-dev-anthropic-claude-sonnet-5",
                    "inferenceProfileArn": "arn:aws:bedrock:us-west-2:1:application-inference-profile/other",
                }
            ],
        )
        resolved = BedrockCrossRegionModelHelper.get_cross_region_model_id(
            _FakeSession(bedrock), MODEL, "us-west-2"
        )
        assert resolved == SYSTEM_PROFILE

    def test_absence_is_cached_so_every_model_build_does_not_re_list(self):
        bedrock = _FakeBedrock(system_profiles=[SYSTEM_PROFILE], application=[])
        session = _FakeSession(bedrock)
        for _ in range(4):
            BedrockCrossRegionModelHelper.get_cross_region_model_id(
                session, MODEL, "us-west-2"
            )
        assert bedrock.app_list_calls == 1


class TestTokenUsageLogger:
    """Cost Explorer bills per model; filtering, summarization and output-fixing
    share models, so per-stage usage has to come from the logs."""

    def _response(self, usage):
        """Minimal stand-in for a LangChain LLMResult carrying usage metadata."""
        msg = SimpleNamespace(usage_metadata=usage)
        generation = SimpleNamespace(message=msg)
        return SimpleNamespace(generations=[[generation]], llm_output=None)

    def test_logs_stage_and_token_counts(self, caplog, propagating_logger):
        handler = TokenUsageLogger("filtering", "apac.anthropic.claude-sonnet-5")
        with caplog.at_level("INFO"):
            handler.on_llm_end(
                self._response(
                    {
                        "input_tokens": 9000,
                        "output_tokens": 210,
                        "input_token_details": {"cache_read": 1024},
                    }
                )
            )
        message = "\n".join(r.getMessage() for r in caplog.records)
        assert "stage=filtering" in message
        assert "input=9000" in message
        assert "output=210" in message
        assert "cache_read=1024" in message

    def test_missing_usage_is_silent(self, caplog, propagating_logger):
        handler = TokenUsageLogger("summarization", "m")
        with caplog.at_level("INFO"):
            handler.on_llm_end(self._response(None))
        assert not [r for r in caplog.records if "LLM usage" in r.getMessage()]

    def test_malformed_response_never_raises(self):
        """A telemetry callback must not be able to fail a generation."""
        TokenUsageLogger("greeting", "m").on_llm_end(object())

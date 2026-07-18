"""Unit tests for the Bedrock model-info registry and cross-region helpers.

These avoid real boto3/Bedrock calls by testing the pure logic: the model-info
table, max-token validation, and cross-region id construction.
"""

from __future__ import annotations

import app.src.utils as utils_mod
from app.src.constants import LanguageModelId
from app.src.utils import (
    _LANGUAGE_MODEL_INFO,
    BedrockCrossRegionModelHelper,
    BedrockLanguageModelFactory,
)


class TestModelInfoRegistry:
    def test_every_model_id_has_info(self):
        """Every enum member that is meant to be usable has a LanguageModelInfo."""
        missing = [
            m
            for m in LanguageModelId
            if m not in _LANGUAGE_MODEL_INFO
            # legacy models intentionally without info are acceptable, but the
            # current/default models must be present.
            and m
            in {
                LanguageModelId.CLAUDE_V4_6_SONNET,
                LanguageModelId.CLAUDE_V4_5_HAIKU,
                LanguageModelId.CLAUDE_V4_6_OPUS,
            }
        ]
        assert not missing, f"Default models missing info: {missing}"

    def test_info_fields_sane(self):
        for model_id, info in _LANGUAGE_MODEL_INFO.items():
            assert info.context_window_size >= 100_000, model_id
            assert info.max_output_tokens > 0, model_id


class TestValidateMaxTokens:
    def test_caps_to_model_max(self):
        info = _LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V4_6_SONNET]
        capped = BedrockLanguageModelFactory._validate_max_tokens(10_000_000, info)
        assert capped == info.max_output_tokens

    def test_none_uses_model_default(self):
        info = _LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V4_6_SONNET]
        assert (
            BedrockLanguageModelFactory._validate_max_tokens(None, info)
            == info.max_output_tokens
        )


class TestThinkingBudgetClamp:
    """The thinking budget must stay below max_tokens, else Bedrock 400s."""

    def _factory(self):
        # Bypass __init__ (which builds a boto client) — _apply_model_features
        # only needs the class methods/constants.
        return BedrockLanguageModelFactory.__new__(BedrockLanguageModelFactory)

    def test_budget_clamped_below_max_tokens(self):
        factory = self._factory()
        info = _LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V4_6_SONNET]
        config: dict = {"model_kwargs": {}}
        factory._apply_model_features(
            config,
            info,
            is_cross_region=False,
            enable_thinking=True,
            thinking_budget_tokens=10_000,
            max_tokens=4096,
        )
        budget = config["model_kwargs"]["thinking"]["budget_tokens"]
        assert budget < 4096
        assert budget >= 1024

    def test_budget_preserved_when_under_max(self):
        factory = self._factory()
        info = _LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V4_6_SONNET]
        config: dict = {"additional_model_request_fields": {}}
        factory._apply_model_features(
            config,
            info,
            is_cross_region=True,
            enable_thinking=True,
            thinking_budget_tokens=8192,
            max_tokens=64000,
        )
        think = config["additional_model_request_fields"]["thinking"]
        assert think["budget_tokens"] == 8192

    def test_thinking_disabled_when_max_tokens_too_small(self):
        # max_tokens below the minimum budget can't host any valid budget, so
        # thinking must be skipped rather than sending an invalid request.
        factory = self._factory()
        info = _LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V4_6_SONNET]
        config: dict = {"model_kwargs": {}}
        factory._apply_model_features(
            config,
            info,
            is_cross_region=False,
            enable_thinking=True,
            thinking_budget_tokens=4096,
            max_tokens=512,  # < MIN_THINKING_BUDGET
        )
        assert "thinking" not in config["model_kwargs"]

    def test_no_thinking_block_when_unsupported(self):
        factory = self._factory()
        # A model without thinking support must not get a thinking block.
        info = _LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V3_5_SONNET]
        config: dict = {"model_kwargs": {}}
        factory._apply_model_features(
            config, info, is_cross_region=False, enable_thinking=True
        )
        assert "thinking" not in config["model_kwargs"]


class TestAdaptiveThinking:
    """Newer models (Sonnet 5+) reject the legacy thinking form
    (thinking.type="enabled" + budget_tokens) and require adaptive thinking.
    This guards the second regression that blocked summarization after the
    Sonnet 5 switch: every summarize call 400'd with
    "thinking.type.enabled is not supported for this model."
    """

    def _factory(self):
        return BedrockLanguageModelFactory.__new__(BedrockLanguageModelFactory)

    def test_adaptive_model_uses_adaptive_type_cross_region(self):
        factory = self._factory()
        info = _LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V5_SONNET]
        assert info.uses_adaptive_thinking is True
        config: dict = {"additional_model_request_fields": {}}
        factory._apply_model_features(
            config,
            info,
            is_cross_region=True,
            enable_thinking=True,
            thinking_budget_tokens=8192,
            max_tokens=64000,
        )
        thinking = config["additional_model_request_fields"]["thinking"]
        assert thinking == {"type": "adaptive"}
        assert "budget_tokens" not in thinking

    def test_adaptive_model_applies_effort_when_given(self):
        factory = self._factory()
        info = _LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V5_SONNET]
        config: dict = {"additional_model_request_fields": {}}
        factory._apply_model_features(
            config,
            info,
            is_cross_region=True,
            enable_thinking=True,
            effort="high",
        )
        amrf = config["additional_model_request_fields"]
        assert amrf["thinking"] == {"type": "adaptive"}
        assert amrf["output_config"] == {"effort": "high"}

    def test_adaptive_model_omits_effort_by_default(self):
        factory = self._factory()
        info = _LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V5_SONNET]
        config: dict = {"model_kwargs": {}}
        factory._apply_model_features(
            config, info, is_cross_region=False, enable_thinking=True
        )
        assert config["model_kwargs"]["thinking"] == {"type": "adaptive"}
        assert "output_config" not in config["model_kwargs"]

    def _capture_warnings(self, caplog, factory, **kwargs):
        # The project 'app' logger has propagate=False, so caplog's root handler
        # never sees its records; attach caplog's handler to it directly.
        import logging

        info = _LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V5_SONNET]
        config: dict = {"additional_model_request_fields": {}}
        app_logger = logging.getLogger("app")
        app_logger.addHandler(caplog.handler)
        try:
            with caplog.at_level(logging.WARNING, logger="app"):
                factory._apply_model_features(
                    config, info, is_cross_region=True, enable_thinking=True, **kwargs
                )
        finally:
            app_logger.removeHandler(caplog.handler)
        return caplog.records

    def test_non_default_budget_warns_on_adaptive_model(self, caplog):
        # A tuned thinking_budget_tokens is a no-op for adaptive models; the
        # operator must be warned rather than silently ignored.
        factory = self._factory()
        records = self._capture_warnings(caplog, factory, thinking_budget_tokens=20000)
        assert any("ignored for adaptive" in r.message for r in records)

    def test_default_budget_does_not_warn_on_adaptive_model(self, caplog):
        factory = self._factory()
        records = self._capture_warnings(
            caplog,
            factory,
            thinking_budget_tokens=(
                BedrockLanguageModelFactory.DEFAULT_THINKING_BUDGET_TOKENS
            ),
        )
        assert not any("ignored for adaptive" in r.message for r in records)

    def test_legacy_model_still_uses_budget_form(self):
        factory = self._factory()
        info = _LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V4_6_SONNET]
        assert info.uses_adaptive_thinking is False
        config: dict = {"additional_model_request_fields": {}}
        factory._apply_model_features(
            config,
            info,
            is_cross_region=True,
            enable_thinking=True,
            thinking_budget_tokens=8192,
            max_tokens=64000,
        )
        thinking = config["additional_model_request_fields"]["thinking"]
        assert thinking["type"] == "enabled"
        assert thinking["budget_tokens"] == 8192


class TestSamplingParamGating:
    """Newer models (Sonnet 5+) removed the sampling params and reject requests
    that include `temperature`/`top_k`. The factory must omit them for those
    models while still sending them for transitional models that accept them.
    This guards the regression that silently produced zero-post digests: every
    Bedrock call 400'd with "`temperature` is deprecated for this model."
    """

    def _factory(self):
        factory = BedrockLanguageModelFactory.__new__(BedrockLanguageModelFactory)
        factory.region_name = "us-west-2"
        factory._client = None

        class _Session:
            profile_name = "default"

        factory.boto_session = _Session()
        return factory

    def test_sonnet_5_omits_temperature_cross_region(self):
        factory = self._factory()
        info = _LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V5_SONNET]
        assert info.supports_sampling_params is False
        config = factory._build_model_config(
            info, "us.anthropic.claude-sonnet-5", True, temperature=0.0, max_tokens=4096
        )
        assert "temperature" not in config

    def test_sonnet_5_omits_sampling_params_local(self):
        factory = self._factory()
        info = _LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V5_SONNET]
        config = factory._build_model_config(
            info, "anthropic.claude-sonnet-5", False, temperature=0.0, max_tokens=4096
        )
        model_kwargs = config["model_kwargs"]
        assert "temperature" not in model_kwargs
        assert "top_k" not in model_kwargs

    def test_transitional_model_keeps_temperature(self):
        factory = self._factory()
        info = _LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V4_6_SONNET]
        assert info.supports_sampling_params is True
        config = factory._build_model_config(
            info,
            "us.anthropic.claude-sonnet-4-6",
            True,
            temperature=0.0,
            max_tokens=4096,
        )
        assert config["temperature"] == 0.0

    def test_opus_4_8_omits_temperature(self):
        factory = self._factory()
        info = _LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V4_8_OPUS]
        assert info.supports_sampling_params is False
        config = factory._build_model_config(
            info, "us.anthropic.claude-opus-4-8", True, temperature=0.0, max_tokens=4096
        )
        assert "temperature" not in config


class TestGetModelRouting:
    """End-to-end wiring of get_model() — the composition glue that decides
    ChatBedrock vs ChatBedrockConverse and where sampling/thinking params land.
    This is exactly where the Sonnet-5 400s lived; the other tests bypass it by
    calling the private helpers directly."""

    def _factory(self):
        factory = BedrockLanguageModelFactory.__new__(BedrockLanguageModelFactory)
        factory.region_name = "us-west-2"
        factory._client = None

        class _Session:
            profile_name = "default"

        factory.boto_session = _Session()
        return factory

    def _capture(self, monkeypatch):
        """Stub both model classes; return a dict capturing which was built and
        with what kwargs."""
        captured: dict = {}

        def make(name):
            def _ctor(**kwargs):
                captured["class"] = name
                captured["kwargs"] = kwargs
                return object()

            return _ctor

        monkeypatch.setattr(utils_mod, "ChatBedrock", make("ChatBedrock"))
        monkeypatch.setattr(
            utils_mod, "ChatBedrockConverse", make("ChatBedrockConverse")
        )
        return captured

    def test_sonnet5_thinking_local_uses_converse_and_adaptive_no_temp(
        self, monkeypatch
    ):
        factory = self._factory()
        # Resolve to the bare (non-cross-region) id so is_cross_region=False but
        # use_converse=True (Sonnet 5 supports thinking).
        monkeypatch.setattr(
            BedrockCrossRegionModelHelper,
            "get_cross_region_model_id",
            classmethod(lambda cls, s, m, r: m.value),
        )
        captured = self._capture(monkeypatch)
        factory.get_model(
            LanguageModelId.CLAUDE_V5_SONNET,
            enable_thinking=True,
            temperature=0.0,
            max_tokens=4096,
        )
        assert captured["class"] == "ChatBedrockConverse"
        kwargs = captured["kwargs"]
        assert "temperature" not in kwargs
        amrf = kwargs.get("additional_model_request_fields", {})
        assert amrf.get("thinking") == {"type": "adaptive"}

    def test_transitional_cross_region_keeps_temperature(self, monkeypatch):
        factory = self._factory()
        monkeypatch.setattr(
            BedrockCrossRegionModelHelper,
            "get_cross_region_model_id",
            classmethod(lambda cls, s, m, r: f"us.{m.value}"),
        )
        captured = self._capture(monkeypatch)
        factory.get_model(
            LanguageModelId.CLAUDE_V4_6_SONNET,
            temperature=0.0,
            max_tokens=4096,
        )
        assert captured["class"] == "ChatBedrockConverse"  # cross-region
        assert captured["kwargs"]["temperature"] == 0.0

    def test_local_non_thinking_uses_chatbedrock(self, monkeypatch):
        factory = self._factory()
        monkeypatch.setattr(
            BedrockCrossRegionModelHelper,
            "get_cross_region_model_id",
            classmethod(lambda cls, s, m, r: m.value),
        )
        captured = self._capture(monkeypatch)
        factory.get_model(
            LanguageModelId.CLAUDE_V3_5_SONNET,
            temperature=0.0,
            max_tokens=4096,
        )
        assert captured["class"] == "ChatBedrock"
        # Sampling params live under model_kwargs on the non-converse path.
        assert captured["kwargs"]["model_kwargs"]["temperature"] == 0.0


class TestCrossRegionIdConstruction:
    def test_global_prefix(self):
        out = BedrockCrossRegionModelHelper._build_cross_region_model_id(
            LanguageModelId.CLAUDE_V4_6_SONNET, "us-west-2", is_global=True
        )
        assert out == "global.anthropic.claude-sonnet-4-6"

    def test_us_regional_prefix(self):
        out = BedrockCrossRegionModelHelper._build_cross_region_model_id(
            LanguageModelId.CLAUDE_V4_6_SONNET, "us-west-2", is_global=False
        )
        assert out == "us.anthropic.claude-sonnet-4-6"

    def test_apac_regional_prefix(self):
        out = BedrockCrossRegionModelHelper._build_cross_region_model_id(
            LanguageModelId.CLAUDE_V4_6_SONNET, "ap-northeast-2", is_global=False
        )
        assert out == "apac.anthropic.claude-sonnet-4-6"


class _FakeBedrock:
    def __init__(self, profile_ids: list[str]):
        self._profile_ids = profile_ids
        self.calls = 0

    def list_inference_profiles(self, **kwargs):
        self.calls += 1
        return {
            "inferenceProfileSummaries": [
                {"inferenceProfileId": pid} for pid in self._profile_ids
            ]
        }


class _FakeSession:
    def __init__(self, bedrock):
        self._bedrock = bedrock

    def client(self, service, region_name=None):
        assert service == "bedrock"
        return self._bedrock


class TestCrossRegionResolution:
    """The availability-driven selection + memoization + error fallback in
    get_cross_region_model_id (previously untested — only the pure id builder
    was covered)."""

    def setup_method(self):
        # The profile-set cache is class-level; clear it between tests.
        BedrockCrossRegionModelHelper._profile_set_cache.clear()

    def teardown_method(self):
        BedrockCrossRegionModelHelper._profile_set_cache.clear()

    def test_regional_profile_chosen_when_available(self):
        bedrock = _FakeBedrock(["apac.anthropic.claude-sonnet-5"])
        session = _FakeSession(bedrock)
        out = BedrockCrossRegionModelHelper.get_cross_region_model_id(
            session, LanguageModelId.CLAUDE_V5_SONNET, "ap-northeast-2"
        )
        assert out == "apac.anthropic.claude-sonnet-5"

    def test_global_profile_preferred(self):
        bedrock = _FakeBedrock(
            ["global.anthropic.claude-sonnet-5", "apac.anthropic.claude-sonnet-5"]
        )
        session = _FakeSession(bedrock)
        out = BedrockCrossRegionModelHelper.get_cross_region_model_id(
            session, LanguageModelId.CLAUDE_V5_SONNET, "ap-northeast-2"
        )
        assert out == "global.anthropic.claude-sonnet-5"

    def test_falls_back_to_bare_id_when_no_profile(self):
        bedrock = _FakeBedrock(["us.some.other.model"])
        session = _FakeSession(bedrock)
        out = BedrockCrossRegionModelHelper.get_cross_region_model_id(
            session, LanguageModelId.CLAUDE_V5_SONNET, "us-west-2"
        )
        assert out == "anthropic.claude-sonnet-5"

    def test_falls_back_and_does_not_raise_on_list_error(self):
        class _BoomBedrock:
            def list_inference_profiles(self, **kwargs):
                raise RuntimeError("AccessDenied: ListInferenceProfiles")

        session = _FakeSession(_BoomBedrock())
        out = BedrockCrossRegionModelHelper.get_cross_region_model_id(
            session, LanguageModelId.CLAUDE_V5_SONNET, "us-west-2"
        )
        assert out == "anthropic.claude-sonnet-5"

    def test_profile_list_memoized_per_region(self):
        bedrock = _FakeBedrock(["apac.anthropic.claude-sonnet-5"])
        session = _FakeSession(bedrock)
        for _ in range(4):
            BedrockCrossRegionModelHelper.get_cross_region_model_id(
                session, LanguageModelId.CLAUDE_V5_SONNET, "ap-northeast-2"
            )
        # Resolved 4 times but the profile catalog is listed only ONCE.
        assert bedrock.calls == 1

    def test_transient_error_not_cached(self):
        # A failed listing must not be pinned: a later successful call resolves.
        class _FlakyBedrock:
            def __init__(self):
                self.calls = 0

            def list_inference_profiles(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("throttled")
                return {
                    "inferenceProfileSummaries": [
                        {"inferenceProfileId": "us.anthropic.claude-sonnet-5"}
                    ]
                }

        bedrock = _FlakyBedrock()
        session = _FakeSession(bedrock)
        first = BedrockCrossRegionModelHelper.get_cross_region_model_id(
            session, LanguageModelId.CLAUDE_V5_SONNET, "us-west-2"
        )
        assert first == "anthropic.claude-sonnet-5"  # fell back
        second = BedrockCrossRegionModelHelper.get_cross_region_model_id(
            session, LanguageModelId.CLAUDE_V5_SONNET, "us-west-2"
        )
        assert second == "us.anthropic.claude-sonnet-5"  # retried and resolved

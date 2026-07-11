"""Unit tests for the Bedrock model-info registry and cross-region helpers.

These avoid real boto3/Bedrock calls by testing the pure logic: the model-info
table, max-token validation, and cross-region id construction.
"""

from __future__ import annotations

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

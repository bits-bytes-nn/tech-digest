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

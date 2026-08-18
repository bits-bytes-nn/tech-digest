"""Amazon Bedrock language-model construction.

Holds the per-model capability registry and the factory that turns a
``LanguageModelId`` into a configured LangChain chat model, including the
backend-specific differences (ChatBedrock vs ChatBedrockConverse), cross-region
inference-profile resolution, and the thinking/sampling parameter rules that
changed with Sonnet 5.

Split out of ``utils`` so that module is once again a small set of unrelated
helpers rather than the project's largest file.
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, TypeVar

import boto3
from botocore.config import Config as BotoConfig
from langchain_aws import ChatBedrock, ChatBedrockConverse
from pydantic import BaseModel

from .constants import LanguageModelId
from .logger import logger


class LanguageModelInfo(BaseModel):
    context_window_size: int
    max_output_tokens: int
    supports_performance_optimization: bool = False
    supports_prompt_caching: bool = False
    supports_thinking: bool = False
    supports_1m_context_window: bool = False
    # Newer models (Sonnet 5 and up) removed the sampling parameters
    # (temperature/top_k/top_p) and reject requests that include them with a
    # ValidationException. Default True for backward compatibility; set False
    # for models that no longer accept sampling params.
    supports_sampling_params: bool = True
    # Newer models (Sonnet 5 and up) also replaced the explicit thinking budget
    # (thinking.type="enabled" + budget_tokens) with adaptive thinking
    # (thinking.type="adaptive"); the old form is rejected. Depth is instead
    # controlled by output_config.effort. Set True for those models.
    uses_adaptive_thinking: bool = False


_LANGUAGE_MODEL_INFO: dict[LanguageModelId, LanguageModelInfo] = {
    LanguageModelId.CLAUDE_V3_HAIKU: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=4096,
        supports_prompt_caching=True,
    ),
    LanguageModelId.CLAUDE_V3_5_HAIKU: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=8192,
        supports_performance_optimization=True,
        supports_prompt_caching=True,
    ),
    LanguageModelId.CLAUDE_V4_5_HAIKU: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=64000,
        supports_prompt_caching=True,
        supports_thinking=True,
    ),
    LanguageModelId.CLAUDE_V3_5_SONNET: LanguageModelInfo(
        context_window_size=200000, max_output_tokens=8192
    ),
    LanguageModelId.CLAUDE_V3_5_SONNET_V2: LanguageModelInfo(
        context_window_size=200000, max_output_tokens=8192, supports_prompt_caching=True
    ),
    LanguageModelId.CLAUDE_V3_7_SONNET: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=64000,
        supports_prompt_caching=True,
        supports_thinking=True,
    ),
    LanguageModelId.CLAUDE_V4_SONNET: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=64000,
        supports_prompt_caching=True,
        supports_thinking=True,
        supports_1m_context_window=True,
    ),
    LanguageModelId.CLAUDE_V4_5_SONNET: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=64000,
        supports_prompt_caching=True,
        supports_thinking=True,
        supports_1m_context_window=True,
    ),
    LanguageModelId.CLAUDE_V4_6_SONNET: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=64000,
        supports_prompt_caching=True,
        supports_thinking=True,
        supports_1m_context_window=True,
    ),
    LanguageModelId.CLAUDE_V5_SONNET: LanguageModelInfo(
        context_window_size=1000000,
        max_output_tokens=64000,
        supports_prompt_caching=True,
        supports_thinking=True,
        supports_1m_context_window=True,
        supports_sampling_params=False,
        uses_adaptive_thinking=True,
    ),
    LanguageModelId.CLAUDE_V4_OPUS: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=64000,
        supports_prompt_caching=True,
        supports_thinking=True,
        supports_1m_context_window=True,
    ),
    LanguageModelId.CLAUDE_V4_1_OPUS: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=64000,
        supports_prompt_caching=True,
        supports_thinking=True,
        supports_1m_context_window=True,
    ),
    LanguageModelId.CLAUDE_V4_5_OPUS: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=64000,
        supports_prompt_caching=True,
        supports_thinking=True,
        supports_1m_context_window=True,
    ),
    LanguageModelId.CLAUDE_V4_6_OPUS: LanguageModelInfo(
        context_window_size=1000000,
        max_output_tokens=64000,
        supports_prompt_caching=True,
        supports_thinking=True,
        supports_1m_context_window=True,
    ),
    LanguageModelId.CLAUDE_V4_7_OPUS: LanguageModelInfo(
        context_window_size=1000000,
        max_output_tokens=64000,
        supports_prompt_caching=True,
        supports_thinking=True,
        supports_1m_context_window=True,
        supports_sampling_params=False,
        uses_adaptive_thinking=True,
    ),
    LanguageModelId.CLAUDE_V4_8_OPUS: LanguageModelInfo(
        context_window_size=1000000,
        max_output_tokens=64000,
        supports_prompt_caching=True,
        supports_thinking=True,
        supports_1m_context_window=True,
        supports_sampling_params=False,
        uses_adaptive_thinking=True,
    ),
    # Claude 5 generation: same request contract as Sonnet 5 — no sampling
    # parameters, adaptive thinking only, depth steered by output_config.effort
    # (Opus tier additionally accepts "xhigh" and "max").
    LanguageModelId.CLAUDE_V5_OPUS: LanguageModelInfo(
        context_window_size=1000000,
        max_output_tokens=64000,
        supports_prompt_caching=True,
        supports_thinking=True,
        supports_1m_context_window=True,
        supports_sampling_params=False,
        uses_adaptive_thinking=True,
    ),
    # NOTE: add new models here
}


ModelIdT = TypeVar("ModelIdT")
ModelInfoT = TypeVar("ModelInfoT")
WrapperT = TypeVar("WrapperT")


class BaseBedrockModelFactory(Generic[ModelIdT, ModelInfoT, WrapperT], ABC):
    BOTO_READ_TIMEOUT: ClassVar[int] = 300
    BOTO_MAX_ATTEMPTS: ClassVar[int] = 3
    MAX_POOL_CONNECTIONS: ClassVar[int] = 50

    def __init__(
        self,
        boto_session: boto3.Session | None = None,
        region_name: str | None = None,
        profile_name: str | None = None,
    ) -> None:
        self.boto_session = boto_session or boto3.Session(profile_name=profile_name)
        self.region_name = region_name or self.boto_session.region_name
        boto_config = BotoConfig(
            read_timeout=self.BOTO_READ_TIMEOUT,
            connect_timeout=60,
            retries={"max_attempts": self.BOTO_MAX_ATTEMPTS, "mode": "adaptive"},
            max_pool_connections=self.MAX_POOL_CONNECTIONS,
        )
        self._client = self.boto_session.client(
            self._get_boto_service_name(),
            region_name=self.region_name,
            config=boto_config,
        )
        logger.debug(
            "Initialized %s for region: '%s'", self.__class__.__name__, self.region_name
        )

    @abstractmethod
    def _get_boto_service_name(self) -> str: ...

    @abstractmethod
    def _get_model_info_dict(self) -> dict[ModelIdT, ModelInfoT]: ...

    @abstractmethod
    def get_model(self, model_id: ModelIdT, **kwargs: Any) -> WrapperT: ...

    def get_model_info(self, model_id: ModelIdT) -> ModelInfoT | None:
        return self._get_model_info_dict().get(model_id)


class BedrockCrossRegionModelHelper:
    # The system-defined inference-profile catalog is static for a run, and a
    # single newsletter run resolves several models (filter, summarizer, fixing,
    # greeter). Cache the available-profile set per region so we issue ONE
    # list_inference_profiles call instead of ~2 per model — cutting redundant
    # large API round-trips and throttling exposure on the critical path. Only
    # SUCCESSFUL listings are cached (a transient failure must not be pinned).
    _profile_set_cache: ClassVar[dict[str, frozenset[str]]] = {}

    @classmethod
    def get_cross_region_model_id(
        cls,
        boto_session: boto3.Session,
        model_id: LanguageModelId,
        region_name: str,
    ) -> str:
        try:
            profiles = cls._get_available_profiles(boto_session, region_name)
        except Exception as e:
            # Listing failed (throttle, missing bedrock:ListInferenceProfiles).
            # Fall back to the bare id so on-demand-capable models still work;
            # profile-only models will surface a clear error at invoke time.
            logger.warning(
                "Could not list inference profiles for '%s' in %s (%s); "
                "falling back to the bare model id. If this is a profile-only "
                "model (e.g. Sonnet 5), the invoke will fail — check "
                "bedrock:ListInferenceProfiles permission / throttling.",
                model_id.value,
                region_name,
                e,
            )
            return model_id.value

        for is_global in (True, False):
            candidate = cls._build_cross_region_model_id(
                model_id, region_name, is_global=is_global
            )
            if candidate in profiles:
                logger.debug("Using cross-region model: '%s'", candidate)
                return candidate
        logger.debug(
            "Cross-region profiles not available, using standard model: '%s'",
            model_id.value,
        )
        return model_id.value

    @classmethod
    def _get_available_profiles(
        cls, boto_session: boto3.Session, region_name: str
    ) -> frozenset[str]:
        cached = cls._profile_set_cache.get(region_name)
        if cached is not None:
            return cached
        bedrock_client = boto_session.client("bedrock", region_name=region_name)
        response = bedrock_client.list_inference_profiles(
            maxResults=1000, typeEquals="SYSTEM_DEFINED"
        )
        profiles = frozenset(
            profile["inferenceProfileId"]
            for profile in response.get("inferenceProfileSummaries", [])
        )
        cls._profile_set_cache[region_name] = profiles
        return profiles

    @staticmethod
    def _build_cross_region_model_id(
        model_id: LanguageModelId, region_name: str, is_global: bool = False
    ) -> str:
        if is_global:
            return f"global.{model_id.value}"
        prefix = "apac" if region_name.startswith("ap-") else region_name[:2]
        return f"{prefix}.{model_id.value}"


class BedrockLanguageModelFactory(
    BaseBedrockModelFactory[
        LanguageModelId, LanguageModelInfo, ChatBedrock | ChatBedrockConverse
    ]
):
    DEFAULT_TEMPERATURE: ClassVar[float] = 0.0
    DEFAULT_TOP_K: ClassVar[int] = 50
    # Anthropic requires a thinking budget of at least 1024 tokens, and the
    # budget must be strictly less than max_tokens. The default is a usable
    # floor; callers (e.g. Summarizer) pass larger, task-tuned budgets.
    MIN_THINKING_BUDGET: ClassVar[int] = 1024
    DEFAULT_THINKING_BUDGET_TOKENS: ClassVar[int] = 4096
    DEFAULT_LATENCY_MODE: ClassVar[str] = "normal"

    def _get_boto_service_name(self) -> str:
        return "bedrock-runtime"

    def _get_model_info_dict(self) -> dict[LanguageModelId, LanguageModelInfo]:
        return _LANGUAGE_MODEL_INFO

    def get_model(
        self, model_id: LanguageModelId, **kwargs: Any
    ) -> ChatBedrock | ChatBedrockConverse:
        model_info = self.get_model_info(model_id)
        if not model_info:
            raise ValueError(f"Unsupported language model ID: '{model_id.value}'")
        resolved_model_id = BedrockCrossRegionModelHelper.get_cross_region_model_id(
            self.boto_session, model_id, self.region_name or ""
        )
        # Which backend runs decides where every parameter goes: ChatBedrockConverse
        # takes them as top-level kwargs / additional_model_request_fields, while
        # ChatBedrock nests them under model_kwargs. Converse is required for a
        # cross-region inference profile AND for thinking, so the two conditions
        # are OR'd into ONE flag that is threaded down. The helpers below take it
        # as ``use_converse``; they previously named the same argument
        # ``is_cross_region``, which read as if thinking-on-a-plain-model took the
        # ChatBedrock path when it does not.
        is_cross_region = resolved_model_id != model_id.value
        enable_thinking = kwargs.get("enable_thinking", False)
        use_converse = is_cross_region or (
            enable_thinking and model_info.supports_thinking
        )
        model_config = self._build_model_config(
            model_info, resolved_model_id, use_converse, **kwargs
        )
        model_class = ChatBedrockConverse if use_converse else ChatBedrock
        model = model_class(**model_config)
        logger.debug(
            "Created language model: '%s' with class %s",
            resolved_model_id,
            model_class.__name__,
        )
        return model

    def _build_model_config(
        self,
        model_info: LanguageModelInfo,
        resolved_model_id: str,
        use_converse: bool,
        **kwargs: Any,
    ) -> dict[str, Any]:
        enable_thinking = kwargs.get("enable_thinking", False)
        supports_1m_context_window = kwargs.get("supports_1m_context_window", False)
        temperature = kwargs.get("temperature", self.DEFAULT_TEMPERATURE)
        final_temperature = (
            1.0
            if self._should_enable_thinking(enable_thinking, model_info)
            else temperature
        )
        if final_temperature != temperature:
            logger.debug("Adjusting temperature to 1.0 for thinking mode")
        final_max_tokens = self._validate_max_tokens(
            kwargs.get("max_tokens"), model_info
        )
        config = self._build_base_config(
            resolved_model_id, use_converse, model_info, **kwargs
        )
        # Newer models (Sonnet 5+) removed sampling params and reject requests
        # that include `temperature`; only send it where the model accepts it.
        sampling_params: dict[str, Any] = {"max_tokens": final_max_tokens}
        if model_info.supports_sampling_params:
            sampling_params["temperature"] = final_temperature
        if use_converse:
            config.update(sampling_params)
        else:
            config["model_kwargs"].update(sampling_params)
        if supports_1m_context_window and model_info.supports_1m_context_window:
            if use_converse:
                config.setdefault("additional_model_request_fields", {}).update(
                    {"anthropic_beta": ["context-1m-2025-08-07"]}
                )
            else:
                config["model_kwargs"].setdefault(
                    "additionalModelRequestFields", {}
                ).update({"anthropic_beta": ["context-1m-2025-08-07"]})
            logger.debug("Applied 1M context window support")
        self._apply_model_features(config, model_info, use_converse, **kwargs)
        return config

    def _build_base_config(
        self,
        resolved_model_id: str,
        use_converse: bool,
        model_info: LanguageModelInfo,
        **kwargs: Any,
    ) -> dict[str, Any]:
        config = {
            "model_id": resolved_model_id,
            "region_name": self.region_name,
            "client": self._client,
            "callbacks": kwargs.get("callbacks"),
        }
        if (
            self.boto_session.profile_name
            and self.boto_session.profile_name != "default"
        ):
            config["credentials_profile_name"] = self.boto_session.profile_name
        common_params = {
            "stop_sequences": ["\n\nHuman:"],
        }
        if use_converse:
            config.update(common_params)
        else:
            model_kwargs: dict[str, Any] = {**common_params}
            # top_k is a sampling param; newer models (Sonnet 5+) reject it.
            if model_info.supports_sampling_params:
                model_kwargs["top_k"] = kwargs.get("top_k", self.DEFAULT_TOP_K)
            config["model_kwargs"] = model_kwargs
        return config

    def _apply_model_features(
        self,
        config: dict[str, Any],
        model_info: LanguageModelInfo,
        use_converse: bool,
        **kwargs: Any,
    ) -> None:
        enable_perf = kwargs.get("enable_performance_optimization", False)
        enable_think = kwargs.get("enable_thinking", False)
        if self._should_enable_performance_optimization(
            enable_perf, model_info, use_converse
        ):
            latency = kwargs.get("latency_mode", self.DEFAULT_LATENCY_MODE)
            config.setdefault("performanceConfig", {}).update({"latency": latency})
            logger.debug(
                "Applied performance optimization (latency_mode='%s')", latency
            )
        if self._should_enable_thinking(enable_think, model_info):
            if model_info.uses_adaptive_thinking:
                self._apply_adaptive_thinking(config, use_converse, **kwargs)
            else:
                self._apply_budget_thinking(config, model_info, use_converse, **kwargs)

    def _apply_adaptive_thinking(
        self, config: dict[str, Any], use_converse: bool, **kwargs: Any
    ) -> None:
        # Newer models (Sonnet 5+) require adaptive thinking; the model
        # self-moderates how much it thinks. Depth is optionally steered via
        # output_config.effort ("low"|"medium"|"high"|"max") rather than a token
        # budget. Omitting effort uses the model default.
        think_config: dict[str, Any] = {"thinking": {"type": "adaptive"}}
        effort = kwargs.get("effort")
        if effort:
            think_config["output_config"] = {"effort": effort}
        # Surface a dead knob: adaptive-thinking models ignore thinking_budget_tokens
        # (they use effort instead). Warn only when a NON-default budget was set,
        # so the operator knows their tuning is a no-op — but stay quiet on the
        # default the summarizer always passes.
        budget = kwargs.get("thinking_budget_tokens")
        if budget is not None and budget != self.DEFAULT_THINKING_BUDGET_TOKENS:
            logger.warning(
                "thinking_budget_tokens=%s is ignored for adaptive-thinking "
                "models; use 'effort' (low/medium/high/max) to steer depth.",
                budget,
            )
        if use_converse:
            config.setdefault("additional_model_request_fields", {}).update(
                think_config
            )
        else:
            config.setdefault("model_kwargs", {}).update(think_config)
        logger.debug("Applied adaptive thinking (effort='%s')", effort or "default")

    def _apply_budget_thinking(
        self,
        config: dict[str, Any],
        model_info: LanguageModelInfo,
        use_converse: bool,
        **kwargs: Any,
    ) -> None:
        budget = kwargs.get(
            "thinking_budget_tokens", self.DEFAULT_THINKING_BUDGET_TOKENS
        )
        # Bedrock requires MIN_THINKING_BUDGET <= budget < max_tokens (the
        # budget is drawn from the output allowance).
        effective_max = self._validate_max_tokens(kwargs.get("max_tokens"), model_info)
        if effective_max <= self.MIN_THINKING_BUDGET:
            # No budget can satisfy MIN_THINKING_BUDGET <= budget < max_tokens;
            # thinking is impossible at this max_tokens. Skip it rather than
            # send a request Bedrock will reject.
            logger.warning(
                "max_tokens (%d) too small for thinking (min budget %d); "
                "disabling thinking for this request.",
                effective_max,
                self.MIN_THINKING_BUDGET,
            )
            return
        if budget >= effective_max:
            clamped = max(self.MIN_THINKING_BUDGET, effective_max - 1024)
            logger.warning(
                "thinking_budget_tokens (%d) >= max_tokens (%d); clamping to %d",
                budget,
                effective_max,
                clamped,
            )
            budget = clamped
        budget = max(budget, self.MIN_THINKING_BUDGET)
        think_config = {"thinking": {"type": "enabled", "budget_tokens": budget}}
        if use_converse:
            config.setdefault("additional_model_request_fields", {}).update(
                think_config
            )
        else:
            config.setdefault("model_kwargs", {}).update(think_config)
        logger.debug("Applied thinking mode (budget_tokens=%d)", budget)

    @staticmethod
    def _validate_max_tokens(
        max_tokens: int | None, model_info: LanguageModelInfo
    ) -> int:
        final_max_tokens = max_tokens or model_info.max_output_tokens
        if final_max_tokens > model_info.max_output_tokens:
            logger.warning(
                "Requested max_tokens (%d) exceeds model's maximum (%d). Adjusting.",
                final_max_tokens,
                model_info.max_output_tokens,
            )
            return model_info.max_output_tokens
        return final_max_tokens

    @staticmethod
    def _should_enable_performance_optimization(
        enable: bool, model_info: LanguageModelInfo, use_converse: bool
    ) -> bool:
        return (
            enable and model_info.supports_performance_optimization and not use_converse
        )

    @staticmethod
    def _should_enable_thinking(enable: bool, model_info: LanguageModelInfo) -> bool:
        return enable and model_info.supports_thinking

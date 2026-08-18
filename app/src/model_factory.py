"""Amazon Bedrock language-model construction.

Holds the per-model capability registry and the factory that turns a
``LanguageModelId`` into a configured LangChain chat model, including the
backend-specific differences (ChatBedrock vs ChatBedrockConverse), cross-region
inference-profile resolution, and the thinking/sampling parameter rules that
changed with Sonnet 5.

Split out of ``utils`` so that module is once again a small set of unrelated
helpers rather than the project's largest file.
"""

import os
import re
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, TypeVar

import boto3
from botocore.config import Config as BotoConfig
from langchain_aws import ChatBedrock, ChatBedrockConverse
from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel

from .constants import EnvVars, LanguageModelId
from .logger import logger


class TokenUsageLogger(BaseCallbackHandler):
    """Log every LLM call's token usage, tagged with the pipeline stage.

    Cost Explorer bills per MODEL, not per stage. Filtering, summarization and the
    output-fixing repair all run on the same model, so a bill of "Sonnet 5: 4.7M
    input tokens" cannot be attributed to a stage — and filtering (one call per
    collected post, each carrying a full article) is a very different shape from
    summarization (a handful of calls). Any optimisation without this is a guess.

    Best-effort by construction: a callback must never be able to fail a
    generation, so every read is defensive.
    """

    def __init__(self, stage: str, model_id: str) -> None:
        self.stage = stage
        self.model_id = model_id

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        try:
            usage: dict[str, Any] = {}
            for generations in getattr(response, "generations", []) or []:
                for generation in generations or []:
                    message = getattr(generation, "message", None)
                    meta = getattr(message, "usage_metadata", None)
                    if meta:
                        usage = dict(meta)
            if not usage:
                usage = (getattr(response, "llm_output", None) or {}).get(
                    "usage", {}
                ) or {}
            if not usage:
                return
            details = usage.get("input_token_details") or {}
            logger.info(
                "LLM usage stage=%s model=%s input=%s output=%s "
                "cache_read=%s cache_write=%s",
                self.stage,
                self.model_id,
                usage.get("input_tokens"),
                usage.get("output_tokens"),
                details.get("cache_read"),
                details.get("cache_creation"),
            )
        except Exception:  # pragma: no cover - telemetry must never break a call
            logger.debug("Could not read LLM usage metadata", exc_info=True)


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
    # (region, profile_name) -> ARN, or "" for "looked up, does not exist". The
    # negative entry matters: without it every model build re-lists APPLICATION
    # profiles just to rediscover that none are provisioned.
    _application_profile_cache: ClassVar[dict[tuple[str, str], str]] = {}

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

        resolved = model_id.value
        for is_global in (True, False):
            candidate = cls._build_cross_region_model_id(
                model_id, region_name, is_global=is_global
            )
            if candidate in profiles:
                logger.debug("Using cross-region model: '%s'", candidate)
                resolved = candidate
                break
        else:
            logger.debug(
                "Cross-region profiles not available, using standard model: '%s'",
                model_id.value,
            )
        # Prefer this project's APPLICATION inference profile when one exists. It
        # is the only way on-demand Bedrock token spend carries a cost-allocation
        # tag — InvokeModel has no taggable resource behind it otherwise — and this
        # account bills several workloads against the same Claude models.
        return cls._application_profile_arn(
            boto_session, model_id, region_name, resolved
        )

    @staticmethod
    def application_profile_name(model_id: LanguageModelId) -> str:
        """Deterministic name for this project/stage's application inference
        profile for a model. Shared by the resolver and
        ``scripts/put_inference_profiles.py`` so the two cannot drift."""
        project = os.getenv(EnvVars.PROJECT_NAME.value, "tech-digest")
        stage = os.getenv(
            EnvVars.STAGE.value,
            os.getenv(EnvVars.CONFIG_FILE_SUFFIX.value, "dev"),
        )
        slug = re.sub(r"[^A-Za-z0-9]+", "-", model_id.value).strip("-")
        return f"{project}-{stage}-{slug}"

    @classmethod
    def _application_profile_arn(
        cls,
        boto_session: boto3.Session,
        model_id: LanguageModelId,
        region_name: str,
        fallback: str,
    ) -> str:
        """ARN of this project's application inference profile, or ``fallback``.

        Best-effort by design: cost attribution must never be able to stop a
        generation, so a missing profile, a denied ListInferenceProfiles or any
        unexpected error keeps the id the caller would have used anyway.
        """
        wanted = cls.application_profile_name(model_id)
        cached = cls._application_profile_cache.get((region_name, wanted))
        if cached is not None:
            return cached or fallback
        try:
            client = boto_session.client("bedrock", region_name=region_name)
            paginator = client.get_paginator("list_inference_profiles")
            for page in paginator.paginate(typeEquals="APPLICATION"):
                for summary in page.get("inferenceProfileSummaries", []):
                    if summary.get("inferenceProfileName") == wanted:
                        arn = summary["inferenceProfileArn"]
                        cls._application_profile_cache[(region_name, wanted)] = arn
                        logger.debug(
                            "Using application inference profile '%s' (%s)",
                            wanted,
                            arn,
                        )
                        return arn
        except Exception as e:
            logger.debug(
                "Could not look up application inference profile '%s': %s", wanted, e
            )
            return fallback
        # Cache the negative result too: without it every model build re-lists.
        cls._application_profile_cache[(region_name, wanted)] = ""
        logger.debug(
            "No application inference profile named '%s'; using '%s'. Bedrock spend "
            "will not be tagged — run scripts/put_inference_profiles.py to enable "
            "cost attribution.",
            wanted,
            fallback,
        )
        return fallback

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
            model_info, resolved_model_id, use_converse, model_id=model_id, **kwargs
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
        # A `stage` names the pipeline step making the call so its token usage is
        # separable in the logs; Cost Explorer only sees the model.
        callbacks = list(kwargs.get("callbacks") or [])
        if stage := kwargs.get("stage"):
            callbacks.append(TokenUsageLogger(str(stage), resolved_model_id))
        config = {
            "model_id": resolved_model_id,
            "region_name": self.region_name,
            "client": self._client,
            "callbacks": callbacks or None,
        }
        # An application inference profile is identified by ARN, and an ARN does not
        # encode the model provider, so langchain_aws cannot infer it and refuses to
        # construct ("Model provider should be supplied when passing a model ARN").
        # Derive it from the catalog id ("anthropic.claude-...") rather than
        # hardcoding, so adding a non-Anthropic model stays correct.
        if resolved_model_id.startswith("arn:"):
            config["provider"] = kwargs.get("provider") or self._provider_of(
                kwargs.get("model_id")
            )
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
    def _provider_of(model_id: LanguageModelId | None) -> str:
        """Bedrock provider prefix of a catalog model id ("anthropic.claude-..")."""
        if model_id is None:
            return "anthropic"
        return model_id.value.split(".", 1)[0]

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

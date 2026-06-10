import asyncio
import functools
import math
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Generic, TypeVar

import boto3
import tenacity
from botocore.config import Config as BotoConfig
from bs4 import BeautifulSoup
from langchain_aws import ChatBedrock, ChatBedrockConverse
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from tqdm import tqdm

from .constants import LanguageModelId
from .logger import logger

MAX_RETRIES: int = 5
RETRY_MAX_WAIT: int = 120
RETRY_MULTIPLIER: int = 30

EMAIL_PATTERN: str = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"


class LanguageModelInfo(BaseModel):
    context_window_size: int
    max_output_tokens: int
    supports_performance_optimization: bool = False
    supports_prompt_caching: bool = False
    supports_thinking: bool = False
    supports_1m_context_window: bool = False


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
    ),
    LanguageModelId.CLAUDE_V4_8_OPUS: LanguageModelInfo(
        context_window_size=1000000,
        max_output_tokens=64000,
        supports_prompt_caching=True,
        supports_thinking=True,
        supports_1m_context_window=True,
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
    @staticmethod
    def get_cross_region_model_id(
        boto_session: boto3.Session,
        model_id: LanguageModelId,
        region_name: str,
    ) -> str:
        try:
            bedrock_client = boto_session.client("bedrock", region_name=region_name)
            global_model_id = (
                BedrockCrossRegionModelHelper._build_cross_region_model_id(
                    model_id, region_name, is_global=True
                )
            )
            if BedrockCrossRegionModelHelper._is_cross_region_model_available(
                bedrock_client, global_model_id
            ):
                logger.debug("Using global cross-region model: '%s'", global_model_id)
                return global_model_id
            regional_model_id = (
                BedrockCrossRegionModelHelper._build_cross_region_model_id(
                    model_id, region_name, is_global=False
                )
            )
            if BedrockCrossRegionModelHelper._is_cross_region_model_available(
                bedrock_client, regional_model_id
            ):
                logger.debug(
                    "Using regional cross-region model: '%s'", regional_model_id
                )
                return regional_model_id
            logger.debug(
                "Cross-region models not available, using standard model: '%s'",
                model_id.value,
            )
            return model_id.value
        except Exception as e:
            logger.warning(
                "Failed to resolve cross-region model for '%s': %s. Falling back to standard model.",
                model_id.value,
                e,
            )
            return model_id.value

    @staticmethod
    def _build_cross_region_model_id(
        model_id: LanguageModelId, region_name: str, is_global: bool = False
    ) -> str:
        if is_global:
            return f"global.{model_id.value}"
        prefix = "apac" if region_name.startswith("ap-") else region_name[:2]
        return f"{prefix}.{model_id.value}"

    @staticmethod
    def _is_cross_region_model_available(
        bedrock_client: Any, cross_region_id: str
    ) -> bool:
        try:
            response = bedrock_client.list_inference_profiles(
                maxResults=1000, typeEquals="SYSTEM_DEFINED"
            )
            available_profiles = {
                profile["inferenceProfileId"]
                for profile in response.get("inferenceProfileSummaries", [])
            }
            return cross_region_id in available_profiles
        except Exception as e:
            raise RuntimeError(
                f"Failed to check cross-region model availability: {e}"
            ) from e


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
        is_cross_region: bool,
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
        config = self._build_base_config(resolved_model_id, is_cross_region, **kwargs)
        if is_cross_region:
            config.update(
                {"max_tokens": final_max_tokens, "temperature": final_temperature}
            )
        else:
            config["model_kwargs"].update(
                {"max_tokens": final_max_tokens, "temperature": final_temperature}
            )
        if supports_1m_context_window and model_info.supports_1m_context_window:
            if is_cross_region:
                config.setdefault("additional_model_request_fields", {}).update(
                    {"anthropic_beta": ["context-1m-2025-08-07"]}
                )
            else:
                config["model_kwargs"].setdefault(
                    "additionalModelRequestFields", {}
                ).update({"anthropic_beta": ["context-1m-2025-08-07"]})
            logger.debug("Applied 1M context window support")
        self._apply_model_features(config, model_info, is_cross_region, **kwargs)
        return config

    def _build_base_config(
        self, resolved_model_id: str, is_cross_region: bool, **kwargs: Any
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
        if is_cross_region:
            config.update(common_params)
        else:
            config["model_kwargs"] = {
                "top_k": kwargs.get("top_k", self.DEFAULT_TOP_K),
                **common_params,
            }
        return config

    def _apply_model_features(
        self,
        config: dict[str, Any],
        model_info: LanguageModelInfo,
        is_cross_region: bool,
        **kwargs: Any,
    ) -> None:
        enable_perf = kwargs.get("enable_performance_optimization", False)
        enable_think = kwargs.get("enable_thinking", False)
        if self._should_enable_performance_optimization(
            enable_perf, model_info, is_cross_region
        ):
            latency = kwargs.get("latency_mode", self.DEFAULT_LATENCY_MODE)
            config.setdefault("performanceConfig", {}).update({"latency": latency})
            logger.debug(
                "Applied performance optimization (latency_mode='%s')", latency
            )
        if self._should_enable_thinking(enable_think, model_info):
            budget = kwargs.get(
                "thinking_budget_tokens", self.DEFAULT_THINKING_BUDGET_TOKENS
            )
            # Bedrock requires MIN_THINKING_BUDGET <= budget < max_tokens (the
            # budget is drawn from the output allowance).
            effective_max = self._validate_max_tokens(
                kwargs.get("max_tokens"), model_info
            )
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
            if is_cross_region:
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
        enable: bool, model_info: LanguageModelInfo, is_cross_region: bool
    ) -> bool:
        return (
            enable
            and model_info.supports_performance_optimization
            and not is_cross_region
        )

    @staticmethod
    def _should_enable_thinking(enable: bool, model_info: LanguageModelInfo) -> bool:
        return enable and model_info.supports_thinking


class BatchProcessor(BaseModel):
    max_concurrency: int = Field(default=5, ge=1)
    retry_multiplier: float = Field(default=30.0, ge=1.0)
    retry_max_wait: int = Field(default=120, ge=0)
    max_retries: int = Field(default=5, ge=1)
    batch_size: int = Field(default=10, ge=1)

    def execute_with_fallback(
        self,
        items_to_process: list[Any],
        prepare_inputs_func: Callable[[list[Any]], list[dict[str, Any]]],
        batch_func: Callable[..., list[Any]],
        sequential_func: Callable[..., Any],
        task_name: str,
        run_config: dict[str, Any] | None = None,
        show_progress: bool = True,
    ) -> list[Any]:
        if not items_to_process:
            return []
        max_concurrency = (
            run_config.get("max_concurrency", self.max_concurrency)
            if run_config
            else self.max_concurrency
        )
        batch_size = (
            run_config.get("batch_size", self.batch_size)
            if run_config
            else self.batch_size
        )
        prepared_batch_func = self._create_batch_func(batch_func, max_concurrency)
        retrying_sequential_func = self._create_retry_decorator(task_name)(
            sequential_func
        )
        all_results = []
        num_items = len(items_to_process)
        num_chunks = math.ceil(num_items / batch_size)
        logger.info(
            "Starting processing for '%s': %d items in %d chunks (batch size: %d)",
            task_name,
            num_items,
            num_chunks,
            batch_size,
        )
        for i in tqdm(
            range(0, num_items, batch_size),
            desc=f"Processing: {task_name}",
            disable=not show_progress,
        ):
            chunk_items = items_to_process[i : i + batch_size]
            chunk_num = (i // batch_size) + 1
            logger.debug(
                "Processing chunk %d/%d (%d items)",
                chunk_num,
                num_chunks,
                len(chunk_items),
            )
            chunk_inputs = prepare_inputs_func(chunk_items)
            if not chunk_inputs:
                logger.warning(
                    "No valid inputs prepared for chunk %d, skipping", chunk_num
                )
                continue
            try:
                logger.debug("Attempting batch processing for chunk %d", chunk_num)
                chunk_results = prepared_batch_func(chunk_inputs)
                all_results.extend(chunk_results)
                logger.debug("Chunk %d processed successfully in batch mode", chunk_num)
            except Exception as e:
                logger.warning(
                    "Batch processing failed for chunk %d: %s. Falling back to sequential processing",
                    chunk_num,
                    e,
                )
                chunk_results = self._process_sequentially_with_fallback(
                    chunk_inputs,
                    retrying_sequential_func,
                    f"{task_name} (chunk {chunk_num})",
                    show_progress=show_progress,
                )
                all_results.extend(chunk_results)
        logger.info("Completed '%s': processed %d results", task_name, len(all_results))
        return all_results

    @staticmethod
    def _create_batch_func(
        batch_func: Callable[..., list[Any]], max_concurrency: int
    ) -> Callable:
        def _batch_func(inputs: list[dict[str, Any]]) -> list[Any]:
            return batch_func(
                inputs, config=RunnableConfig(max_concurrency=max_concurrency)
            )

        return _batch_func

    def _create_retry_decorator(self, operation_name: str) -> Callable:
        return tenacity.retry(
            wait=tenacity.wait_exponential(
                multiplier=self.retry_multiplier, max=self.retry_max_wait
            ),
            stop=tenacity.stop_after_attempt(self.max_retries),
            before_sleep=self._create_retry_log_callback(operation_name),
            reraise=True,
        )

    @staticmethod
    def _create_retry_log_callback(operation_name: str) -> Callable:
        def log_retry(retry_state):
            wait_time = retry_state.next_action.sleep if retry_state.next_action else 0
            logger.warning(
                "Retrying '%s' (attempt %d failed). Waiting %.1fs",
                operation_name,
                retry_state.attempt_number,
                wait_time,
            )

        return log_retry

    @staticmethod
    def _process_sequentially_with_fallback(
        inputs: list[dict[str, Any]],
        sequential_func: Callable[[dict[str, Any]], Any],
        task_name: str,
        show_progress: bool = True,
    ) -> list[Any]:
        logger.info("Processing %d items sequentially for '%s'", len(inputs), task_name)
        results: list[Any] = []
        progress_desc = f"Sequential Processing: '{task_name}'"
        successful_count = 0
        for single_input in tqdm(inputs, desc=progress_desc, disable=not show_progress):
            try:
                results.append(sequential_func(single_input))
                successful_count += 1
            except Exception as e:
                logger.error(
                    "Sequential processing failed for single item in '%s': %s",
                    task_name,
                    e,
                )
                # Append None (not skip) so the results list stays positionally
                # aligned with inputs — callers zip results back onto their
                # items and a dropped element would misattribute every result
                # after it. Callers must treat None as "failed, no output".
                results.append(None)
        logger.info(
            "Sequential processing completed for '%s': %d/%d items processed successfully",
            task_name,
            successful_count,
            len(inputs),
        )
        return results


class HTMLTagOutputParser(BaseOutputParser):
    tag_names: str | list[str]
    # Tags that MUST be present and non-empty; if any is missing the parser
    # raises OutputParserException so a wrapping OutputFixingParser actually
    # triggers its repair model. Empty by default to preserve lenient behavior
    # (the filter tolerates missing optional fields downstream).
    required_tags: list[str] = Field(default_factory=list)

    def parse(self, text: str) -> str | dict[str, str]:
        if not text:
            if self.required_tags:
                raise OutputParserException(
                    f"Empty model output; required tags missing: {self.required_tags}"
                )
            return {} if isinstance(self.tag_names, list) else ""
        soup = BeautifulSoup(text, "html.parser")
        parsed: dict[str, str] = {}
        tag_list = (
            self.tag_names if isinstance(self.tag_names, list) else [self.tag_names]
        )
        for tag_name in tag_list:
            if tag := soup.find(tag_name):
                if hasattr(tag, "decode_contents"):
                    parsed[tag_name] = str(tag.decode_contents()).strip()
                else:
                    parsed[tag_name] = str(tag).strip()
        missing = [t for t in self.required_tags if not parsed.get(t)]
        if missing:
            raise OutputParserException(
                f"Required tag(s) missing or empty in model output: {missing}. "
                f"Got tags: {sorted(parsed)}"
            )
        if isinstance(self.tag_names, list):
            return parsed
        return next(iter(parsed.values()), "")

    @property
    def _type(self) -> str:
        return "html_tag_output_parser"


class RetryableBase:
    @staticmethod
    def _retry(operation_name: str) -> Callable:
        return tenacity.retry(
            wait=tenacity.wait_exponential(
                multiplier=RETRY_MULTIPLIER, max=RETRY_MAX_WAIT
            ),
            stop=tenacity.stop_after_attempt(MAX_RETRIES),
            before_sleep=lambda retry_state: logger.warning(
                "Retrying '%s' (attempt %d failed). Waiting %.1fs",
                operation_name,
                retry_state.attempt_number,
                retry_state.next_action.sleep if retry_state.next_action else 0,
            ),
            reraise=True,
        )


def validate_email(email: str) -> bool:
    return bool(re.match(EMAIL_PATTERN, email.strip()))


def validate_emails(emails: list[str]) -> list[str]:
    valid_emails = [email.strip() for email in emails if validate_email(email)]
    if len(valid_emails) < len(emails):
        logger.warning(
            "Filtered out %d invalid email addresses", len(emails) - len(valid_emails)
        )
    return valid_emails


def format_alarm(
    *,
    event: str,
    status: str,
    fields: dict[str, str],
    project: str = "tech-digest",
    timestamp: datetime | None = None,
) -> tuple[str, str]:
    """Build a ``(subject, message)`` pair in the project family's unified alarm
    format, shared verbatim across omnisummary/paper-bridge/scholar-lens:

        Subject: [<project>] <event> — <STATUS>

        <event> <STATUS>

        Key:   Value

        — 2026-06-10 04:12:00 UTC

    ``status`` is a short uppercase state (``FAILED``/``ALERT``). ``fields`` is an
    ordered mapping; single-line values render as an aligned ``Key: Value`` block,
    multi-line values render under their own ``Key:`` header. Omit a row by leaving
    it out of the dict.
    """
    ts = (timestamp or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M:%S")
    subject = f"[{project}] {event} — {status}"

    inline = {k: v for k, v in fields.items() if "\n" not in v}
    block = {k: v for k, v in fields.items() if "\n" in v}

    lines = [f"{event} {status}", ""]
    if inline:
        width = max(len(k) for k in inline)
        lines += [f"{k + ':':<{width + 1}} {v}" for k, v in inline.items()]
    for k, v in block.items():
        lines += ["", f"{k}:", v.strip("\n")]
    lines.append("")
    lines.append(f"— {ts} UTC")

    return subject, "\n".join(lines)


def get_date_range(
    end_date_str: str | None, days_back: int
) -> tuple[datetime, datetime]:
    if end_date_str:
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").replace(tzinfo=UTC)
    else:
        end_date = datetime.now(UTC)
    end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)
    start_date = end_date - timedelta(days=days_back)
    return start_date, end_date


def measure_execution_time(func: Callable) -> Callable:
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        execution_time = time.time() - start_time
        logger.info(
            "'%s' execution time: %.2fs (%.2fmin)",
            func.__name__,
            execution_time,
            execution_time / 60,
        )
        return result

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        logger.info(
            "'%s' execution time: %.2fs (%.2fmin)",
            func.__name__,
            execution_time,
            execution_time / 60,
        )
        return result

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

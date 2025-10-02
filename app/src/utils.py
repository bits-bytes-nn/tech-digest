import functools
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar, Generic, TypeVar
from datetime import datetime, timedelta, timezone

import asyncio
import math
import boto3
import tenacity
from botocore.config import Config as BotoConfig
from bs4 import BeautifulSoup, Tag
from langchain.schema import BaseOutputParser
from langchain_aws import ChatBedrock, ChatBedrockConverse
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm

from .constants import LanguageModelId
from .logger import logger

MAX_RETRIES: int = 5
RETRY_MAX_WAIT: int = 120
RETRY_MULTIPLIER: int = 30


class LanguageModelInfo(BaseModel):
    context_window_size: int
    max_output_tokens: int
    supports_performance_optimization: bool = False
    supports_prompt_caching: bool = False
    supports_thinking: bool = False


_LANGUAGE_MODEL_INFO: dict[LanguageModelId, LanguageModelInfo] = {
    LanguageModelId.CLAUDE_V3_5_HAIKU: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=4096,
        supports_performance_optimization=True,
        supports_prompt_caching=True,
    ),
    LanguageModelId.CLAUDE_V3_5_SONNET: LanguageModelInfo(
        context_window_size=200000, max_output_tokens=4096
    ),
    LanguageModelId.CLAUDE_V3_5_SONNET_V2: LanguageModelInfo(
        context_window_size=200000, max_output_tokens=4096, supports_prompt_caching=True
    ),
    LanguageModelId.CLAUDE_V3_7_SONNET: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=8192,
        supports_prompt_caching=True,
        supports_thinking=True,
    ),
    LanguageModelId.CLAUDE_V4_SONNET: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=8192,
        supports_prompt_caching=True,
        supports_thinking=True,
    ),
    LanguageModelId.CLAUDE_V4_5_SONNET: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=8192,
        supports_prompt_caching=True,
        supports_thinking=True,
    ),
    LanguageModelId.CLAUDE_V4_OPUS: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=8192,
        supports_prompt_caching=True,
        supports_thinking=True,
    ),
    LanguageModelId.CLAUDE_V4_1_OPUS: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=8192,
        supports_prompt_caching=True,
        supports_thinking=True,
    ),
    # NOTE: add new models here
}

ModelIdT = TypeVar("ModelIdT")
ModelInfoT = TypeVar("ModelInfoT")
WrapperT = TypeVar("WrapperT")


class BaseBedrockModelFactory(Generic[ModelIdT, ModelInfoT, WrapperT], ABC):
    BOTO_READ_TIMEOUT: ClassVar[int] = 300
    BOTO_MAX_ATTEMPTS: ClassVar[int] = 3

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
            retries={"max_attempts": self.BOTO_MAX_ATTEMPTS},
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

    def get_supported_models(self) -> list[ModelIdT]:
        return list(self._get_model_info_dict().keys())


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
    DEFAULT_THINKING_BUDGET_TOKENS: ClassVar[int] = 2048
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
        model_config = self._build_model_config(
            model_info, resolved_model_id, is_cross_region, **kwargs
        )
        model_class = ChatBedrockConverse if is_cross_region else ChatBedrock
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
    ):
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
            think_config = {"thinking": {"type": "enabled", "budget_tokens": budget}}
            config.setdefault("additional_model_request_fields", {}).update(
                think_config
            )
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
        if run_config:
            self.max_concurrency = run_config.get(
                "max_concurrency", self.max_concurrency
            )
            self.batch_size = run_config.get("batch_size", self.batch_size)
        prepared_batch_func = self._create_batch_func(batch_func)
        retrying_sequential_func = self._create_retry_decorator(task_name)(
            sequential_func
        )
        all_results = []
        num_items = len(items_to_process)
        num_chunks = math.ceil(num_items / self.batch_size)
        logger.info(
            "Starting processing for '%s': %d items in %d chunks (batch size: %d)",
            task_name,
            num_items,
            num_chunks,
            self.batch_size,
        )
        for i in tqdm(
            range(0, num_items, self.batch_size),
            desc=f"Processing: {task_name}",
            disable=not show_progress,
        ):
            chunk_items = items_to_process[i : i + self.batch_size]
            chunk_num = (i // self.batch_size) + 1
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

    def _create_batch_func(self, batch_func: Callable[..., list[Any]]) -> Callable:
        def _batch_func(inputs: list[dict[str, Any]]) -> list[Any]:
            return batch_func(
                inputs, config=RunnableConfig(max_concurrency=self.max_concurrency)
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
        results = []
        progress_desc = f"Sequential Processing: '{task_name}'"
        successful_count = 0
        for single_input in tqdm(inputs, desc=progress_desc, disable=not show_progress):
            try:
                result = sequential_func(single_input)
                results.append(result)
                successful_count += 1
            except Exception as e:
                logger.error(
                    "Sequential processing failed for single item in '%s': %s",
                    task_name,
                    e,
                )
                continue
        logger.info(
            "Sequential processing completed for '%s': %d/%d items processed successfully",
            task_name,
            successful_count,
            len(inputs),
        )
        return results

    async def aexecute_with_fallback(
        self,
        items_to_process: list[Any],
        prepare_inputs_func: Callable[[list[Any]], list[dict[str, Any]]],
        batch_func: Callable[..., Any],
        sequential_func: Callable[..., Any],
        task_name: str,
        run_config: dict[str, Any] | None = None,
        show_progress: bool = True,
    ) -> list[Any]:
        if not items_to_process:
            return []
        if run_config:
            self.max_concurrency = run_config.get(
                "max_concurrency", self.max_concurrency
            )
            self.batch_size = run_config.get("batch_size", self.batch_size)
        prepared_batch_func = self._create_async_batch_func(batch_func)
        retrying_sequential_func = self._create_retry_decorator(task_name)(
            sequential_func
        )
        all_results = []
        num_items = len(items_to_process)
        num_chunks = math.ceil(num_items / self.batch_size)
        logger.info(
            "Starting async processing for '%s': %d items in %d chunks (batch size: %d)",
            task_name,
            num_items,
            num_chunks,
            self.batch_size,
        )
        chunk_iterator = async_tqdm(
            range(0, num_items, self.batch_size),
            desc=f"Processing: {task_name}",
            disable=not show_progress,
        )
        for i in chunk_iterator:
            chunk_items = items_to_process[i : i + self.batch_size]
            chunk_num = (i // self.batch_size) + 1
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
                chunk_results = await prepared_batch_func(chunk_inputs)
                all_results.extend(chunk_results)
            except Exception as e:
                logger.warning(
                    "Async batch processing failed for chunk %d: %s. Falling back to concurrent sequential processing",
                    chunk_num,
                    e,
                )
                chunk_results = await self._aprocess_sequentially_with_fallback(
                    chunk_inputs,
                    retrying_sequential_func,
                    f"{task_name} (chunk {chunk_num})",
                    show_progress,
                )
                all_results.extend(chunk_results)
        logger.info("Completed '%s': processed %d results", task_name, len(all_results))
        return all_results

    def _create_async_batch_func(self, batch_func: Callable[..., Any]) -> Callable:
        async def _batch_func(inputs: list[dict[str, Any]]) -> list[Any]:
            return await batch_func(
                inputs, config=RunnableConfig(max_concurrency=self.max_concurrency)
            )

        return _batch_func

    async def _aprocess_sequentially_with_fallback(
        self,
        inputs: list[dict[str, Any]],
        sequential_func: Callable[[dict[str, Any]], Any],
        task_name: str,
        show_progress: bool = True,
    ) -> list[Any]:
        logger.info("Processing %d items concurrently for '%s'", len(inputs), task_name)
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _process_one(single_input):
            async with semaphore:
                try:
                    return await sequential_func(single_input)
                except Exception as e:
                    logger.error(
                        "Concurrent sequential processing failed for item in '%s': %s",
                        task_name,
                        e,
                    )
                    return None

        tasks = [_process_one(single_input) for single_input in inputs]
        progress_desc = f"Concurrent Fallback: '{task_name}'"
        results = await async_tqdm.gather(
            *tasks, disable=not show_progress, desc=progress_desc
        )
        successful_results = [res for res in results if res is not None]
        logger.info(
            "Concurrent sequential processing completed for '%s': %d/%d items processed successfully",
            task_name,
            len(successful_results),
            len(inputs),
        )
        return successful_results


class HTMLTagOutputParser(BaseOutputParser):
    tag_names: str | list[str]

    def parse(self, text: str) -> str | dict[str, str]:
        if not text:
            return {} if isinstance(self.tag_names, list) else ""
        soup = BeautifulSoup(text, "html.parser")
        parsed: dict[str, str] = {}
        tag_list = (
            self.tag_names if isinstance(self.tag_names, list) else [self.tag_names]
        )
        for tag_name in tag_list:
            if tag := soup.find(tag_name):
                if isinstance(tag, Tag) and hasattr(tag, "decode_contents"):
                    parsed[tag_name] = str(tag.decode_contents(formatter=None)).strip()
                else:
                    parsed[tag_name] = str(tag).strip()
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


def get_date_range(
    end_date_str: str | None, days_back: int
) -> tuple[datetime, datetime]:
    if end_date_str:
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
    else:
        end_date = datetime.now(timezone.utc)
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

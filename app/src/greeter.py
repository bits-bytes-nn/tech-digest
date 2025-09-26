from collections.abc import Callable

import boto3
import tenacity
from langchain.schema import StrOutputParser

from .constants import Language, LanguageModelId
from .logger import logger
from .prompts import GreetingPrompt
from .utils import BedrockLanguageModelFactory

MAX_RETRIES: int = 5
RETRY_MULTIPLIER: int = 30
RETRY_MAX_WAIT: int = 120


class Greeter:
    def __init__(
        self,
        boto_session: boto3.Session,
        greeting_model_id: LanguageModelId,
        language: Language = Language.KO,
    ):
        self.boto_session = boto_session
        self.greeting_model_id = greeting_model_id
        self.language = language
        llm_factory = BedrockLanguageModelFactory(boto_session=self.boto_session)
        model_info = llm_factory.get_model_info(greeting_model_id)
        if not model_info:
            raise ValueError(f"Unsupported model ID: '{greeting_model_id.value}'")
        greeter_llm = llm_factory.get_model(
            model_id=greeting_model_id,
            max_tokens=model_info.max_output_tokens,
            temperature=0.4,
        )
        self.greeter_chain = (
            GreetingPrompt.for_language(language).get_prompt()
            | greeter_llm
            | StrOutputParser()
        )

    def _create_retry_decorator(self, operation_name: str) -> Callable:
        return tenacity.retry(
            wait=tenacity.wait_exponential(
                multiplier=RETRY_MULTIPLIER, max=RETRY_MAX_WAIT
            ),
            stop=tenacity.stop_after_attempt(MAX_RETRIES),
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

    def greet(self, context: str | None = None) -> str:
        decorator = self._create_retry_decorator("greeting")
        decorated_invoke = decorator(self.greeter_chain.invoke)
        return decorated_invoke({"context": context or ""})

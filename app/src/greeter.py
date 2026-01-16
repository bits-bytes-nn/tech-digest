import boto3
from langchain_core.output_parsers import StrOutputParser

from .constants import Language, LanguageModelId
from .prompts import GreetingPrompt
from .utils import BedrockLanguageModelFactory, RetryableBase

MAX_RETRIES: int = 5
RETRY_MULTIPLIER: int = 30
RETRY_MAX_WAIT: int = 120


class Greeter(RetryableBase):
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
        greeting_llm = llm_factory.get_model(
            model_id=greeting_model_id, temperature=0.4
        )
        self.greeter = (
            GreetingPrompt.for_language(language).get_prompt()
            | greeting_llm
            | StrOutputParser()
        )

    @RetryableBase._retry("greeting")
    def greet(self, context: str | None = None) -> str:
        return self.greeter.invoke({"context": context or ""})

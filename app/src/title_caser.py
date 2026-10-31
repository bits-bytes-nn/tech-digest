import json

import boto3
from langchain_core.output_parsers import StrOutputParser

from app.domain.constants import LanguageModelId
from app.domain.text import parse_recased
from app.logger import logger

from .model_factory import BedrockLanguageModelFactory
from .prompts import TitleCasePrompt
from .utils import retry_with_backoff

# Output cap for the title-case call. Left unset, the request reserves the
# model's full output maximum against the tokens-per-minute quota; a JSON array
# of a dozen titles is a few hundred tokens, and the rest covers reasoning.
TITLE_CASE_MAX_TOKENS: int = 4096


class TitleCaser:
    def __init__(self, boto_session: boto3.Session, model_id: LanguageModelId):
        llm_factory = BedrockLanguageModelFactory(boto_session=boto_session)
        llm = llm_factory.get_model(
            model_id=model_id,
            temperature=0.0,
            max_tokens=TITLE_CASE_MAX_TOKENS,
            stage="title-case",
        )
        self.chain = TitleCasePrompt.get_prompt() | llm | StrOutputParser()

    @retry_with_backoff("title-case")
    def _invoke(self, titles: list[str]) -> str:
        return self.chain.invoke({"titles": json.dumps(titles, ensure_ascii=False)})

    def recase(self, titles: list[str]) -> list[str]:
        """The titles in title case, or unchanged if the model fails.

        Casing is cosmetic and every summary is already paid for, so no failure
        here may lose the issue. A malformed answer is not re-asked: the same
        prompt tends to fail the same way, and the original titles are fine.
        """
        if not titles:
            return []
        try:
            return parse_recased(self._invoke(titles), titles)
        except Exception as e:
            logger.error("Title casing failed (%s); keeping the feed titles.", e)
            return list(titles)

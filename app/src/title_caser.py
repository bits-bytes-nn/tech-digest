import json

import boto3
from langchain_core.output_parsers import StrOutputParser

from .constants import LanguageModelId
from .logger import logger
from .model_factory import BedrockLanguageModelFactory
from .prompts import TitleCasePrompt
from .utils import retry_with_backoff

# Typographic marks the model tends to swap for their ASCII forms when it
# echoes a title back. Equal for comparison only; the feed's mark is kept.
_ASCII_MARKS = str.maketrans("\u2018\u2019\u201c\u201d\u2013\u2014", "''\"\"--")


def accept_recased(original: str, recased: object) -> str:
    """``original`` with ``recased``'s letter case if the two differ in case
    only, else ``original``.

    The model is trusted to judge casing, not wording: a title it translated,
    shortened or "fixed" would misquote the source article on the card. A curly
    apostrophe or dash it straightened is not a wording change, so the casing is
    taken and the feed's punctuation kept.
    """
    if (
        isinstance(recased, str)
        and len(recased) == len(original)
        and recased.translate(_ASCII_MARKS).lower()
        == original.translate(_ASCII_MARKS).lower()
    ):
        return "".join(
            r if r.lower() == o.lower() else o
            for o, r in zip(original, recased, strict=True)
        )
    logger.warning(
        "Title-case answer %r changes more than case; keeping %r.", recased, original
    )
    return original


def parse_recased(answer: str, originals: list[str]) -> list[str]:
    """Pair the model's JSON array with the input titles, title by title.

    Raises:
        ValueError: The answer is not a JSON array of the input's length.
    """
    # Only the array is parsed, so a code fence or a sentence around it does not
    # cost the whole issue its casing.
    start, end = answer.find("["), answer.rfind("]")
    parsed = json.loads(answer[start : end + 1] if 0 <= start < end else answer)
    if not isinstance(parsed, list) or len(parsed) != len(originals):
        raise ValueError(f"expected a JSON array of {len(originals)} titles")
    return [accept_recased(o, r) for o, r in zip(originals, parsed, strict=True)]


class TitleCaser:
    def __init__(self, boto_session: boto3.Session, model_id: LanguageModelId):
        llm_factory = BedrockLanguageModelFactory(boto_session=boto_session)
        if not llm_factory.get_model_info(model_id):
            raise ValueError(f"Unsupported model ID: '{model_id.value}'")
        llm = llm_factory.get_model(
            model_id=model_id, temperature=0.0, stage="title-case"
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

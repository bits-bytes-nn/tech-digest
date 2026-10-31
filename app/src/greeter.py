import boto3
from langchain_core.output_parsers import StrOutputParser

from .constants import Language, LanguageModelId
from .logger import logger
from .model_factory import BedrockLanguageModelFactory
from .prompts import GreetingPrompt, GreetingRevisionPrompt
from .utils import retry_with_backoff

# Output cap for the greeting call. Left unset, the request reserves the model's
# full output maximum against the tokens-per-minute quota; this leaves ample room
# for a paragraph plus the reasoning some models do whether asked or not.
GREETING_MAX_TOKENS: int = 4096


def measure_greeting(text: str, language: Language) -> int:
    """Length of a greeting in the unit its prompt budgets in.

    Korean is budgeted in characters (a Korean syllable carries roughly as much
    as an English word, so word counts are meaningless); English in words.
    """
    if language is Language.KO:
        return len(text.strip())
    return len(text.split())


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
        greeting_llm = llm_factory.get_model(
            model_id=greeting_model_id,
            temperature=0.4,
            max_tokens=GREETING_MAX_TOKENS,
            stage="greeting",
        )
        self.greeter = (
            GreetingPrompt.for_language(language).get_prompt()
            | greeting_llm
            | StrOutputParser()
        )
        self.reviser = (
            GreetingRevisionPrompt.for_language(language).get_prompt()
            | greeting_llm
            | StrOutputParser()
        )

    @property
    def length_range(self) -> tuple[int, int]:
        return (
            GreetingPrompt.KO_LENGTH_RANGE
            if self.language is Language.KO
            else GreetingPrompt.EN_WORD_RANGE
        )

    def greet(self, context: str | None = None) -> str:
        """Generate the intro, then verify its length and revise once if needed.

        The greeting sits above the first article card, so an over-long intro
        pushes real content below the fold — and length is the constraint the
        cheap greeting model misses most often. Measuring it in code and feeding
        the number back beats restating the rule more loudly in the prompt.
        A still-out-of-range revision is kept, not retried further: a slightly
        long greeting is a cosmetic flaw and must never fail the digest.
        """
        draft = self._generate(context)
        greeting = self._enforce_length(draft)
        # The one spacing slip a model keeps making; fixed in code, not prompted.
        return (
            greeting.replace("이번주", "이번 주")
            if self.language is Language.KO
            else greeting
        )

    @retry_with_backoff("greeting")
    def _generate(self, context: str | None = None) -> str:
        return self.greeter.invoke({"context": context or ""})

    @retry_with_backoff("greeting-revision")
    def _revise(self, draft: str, measured: int) -> str:
        low, high = self.length_range
        return self.reviser.invoke(
            {
                "draft": draft,
                "measured": measured,
                "target_min": low,
                "target_max": high,
                "target": (low + high) // 2,
            }
        )

    def _enforce_length(self, draft: str) -> str:
        low, high = self.length_range
        measured = measure_greeting(draft, self.language)
        if low <= measured <= high:
            logger.debug("Greeting length %d within [%d, %d].", measured, low, high)
            return draft
        logger.warning(
            "Greeting length %d outside [%d, %d]; requesting one revision.",
            measured,
            low,
            high,
        )
        try:
            revised = self._revise(draft, measured)
        except Exception as e:
            logger.error("Greeting revision failed (%s); keeping the draft.", e)
            return draft
        revised_measured = measure_greeting(revised, self.language)
        if low <= revised_measured <= high:
            logger.info(
                "Greeting revised from %d to %d units.", measured, revised_measured
            )
            return revised

        # Neither attempt landed in the window — keep whichever is closer to it,
        # so a failed revision can never make the greeting worse than the draft.
        def distance(value: int) -> int:
            return max(low - value, value - high, 0)

        logger.warning(
            "Greeting still outside [%d, %d] after revision (%d units); "
            "keeping the closer of the two.",
            low,
            high,
            revised_measured,
        )
        return revised if distance(revised_measured) < distance(measured) else draft

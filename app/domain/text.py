"""Deterministic checks on the short texts a model writes: titles, greetings."""

import json

from app.domain.constants import Language
from app.logger import logger

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


def measure_greeting(text: str, language: Language) -> int:
    """Length of a greeting in the unit its prompt budgets in.

    Korean is budgeted in characters (a Korean syllable carries roughly as much
    as an English word, so word counts are meaningless); English in words.
    """
    if language is Language.KO:
        return len(text.strip())
    return len(text.split())

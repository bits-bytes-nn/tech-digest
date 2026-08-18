"""Greeting length enforcement.

The greeting sits above the first article card, so an over-long intro pushes real
content below the fold. A real run produced 230 Korean characters against a
100-150 target and used the exact "past vs present" template the prompt bans, so
the length rule is verified in code and fed back once rather than restated more
loudly in the prompt.
"""

from __future__ import annotations

from app.src.constants import Language
from app.src.greeter import Greeter, measure_greeting
from app.src.prompts import GreetingPrompt


class _Chain:
    """Records invocations and returns queued responses."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def invoke(self, payload):
        self.calls.append(payload)
        if not self._responses:
            raise AssertionError("chain invoked more times than expected")
        result = self._responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _greeter(language: Language, draft, *revisions) -> Greeter:
    g = Greeter.__new__(Greeter)
    g.language = language
    g.greeter = _Chain(draft)
    g.reviser = _Chain(*revisions) if revisions else _Chain()
    return g


def _ko(chars: int) -> str:
    return "안녕 친구들! 난 Peccy야 😎 " + "글" * max(0, chars - 20)


class TestMeasureGreeting:
    def test_korean_measured_in_characters(self):
        assert measure_greeting("가나다", Language.KO) == 3

    def test_english_measured_in_words(self):
        assert measure_greeting("one two three", Language.EN) == 3

    def test_surrounding_whitespace_ignored(self):
        assert measure_greeting("  가나  ", Language.KO) == 2


class TestLengthEnforcement:
    def test_in_range_draft_is_not_revised(self):
        low, high = GreetingPrompt.KO_LENGTH_RANGE
        draft = _ko((low + high) // 2)
        g = _greeter(Language.KO, draft)
        assert g.greet("ctx") == draft
        assert g.reviser.calls == []

    def test_over_long_draft_is_revised_once(self):
        low, high = GreetingPrompt.KO_LENGTH_RANGE
        draft, fixed = _ko(high + 90), _ko(high - 10)
        g = _greeter(Language.KO, draft, fixed)
        assert g.greet("ctx") == fixed
        # The measured count and the window are handed to the reviser, so the
        # model is told how far off it was rather than asked to guess.
        assert g.reviser.calls[0]["measured"] == measure_greeting(draft, Language.KO)
        assert g.reviser.calls[0]["target_min"] == low
        assert g.reviser.calls[0]["target_max"] == high

    def test_too_short_draft_is_also_revised(self):
        low, _high = GreetingPrompt.KO_LENGTH_RANGE
        draft, fixed = _ko(30), _ko(low + 10)
        g = _greeter(Language.KO, draft, fixed)
        assert g.greet("ctx") == fixed

    def test_revision_failure_keeps_the_draft(self):
        """A cosmetic length miss must never fail the digest."""
        _low, high = GreetingPrompt.KO_LENGTH_RANGE
        draft = _ko(high + 90)
        g = _greeter(Language.KO, draft, RuntimeError("bedrock down"))
        g._revise = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down"))  # type: ignore[method-assign]
        assert g.greet("ctx") == draft

    def test_still_out_of_range_keeps_the_closer_attempt(self):
        _low, high = GreetingPrompt.KO_LENGTH_RANGE
        draft = _ko(high + 200)
        closer = _ko(high + 5)
        g = _greeter(Language.KO, draft, closer)
        assert g.greet("ctx") == closer

    def test_worse_revision_is_discarded(self):
        _low, high = GreetingPrompt.KO_LENGTH_RANGE
        draft = _ko(high + 5)
        worse = _ko(high + 400)
        g = _greeter(Language.KO, draft, worse)
        assert g.greet("ctx") == draft

    def test_english_uses_the_word_window(self):
        low, high = GreetingPrompt.EN_WORD_RANGE
        draft = " ".join(["word"] * (high + 40))
        fixed = " ".join(["word"] * ((low + high) // 2))
        g = _greeter(Language.EN, draft, fixed)
        assert g.greet("ctx") == fixed


class TestPromptWindowsAgree:
    def test_code_and_prompt_state_the_same_korean_window(self):
        low, high = GreetingPrompt.KO_LENGTH_RANGE
        template = GreetingPrompt._human_prompt_template[Language.KO]
        assert f"{low}~{high}자" in template

    def test_code_and_prompt_state_the_same_english_window(self):
        low, high = GreetingPrompt.EN_WORD_RANGE
        template = GreetingPrompt._human_prompt_template[Language.EN]
        assert f"{low}-{high} words" in template

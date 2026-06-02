"""Tests for prompt-caching prefix construction: the large static rubric must
move into a fully-static system prefix with the backend-appropriate cache marker
(cachePoint for Converse, cache_control for ChatBedrock — never both), while the
volatile post stays in the human message, otherwise the cache never hits."""

from __future__ import annotations

import pytest

from app.src.constants import FilteringCriteria, Language
from app.src.prompts.prompts import FilteringPrompt, SummarizationPrompt


def _split_messages(prompt_cls, **fmt):
    tmpl = prompt_cls.get_prompt(enable_prompt_cache=True)
    msgs = tmpl.format_messages(**fmt)
    return msgs[0], msgs[1]  # system, human


class TestFilteringCacheSplit:
    def _fmt(self):
        return {
            "post": "ACTUAL_POST_BODY",
            "original_title": "Some Title",
            "included_topics": "Reinforcement Learning",
            "excluded_topics": "Biology",
        }

    def test_converse_system_has_cachepoint_only(self):
        # Default backend is Converse → cachePoint block, no cache_control
        # (a cache_control text block is fine for Converse, but cachePoint must
        # NOT leak into a ChatBedrock request — see TestBackendRoundTrip).
        tmpl = FilteringPrompt.for_criteria(FilteringCriteria.ALL).get_prompt(
            enable_prompt_cache=True, use_converse=True
        )
        system = tmpl.format_messages(**self._fmt())[0]
        assert isinstance(system.content, list)
        assert any("cachePoint" in b for b in system.content)

    def test_chatbedrock_system_has_cache_control_only(self):
        tmpl = FilteringPrompt.for_criteria(FilteringCriteria.ALL).get_prompt(
            enable_prompt_cache=True, use_converse=False
        )
        system = tmpl.format_messages(**self._fmt())[0]
        assert isinstance(system.content, list)
        text_blocks = [b for b in system.content if b.get("type") == "text"]
        assert text_blocks and "cache_control" in text_blocks[0]
        # No cachePoint on the ChatBedrock path (it would crash that backend).
        assert not any("cachePoint" in b for b in system.content)

    def test_post_not_in_cached_system_prefix(self):
        system, human = _split_messages(
            FilteringPrompt.for_criteria(FilteringCriteria.ALL), **self._fmt()
        )
        sys_text = system.content[0]["text"]
        # The volatile post must NOT be in the cached prefix.
        assert "ACTUAL_POST_BODY" not in sys_text
        assert "ACTUAL_POST_BODY" in human.content

    def test_rubric_moved_into_system(self):
        system, human = _split_messages(
            FilteringPrompt.for_criteria(FilteringCriteria.ALL), **self._fmt()
        )
        sys_text = system.content[0]["text"]
        # The rubric (EVALUATION PROCESS onward) lives in the system prefix.
        assert "EVALUATION PROCESS" in sys_text
        assert "EVALUATION PROCESS" not in human.content

    def test_system_prefix_is_large_enough_to_cache(self):
        system, _ = _split_messages(
            FilteringPrompt.for_criteria(FilteringCriteria.ALL), **self._fmt()
        )
        # Bedrock requires ~1024 tokens minimum; ~4 chars/token heuristic.
        assert len(system.content[0]["text"]) > 4000


class TestSummarizationCacheSplit:
    def test_post_stays_in_human(self):
        system, human = _split_messages(
            SummarizationPrompt.for_language(Language.KO), post="ACTUAL_POST_BODY"
        )
        sys_text = system.content[0]["text"]
        assert "ACTUAL_POST_BODY" not in sys_text
        assert "ACTUAL_POST_BODY" in human.content
        assert "CORE PRINCIPLES" in sys_text


class TestBehavioralPreservation:
    """The split must be a lossless partition — the model receives the same
    instructions (modulo the 4 reworded directional phrases, which are applied
    to both paths)."""

    def test_filter_split_reconstructs_human_template(self):
        cls = FilteringPrompt.for_criteria(FilteringCriteria.ALL)
        new_system, new_human = cls._split_for_cache(
            cls.system_prompt_template, cls.human_prompt_template
        )
        # The moved tail begins at the marker; head + tail == original human.
        tail = new_system.split(cls.system_prompt_template, 1)[-1].lstrip("\n")
        assert (new_human + "\n\n" + tail).strip() == cls.human_prompt_template.strip()

    def test_topic_variables_still_render(self):
        _system, human = _split_messages(
            FilteringPrompt.for_criteria(FilteringCriteria.ALL),
            post="P",
            original_title="T",
            included_topics="Reinforcement Learning",
            excluded_topics="Biology",
        )
        # The topic data stays in the human message (run-stable, before marker).
        assert "Reinforcement Learning" in human.content
        assert "Biology" in human.content


class TestBackendRoundTrip:
    """The produced messages must survive each langchain_aws backend's
    message-conversion without raising — this is the bug class that a
    structure-only test misses."""

    def _messages(self, use_converse: bool):
        tmpl = FilteringPrompt.for_criteria(FilteringCriteria.ALL).get_prompt(
            enable_prompt_cache=True, use_converse=use_converse
        )
        return tmpl.format_messages(
            post="P", original_title="T", included_topics="RL", excluded_topics="Bio"
        )

    def test_converse_conversion_does_not_raise(self):
        # Private langchain_aws helper — importorskip so a dep upgrade that
        # relocates it degrades to a skip rather than failing the suite.
        mod = pytest.importorskip("langchain_aws.chat_models.bedrock_converse")
        _messages_to_bedrock = mod._messages_to_bedrock

        msgs = self._messages(use_converse=True)
        _bedrock_msgs, system = _messages_to_bedrock(msgs)
        # cachePoint must reach the Converse system array.
        assert any("cachePoint" in block for block in system)

    def test_chatbedrock_conversion_does_not_raise(self):
        mod = pytest.importorskip("langchain_aws.chat_models.bedrock")
        _format_anthropic_messages = mod._format_anthropic_messages

        msgs = self._messages(use_converse=False)
        # Must not raise "System message content item must be type 'text'".
        system, _formatted = _format_anthropic_messages(msgs)
        # cache_control must be present on the system text block.
        assert system and any(
            isinstance(b, dict) and b.get("cache_control") for b in system
        )


class TestNoCachePathUnchanged:
    def test_filter_without_cache_is_plain_two_message_prompt(self):
        tmpl = FilteringPrompt.for_criteria(FilteringCriteria.ALL).get_prompt(
            enable_prompt_cache=False
        )
        msgs = tmpl.format_messages(
            post="X",
            original_title="T",
            included_topics="RL",
            excluded_topics="Bio",
        )
        assert len(msgs) == 2
        # Plain string content, no cache block list.
        assert isinstance(msgs[0].content, str)
        # Full rubric still present (just not split out).
        assert "EVALUATION PROCESS" in msgs[1].content

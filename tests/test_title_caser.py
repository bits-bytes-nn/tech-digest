"""Unit tests for the title caser's deterministic boundary: the model judges
casing, code makes sure nothing but casing changes."""

from __future__ import annotations

import json

import pytest

from app.application.title_case import TitleCaser
from app.domain.text import accept_recased, parse_recased

TITLES = [
    "Multimodal open d1 decision models for the edge",
    "Fine-tuning gpt-oss with LoRA",
]


class TestAcceptRecased:
    def test_a_case_only_change_is_accepted(self):
        assert accept_recased("a guide to agents", "A Guide to Agents") == (
            "A Guide to Agents"
        )

    @pytest.mark.parametrize(
        "answer",
        [
            "A Guide To AI Agents",  # a word added
            "A Guide: Agents",  # punctuation changed
            "에이전트 가이드",  # translated
            None,
            ["A Guide to Agents"],
        ],
    )
    def test_anything_beyond_case_keeps_the_original(self, answer):
        assert accept_recased("a guide to agents", answer) == "a guide to agents"

    @pytest.mark.parametrize(
        ("original", "answer", "expected"),
        [
            (
                "TorchNeuron, Amazon\u2019s new PyTorch backend",
                "TorchNeuron, Amazon's New PyTorch Backend",
                "TorchNeuron, Amazon\u2019s New PyTorch Backend",
            ),
            (
                "QA automation with Nova Act \u2013 part 3",
                "QA Automation with Nova Act - Part 3",
                "QA Automation with Nova Act \u2013 Part 3",
            ),
            (
                "\u201cagentic\u201d retrieval",
                '"Agentic" Retrieval',
                "\u201cAgentic\u201d Retrieval",
            ),
        ],
    )
    def test_straightened_punctuation_takes_the_case_and_keeps_the_mark(
        self, original, answer, expected
    ):
        assert accept_recased(original, answer) == expected

    def test_a_straight_apostrophe_in_the_feed_is_not_curled(self):
        assert accept_recased("amazon's backend", "Amazon\u2019s Backend") == (
            "Amazon's Backend"
        )


class TestParseRecased:
    def test_pairs_titles_in_order(self):
        answer = json.dumps(
            [
                "Multimodal Open d1 Decision Models for the Edge",
                "Fine-Tuning gpt-oss with LoRA",
            ]
        )
        assert parse_recased(answer, TITLES) == [
            "Multimodal Open d1 Decision Models for the Edge",
            "Fine-Tuning gpt-oss with LoRA",
        ]

    def test_a_sentence_around_the_array_is_tolerated(self):
        answer = "Here are the titles:\n" + json.dumps(TITLES) + "\nDone."
        assert parse_recased(answer, TITLES) == TITLES

    def test_a_code_fence_is_tolerated(self):
        answer = "```json\n" + json.dumps([t.upper() for t in TITLES]) + "\n```"
        assert parse_recased(answer, TITLES) == [t.upper() for t in TITLES]

    def test_one_bad_title_does_not_discard_the_others(self):
        answer = json.dumps(["Multimodal Models", "Fine-Tuning gpt-oss with LoRA"])
        assert parse_recased(answer, TITLES) == [
            TITLES[0],
            "Fine-Tuning gpt-oss with LoRA",
        ]

    @pytest.mark.parametrize(
        "answer", ["not json", json.dumps(["only one"]), json.dumps({"a": 1})]
    )
    def test_a_malformed_answer_raises(self, answer):
        with pytest.raises(ValueError):
            parse_recased(answer, TITLES)


class TestRecase:
    def _caser(self, invoke) -> TitleCaser:
        caser = TitleCaser.__new__(TitleCaser)
        caser._invoke = invoke  # type: ignore[method-assign]
        return caser

    def test_model_failure_keeps_the_feed_titles(self):
        def fail(_titles):
            raise RuntimeError("Bedrock unavailable")

        assert self._caser(fail).recase(TITLES) == TITLES

    def test_malformed_answer_keeps_the_feed_titles(self):
        assert self._caser(lambda _t: "Sure! Here you go.").recase(TITLES) == TITLES

    def test_no_titles_makes_no_call(self):
        assert self._caser(lambda _t: pytest.fail("called")).recase([]) == []


def test_titles_with_braces_reach_the_prompt_verbatim():
    from app.application.prompts.prompts import TitleCasePrompt

    titles = json.dumps(["Using {placeholders} in f-strings"])
    rendered = TitleCasePrompt.get_prompt().format_messages(titles=titles)
    assert "Using {placeholders} in f-strings" in rendered[-1].content

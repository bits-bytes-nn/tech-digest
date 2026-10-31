from abc import ABC
from dataclasses import dataclass
from typing import ClassVar

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from app.domain.constants import FilteringCriteria, Language


@dataclass(frozen=True)
class BasePrompt(ABC):
    system_prompt_template: str
    human_prompt_template: str
    input_variables: list[str]
    output_variables: list[str] | None = None

    def __post_init__(self) -> None:
        self._validate_prompt_variables()

    def _validate_prompt_variables(self) -> None:
        if not self.input_variables:
            return
        for var in self.input_variables:
            if not isinstance(var, str) or not var:
                raise ValueError(f"Invalid input variable: {var}")
            if (
                f"{{{var}}}" not in self.human_prompt_template
                and f"{{{var}}}" not in self.system_prompt_template
            ):
                raise ValueError(
                    f"Input variable '{var}' not found in any prompt template."
                )

    # When set, prompt caching moves everything from this marker onward in the
    # human template into the (cached) system prefix. This lets the large static
    # rubric/instructions — which would otherwise sit after the volatile
    # ``{post}`` and be uncacheable — become a stable cache prefix, WITHOUT
    # duplicating the text (the human template remains the single source).
    cache_split_marker: ClassVar[str | None] = None

    @classmethod
    def get_prompt(
        cls, enable_prompt_cache: bool = False, use_converse: bool = True
    ) -> ChatPromptTemplate:
        # Construct once to run variable validation (__post_init__).
        cls(
            input_variables=cls.input_variables,
            output_variables=cls.output_variables,
            system_prompt_template=cls.system_prompt_template,
            human_prompt_template=cls.human_prompt_template,
        )
        system_template = cls.system_prompt_template
        human_template = cls.human_prompt_template

        if enable_prompt_cache and cls.cache_split_marker:
            system_template, human_template = cls._split_for_cache(
                system_template, human_template
            )

        if not enable_prompt_cache:
            return ChatPromptTemplate.from_messages(
                [
                    SystemMessagePromptTemplate.from_template(system_template),
                    HumanMessagePromptTemplate.from_template(human_template),
                ]
            )

        # Mark the system prompt as a cache prefix. The two Bedrock backends use
        # DIFFERENT, mutually-incompatible cache markers, so emit only the one
        # matching the backend that will actually run:
        #   - ChatBedrockConverse: a trailing ``{"cachePoint": {...}}`` block.
        #     (Putting a ``cache_control`` text block here is fine, but a
        #      cachePoint block in a ChatBedrock request raises a ValueError.)
        #   - ChatBedrock: ``cache_control: {ephemeral}`` on the text block.
        # The volatile ``{post}`` stays in the human message (after the cache
        # breakpoint); the system prefix is byte-identical across every post in
        # a run, so the cache hits.
        if use_converse:
            system_content: list[dict] = [
                {"type": "text", "text": system_template},
                {"cachePoint": {"type": "default"}},
            ]
        else:
            system_content = [
                {
                    "type": "text",
                    "text": system_template,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        return ChatPromptTemplate.from_messages(
            [("system", system_content), ("human", human_template)]
        )

    @classmethod
    def _split_for_cache(
        cls, system_template: str, human_template: str
    ) -> tuple[str, str]:
        """Move the human template's tail (from ``cache_split_marker`` on) into
        the system prefix. Returns ``(new_system, new_human)``."""
        marker = cls.cache_split_marker
        if not marker or marker not in human_template:
            return system_template, human_template
        head, sep, tail = human_template.partition(marker)
        new_system = f"{system_template}\n\n{sep}{tail}".rstrip()
        new_human = head.rstrip()
        return new_system, new_human


class FilteringPrompt(BasePrompt):
    input_variables: list[str] = [
        "post",
        "original_title",
        "source_url",
        "included_topics",
        "excluded_topics",
    ]
    output_variables: list[str] = ["score", "reason"]
    system_prompt_template: str = ""
    human_prompt_template: str = ""
    # Everything from this marker onward is the static rubric; caching moves it
    # into the cached system prefix (the data block before it stays in human).
    cache_split_marker: ClassVar[str | None] = "**EVALUATION PROCESS:**"

    _system_prompt_template: ClassVar[dict[FilteringCriteria, str]] = {
        FilteringCriteria.ALL: """You are an expert machine learning research evaluator with deep expertise in ML
theory, algorithms, and research methodologies.

Your role:
- Assess ML content quality and research value with precision
- Distinguish theoretical advances from basic implementations
- Filter non-ML topics and promotional materials
- Apply evaluation criteria consistently
- Prioritize content matching specified included topics""",
        FilteringCriteria.AMAZON: """You are an expert ML content evaluator specializing in Amazon/AWS machine learning
services and cloud-based ML implementations.

Your role:
- Evaluate ML content within Amazon/AWS context
- Assess technical depth of AWS ML service implementations
- Identify research value in cloud-based ML environments
- Filter based on Amazon/AWS ML relevance
- Prioritize content matching specified included topics""",
    }

    _human_prompt_template: ClassVar[dict[FilteringCriteria, str]] = {
        FilteringCriteria.ALL: """Evaluate this content for machine learning research quality and relevance.

**ORIGINAL TITLE:**
{original_title}

**SOURCE URL:** {source_url}

**CONTENT:**
{post}

**INCLUDED TOPICS:** {included_topics}
**EXCLUDED TOPICS:** {excluded_topics}

---

**EVALUATION PROCESS:**

**STEP 1: TOPIC VALIDATION**

REJECT if content contains:
- Non-ML topics (data analytics, BI, visualization, databases, ETL, general software development)
- Topics in the provided Excluded Topics list
- Pure implementation tutorials without research insights
- Marketing/promotional materials
- Hardware reviews or basic platform tutorials

**EXCEPTION:** If content SUBSTANTIALLY covers any included topic (not just mentions), AUTO-ACCEPT and proceed to
scoring — unless its primary subject is an excluded topic. An excluded topic overrides an included one: a
model or service built for an excluded field scores 0.05 even when it also matches an included topic.
If unsure whether an included topic is the PRIMARY focus, score the content as other content.

**INCLUDED TOPIC MATCHING CRITERIA (STRICT):**
- Topic must be a PRIMARY focus of the content (>30% of content)
- NOT just mentioned in passing or as a minor example
- Content must provide meaningful depth on the included topic
- Simple mentions, brief examples, or tangential references DO NOT qualify

ACCEPT if content has:
- Topics from the provided Included Topics list as PRIMARY focus (HIGHEST PRIORITY - automatic acceptance)
- Core ML focus (theory, algorithms, model architectures, research methods)
- Novel research contributions or theoretical insights
- Deep technical ML understanding
- **High-quality practical guides with deep ML insights and proven best practices**

---

**STEP 2: QUALITY SCORING — pick ONE anchor score**

Choose the SINGLE anchor score below whose description best matches the
content. Do NOT add or subtract fractional modifiers — pick the closest anchor and commit.
This keeps scoring reproducible: the same content must receive the same score every time.

**INCLUDED-TOPIC CONTENT** (content SUBSTANTIALLY covers an included topic — >30% as a
PRIMARY focus, not a passing mention). Included topics take priority, so these anchors sit
ABOVE every anchor for other content:
- **0.95** — The included topic IS the post's subject (e.g. the release or launch itself), with technical substance.
- **0.90** — The included topic is the primary focus, covered in depth, though the post is not the release itself.
- **0.85** — Meets the substantial-coverage bar with basic depth, or with noticeable promotional mixing.

**OTHER ML CONTENT** (no included topic as primary focus). Pick the band, then the anchor:
- **0.80** — Groundbreaking: novel theory with rigorous proofs, or a landmark algorithmic/empirical result.
- **0.75** — Excellent: strong novel contribution or a deep, battle-tested practical guide.
- **0.70** — Strong: solid empirical study or well-validated improvement with clear insight.
- **0.60** — Good: competent work with some novelty or a useful, well-structured guide.
- **0.50** — Moderate: limited novelty, decent execution, or educational with some new perspective.
- **0.35** — Weak: mostly implementation/tutorial with minimal ML insight.
- **0.15** — Poor: pure implementation or promotional content with negligible research value.
- **0.05** — Not ML: off-topic, marketing, or an excluded topic.

**UNCERTAINTY:** When torn between two anchors, always choose the LOWER one. Promotional
mixing or poor organisation is a reason to choose the lower anchor, never to adjust a score.

---

**OUTPUT FORMAT:**
<reason>[Explain concisely: (1) Topic validation — does the content SUBSTANTIALLY cover (>30%
primary focus) an included topic? If so, name it. (2) Which anchor score you chose and why.]</reason>
<score>[The chosen anchor exactly, two decimals, e.g. 0.90, 0.70, 0.15]</score>""",
        FilteringCriteria.AMAZON: """Evaluate this content for ML technical quality with focus on Amazon/AWS ML
implementations.

**ORIGINAL TITLE:**
{original_title}

**SOURCE URL:** {source_url}

**CONTENT:**
{post}

**INCLUDED TOPICS:** {included_topics}
**EXCLUDED TOPICS:** {excluded_topics}

---

**EVALUATION PROCESS:**

**STEP 1: TOPIC VALIDATION**

REJECT if content contains:
- Non-ML topics (data analytics, BI, visualization, databases, ETL)
- Topics in the provided Excluded Topics list
- Non-Amazon promotional content
- Competitor platform focus

**EXCEPTION:** If content SUBSTANTIALLY covers any included topic with AWS context (>30% as primary focus), AUTO-ACCEPT
and proceed to scoring — unless its primary subject is an excluded topic.
An excluded topic overrides an included one: a model or service built for an excluded field scores 0.05 even
when it also matches an included topic.
If unsure whether an included topic is the PRIMARY focus, score the content as other content.

**INCLUDED TOPIC MATCHING CRITERIA (STRICT):**
- Topic must be a PRIMARY focus with AWS/Amazon context (>30% of content)
- NOT just mentioned in passing or as a minor example
- Content must provide meaningful depth on the included topic within AWS ecosystem
- Simple mentions, brief examples, or tangential references DO NOT qualify

ACCEPT if content has:
- Topics from the provided Included Topics list as PRIMARY focus with AWS context (HIGHEST PRIORITY)
- Core ML focus with Amazon/AWS context (80%+ of content)
- Technical depth in AWS ML services (SageMaker, Bedrock, etc.)
- Advanced ML understanding within AWS ecosystem
- **High-quality AWS ML practical guides with deep insights and proven best practices**
- ML research or engineering that Amazon itself publishes (the source URL shows Amazon or AWS published
  it). That is Amazon context even when the text names no AWS service: score it on its ML merit

---

**STEP 2: QUALITY SCORING — pick ONE anchor score**

Choose the SINGLE anchor score below whose description best matches the
content. Do NOT add or subtract fractional modifiers — pick the closest anchor and commit.
This keeps scoring reproducible: the same content must receive the same score every time.

**INCLUDED-TOPIC CONTENT** (content SUBSTANTIALLY covers an included topic with Amazon/AWS
context — >30% as a PRIMARY focus, not a passing mention). These anchors sit ABOVE every
anchor for other content:
- **0.95** — The included topic IS the post's subject (e.g. the AWS launch announcement itself), with technical substance.
- **0.90** — The included topic is a primary focus, covered in depth, but shares the post with another purpose (e.g. a
  new feature introduced through a worked example).
- **0.85** — Meets the substantial-coverage bar with basic depth, or with noticeable promotional mixing.

**OTHER AWS ML CONTENT** (no included topic as primary focus). Pick the band, then the anchor:
- **0.80** — Groundbreaking AWS ML architecture/algorithmic innovation, or landmark result.
- **0.75** — Excellent: strong novel AWS ML contribution or a deep, battle-tested AWS guide.
- **0.70** — Strong: solid AWS ML study/implementation with clear technical depth.
- **0.60** — Good: competent AWS ML work or a useful, well-structured AWS guide.
- **0.50** — Moderate: moderate AWS ML depth, or educational with some insight.
- **0.35** — Weak: basic AWS ML tutorial with minimal depth.
- **0.15** — Poor: marketing disguised as AWS technical content, a launch roundup or recap, or minimal AWS ML focus.
- **0.05** — Not relevant: a competitor's platform, non-ML content, or an excluded topic.

**UNCERTAINTY:** When torn between two anchors, always choose the LOWER one. Promotional
mixing or poor organisation is a reason to choose the lower anchor, never to adjust a score.

---

**OUTPUT FORMAT:**
<reason>[Explain concisely: (1) Topic validation — does the content SUBSTANTIALLY cover (>30%
primary focus) an included topic with AWS context? If so, name it. (2) Which anchor score you
chose and why.]</reason>
<score>[The chosen anchor exactly, two decimals, e.g. 0.90, 0.70, 0.15]</score>""",
    }

    @classmethod
    def for_criteria(
        cls, criteria: FilteringCriteria = FilteringCriteria.ALL
    ) -> type["FilteringPrompt"]:
        prompt_class = type(
            f"{criteria.name.capitalize()}FilteringPrompt",
            (cls,),
            {
                "system_prompt_template": cls._system_prompt_template[criteria],
                "human_prompt_template": cls._human_prompt_template[criteria],
            },
        )
        return prompt_class


class GreetingPrompt(BasePrompt):
    input_variables: list[str] = ["context"]
    output_variables: list[str] = ["greeting"]
    # Visible-length window enforced by Greeter after generation. Kept here so
    # the number the prompt states and the number the code checks cannot drift.
    KO_LENGTH_RANGE: ClassVar[tuple[int, int]] = (100, 150)
    EN_WORD_RANGE: ClassVar[tuple[int, int]] = (50, 75)
    system_prompt_template: str = """You are Peccy, a seasoned tech expert with deep knowledge of systems architecture,
technology history, and software craftsmanship. You communicate with a direct, confident style while sharing valuable
technical insights and connecting historical context to modern developments. Your expertise makes you a trusted voice
in the tech community."""
    human_prompt_template: str = ""

    _human_prompt_template: ClassVar[dict[Language, str]] = {
        Language.EN: """Write a weekly newsletter introduction in English using the context below.

**CONTEXT:**
<context>{context}</context>

**LENGTH:** 50-75 words in total, including the opener. It sits above the first article, so
going long pushes that article below the fold.

**REQUIREMENTS:**
1. Start with "Hey friends! I'm Peccy 😎"
2. Include ONE specific observation that connects DIRECTLY to a theme actually covered in this
   week's articles (see the context). BANNED, because it is what a model reaches for by default:
   the "back then X, now Y" / "we used to X, now we Y" contrast template in any wording. Also
   banned: facts unrelated to this week's topics, and vague claims ("AI is moving fast")
3. If the articles genuinely share a theme, name it concretely. If they do not, do not invent
   one: introduce two or three of them in a phrase each
4. Each card already opens with its one-line summary; do not restate or paraphrase it
5. Casual, confident, technically fluent. No markdown, plain text only
6. No praise or vague adjectives ("groundbreaking", "powerful", "amazing", "various"); say the
   specific thing instead

**OUTPUT:** Newsletter introduction only.""",
        Language.KO: """Write a weekly newsletter introduction in Korean using the context below.

**CONTEXT:**
<context>{context}</context>

**분량:** 공백 포함 **100~150자**, 3~4문장. 여는 인사("안녕 친구들! 난 Peccy야 😎")도 포함입니다.
첫 아티클 위에 놓이므로 길어지면 첫 카드가 화면 아래로 밀려납니다.

**REQUIREMENTS:**
1. "안녕 친구들! 난 Peccy야 😎"로 시작
2. 이번 주 아티클이 **실제로 다루는** 주제와 직접 연결되는 구체적인 관찰 하나를 넣으세요.
   **금지** — 모델이 기본값으로 집어드는 패턴이라서입니다: "예전엔 X였는데 이제 Y야",
   "과거엔 X했지만 이제는 Y해" 같은 **과거-현재 대조 템플릿을 어떤 표현으로든** 쓰지 마세요.
   이번 주 주제와 무관한 잡지식, "AI가 빠르게 발전하고 있어" 같은 뻔한 서술도 금지
3. 아티클끼리 실제로 겹치는 주제가 있으면 그것을 구체적으로 짚고, 없으면 억지로 묶지 말고
   두세 편을 각각 한 구절로 소개하세요
4. 각 카드에는 그 아티클의 한 줄 요약이 실리므로, 그 문장을 옮기거나 풀어 쓰지 마세요
5. 반말, 자신감 있는 어조. 기술 용어는 필요하면 영어 그대로
6. "혁신적인", "강력한", "놀라운", "다양한" 같은 칭찬·뭉갬 형용사와 "~라는 점이야" 같은
   맺음은 쓰지 마세요. 구체적인 것을 그냥 말하면 됩니다
7. 평문만 출력 — 마크다운 금지

**OUTPUT:** Newsletter introduction in Korean only.""",
    }

    @classmethod
    def for_language(cls, language: Language = Language.KO) -> type["GreetingPrompt"]:
        prompt_class = type(
            f"{language.name.capitalize()}GreetingPrompt",
            (cls,),
            {
                "system_prompt_template": cls.system_prompt_template,
                "human_prompt_template": cls._human_prompt_template[language],
            },
        )
        return prompt_class


class GreetingRevisionPrompt(BasePrompt):
    """Ask the model to bring an over/under-length greeting into the window.

    Length is the instruction the greeting model misses most, and something code
    can measure exactly, so the measured count is fed back once with a concrete
    target (the window's midpoint). A bare "must be 100-150" left most revisions
    still too long; a single number plus "drop a whole sentence" lands most of them.
    """

    input_variables: list[str] = [
        "draft",
        "measured",
        "target_min",
        "target_max",
        "target",
    ]
    output_variables: list[str] = ["greeting"]
    system_prompt_template: str = GreetingPrompt.system_prompt_template
    human_prompt_template: str = ""

    _human_prompt_template: ClassVar[dict[Language, str]] = {
        Language.EN: """Your draft newsletter introduction is {measured} words. It must be
{target_min}-{target_max} words, so rewrite it to **about {target} words**.

**DRAFT:**
<draft>{draft}</draft>

Count the words rather than estimating. If it is too long, start by dropping a whole sentence
(three sentences including the opener are enough). Keep the opener "Hey friends! I'm Peccy 😎",
the specific observation and the hand-off to the articles, and keep the casual confident tone.
Do NOT introduce a "back then X, now Y" contrast. Plain text only.

**OUTPUT:** The revised introduction only.""",
        Language.KO: """작성한 뉴스레터 인트로가 {measured}자입니다. {target_min}~{target_max}자여야 하니
**{target}자 안팎**으로 다시 쓰세요.

**DRAFT:**
<draft>{draft}</draft>

글자 수는 어림하지 말고 세세요. 너무 길면 문장 하나를 통째로 빼는 것부터 하세요(여는 인사
포함 3문장이면 충분합니다). 여는 인사 "안녕 친구들! 난 Peccy야 😎"와 구체적인 관찰, 아티클로
넘기는 마지막 문장은 유지하고, 반말·자신감 있는 어조도 그대로 두세요. "예전엔 X였는데 이제
Y야" 같은 과거-현재 대조는 넣지 마세요. 평문만 출력하세요.

**OUTPUT:** 수정된 인트로만 출력.""",
    }

    @classmethod
    def for_language(
        cls, language: Language = Language.KO
    ) -> type["GreetingRevisionPrompt"]:
        return type(
            f"{language.name.capitalize()}GreetingRevisionPrompt",
            (cls,),
            {
                "system_prompt_template": cls.system_prompt_template,
                "human_prompt_template": cls._human_prompt_template[language],
            },
        )


class TitleCasePrompt(BasePrompt):
    """Recase the issue's card titles to one headline style.

    Feeds mix sentence case and title case. Which words are product, model or
    project names written in lowercase (gpt-oss, vLLM, d1) is a judgment a word
    list cannot make, so the model decides; ``TitleCaser`` rejects any answer
    that changes more than letter case.
    """

    input_variables: list[str] = ["titles"]
    output_variables: list[str] = ["titles"]
    system_prompt_template: str = """You are a copy editor for an English-language technical
newsletter. You apply headline title case and change nothing else."""
    human_prompt_template: str = """Convert each title below to title case (Chicago style).

**TITLES (JSON array):**
{titles}

**RULES:**
1. Capitalize the first and last word, the first word after a colon, and every noun, pronoun,
   verb (including "Is", "Are", "Be"), adjective and adverb.
2. Lowercase articles (a, an, the), coordinating conjunctions (and, but, or, nor, for, so, yet),
   "as", "to" and prepositions (of, in, on, at, by, for, from, with, into, via, vs, ...), unless
   rule 1 applies.
3. In a hyphenated compound, case each part by rules 1-2 ("Context-Aware", "End-to-End").
4. Keep acronyms, product, model, library, company and person names exactly as they are
   officially written, including lowercase or mixed-case ones (gpt-oss, vLLM, iOS, d1,
   AgentCore). Keep version numbers, code identifiers, file names and URLs unchanged.
5. Change letter case only: never add, remove, reorder, translate or respell characters,
   spaces or punctuation. A title not written in English is returned unchanged.

**OUTPUT:** A JSON array of the converted titles, same count and order as the input. Nothing
else."""


class SummarizationPrompt(BasePrompt):
    input_variables: list[str] = ["post"]
    output_variables: list[str] = ["summary", "one_liner", "tags", "urls"]
    human_prompt_template: str = ""
    system_prompt_template: str = """You are an expert technical writer and content analyst specializing in software
engineering, machine learning, and system architecture. Your goal is to create clear, engaging, and accurate
explanations that make complex technical concepts accessible without sacrificing depth or precision."""
    # The static analysis instructions begin here; caching moves them into the
    # cached system prefix while the post itself stays in the human message.
    cache_split_marker: ClassVar[str | None] = "**CORE PRINCIPLES:**"

    _human_prompt_template: ClassVar[dict[Language, str]] = {
        Language.EN: """Analyze and explain the following blog post in a comprehensive yet accessible manner:

**CONTENT TO ANALYZE:**
<post>{post}</post>

**CORE PRINCIPLES:**

1. **Factual Accuracy First**
   - Base your analysis ONLY on information explicitly stated in the source material
   - Never speculate, assume, or infer information not present in the original content, and add no
     examples or scenarios the article does not give

2. **Conversational Yet Precise**
   - Write as if explaining to a knowledgeable peer, but maintain technical accuracy
   - Help readers understand the "why" behind technical decisions, not just the "what"
   - Depth follows difficulty, not a fixed quota. What a working developer already knows (an API call,
     a config flag, a console click-path) gets one plain sentence. A mechanism that is complex, novel or
     high-context (a training objective, a scheduling algorithm, a term the article coins or takes from
     a paper) is explained so that a developer reading it cold understands it without looking anything
     up: what problem it solves, the term defined at first use, then how it works step by step.
     Explaining a mechanism in depth is not a reason to quote its results early: the figures still
     appear once, in 📊
   - Defining a standard, widely established concept (what a KV cache or LoRA is) from general knowledge
     is allowed and is not speculation. Everything about THIS article's system, numbers and plans still
     comes only from the article
   - Make the technology the subject of the sentence ("Haiku 5.5 routes...", not "The article explains
     that Haiku 5.5 routes..."). Name the article or vendor only to attribute a figure or claim that is
     theirs ("Anthropic reports...")

3. **Length Budget**
   - Total <summary> body: **1,300-2,000 words of visible text**, excluding HTML tags
   - Rough per-section shape — 🛠️ should be the LONGEST section by a clear margin:
     📌 4-6 sentences / 🔄 2-3 paragraphs / 🛠️ **4-6 paragraphs** /
     📊 1-2 paragraphs (a table may carry the metrics) / 🔮 3-4 sentences
   - The reader is a practitioner who came for the technical substance. Depth in
     🛠️ is the POINT of this newsletter — never sacrifice a mechanism, a
     parameter, a measured number or a design tradeoff to hit a word count
   - When you must cut, cut RESTATEMENT: anything already said in another section,
     scene-setting, and generic framing. Never cut technical specifics
   - Do not scale length UP with the source. A 20,000-word paper gets the same
     budget as a 3,000-word post; a longer source means being more selective about
     WHICH details, not writing more. A short source may yield a shorter summary:
     never pad with guesses, restatement or commentary to reach the budget

4. **Section Distinctiveness**
   - Each section must carry NEW information. Never restate figures, sentences, or conclusions
     already given in an earlier section
   - The 📊 Results section is where the article's headline results are stated, ONCE. 📌 and 🛠️
     name the mechanism and describe its effect in words; the figure ("5x faster", "30% cost
     reduction") appears in 📊 only. A supporting figure needed to explain a trade-off stays in 🛠️
     and is not repeated in 📊. Do not point to another section ("see the results below", "(figures
     in 📊)"); just describe the effect. Do not add business implications, ratios you computed, or
     forecasts the article does not state
   - The 🔮 Future section is NOT a place to re-state limitations already listed in 🛠️. Cover the
     roadmap, expansion directions, and opportunities the article explicitly mentions. If there is
     nothing new and forward-looking to add, OMIT this section

5. **No Filler Register**
   - These are shapes a model reaches for by default, and a reader recognises them as such.
     Do not use them:
     - Praise adjectives as assessment — "groundbreaking", "revolutionary", "powerful",
       "impressive", "remarkable", "cutting-edge", "seamless", "robust". In this newsletter
       the MEASUREMENTS do the assessing: not "impressive throughput" but "3,200 Gbps"
     - "various" / "a variety of" without saying which or how many
     - The "not just X, it's Y" contrast, and "it's worth noting that"
     - Meta-narration: "let's dive in", "let's take a look", "in conclusion", "to summarize".
       Each section IS the content, not an announcement of it
     - "delve into", "landscape", "realm", "underscores", "a testament to", "game-changer"
   - Prefer the concrete verb to the abstract one: not "leverages caching" but "keeps decoded
     KV blocks in the cache so the prefix is not recomputed"

**SPECIFICITY RULES (this newsletter's core value):**
- Carry over EVERY measured number the article reports, WITH its unit ("22-66% lower", "8,000 tokens",
  "3,200 Gbps", "16 GPUs across 2 nodes"). Never replace a number with an adjective like "significantly
  faster" — once the figures are gone the summary is useless to a practitioner
- If the article contains code, configuration, CLI or a schema, include at least the most load-bearing
  snippet in `<pre><code>` (never invent one that is not in the source)
- When three or more numbers are being compared, put them in a TABLE rather than in prose, and do not
  repeat the table's values in the surrounding text
- When you shorten a code snippet, mark each omitted span with `...` rather than silently splicing lines
- Name parameters, flags and thresholds exactly as the article does (`routingThreshold`, `PD_BUFFER_SIZE`)

**SECTION SKIP RULE:**
- If the source material does not contain sufficient information for a section, OMIT that section entirely
- Never write about what the article does NOT say, in any section: not as a section's filler ("no information
  is available") and not as a sentence inside one ("the article does not describe its architecture", "no
  timeline was given", "pricing was not disclosed"), nor as a pointer elsewhere ("the architecture is in
  the paper; this post does not repeat it") or a table cell ("not reported"). A sentence about an omission
  tells the reader nothing about the technology. Leave the missing detail out without comment; if the
  article cites a paper for it, list the paper in <urls>. A limitation the article itself states ("the eval
  set is internal") is content and stays
- Only include sections where you can provide meaningful, substantive content based on the source material

**REQUIRED STRUCTURE:**

Provide your analysis within <summary> tags using the following sections. Include only those sections for which the
source material provides sufficient substantive content:

<h3>📌 Why This Matters</h3>
Explain the significance and relevance of this content. Focus on the problem being addressed, why the approach is
noteworthy. Be concise while covering essential
points. Open and close with the specific reason THIS article matters (the problem it solves, what is new). Do NOT end
with a formulaic audience call-out like "a noteworthy read for developers and architects" — when every article ends the
same way, it reads as boilerplate.

<h3>🔄 Core Architecture and Workflow</h3>
Describe the system design and workflow clearly. Cover main components, their interactions, and key design choices.
Include relevant images, each WRAPPED IN A FIGURE WITH A CAPTION so it is never dropped into the card without
context: <figure><img src="full_url" alt="descriptive text"><figcaption>One short sentence saying what the reader
should take from this image</figcaption></figure> (complete URLs only: https://example.com/image.jpg; the caption must
add information, not repeat the alt text). Avoid redundancy
with the technical deep dive section.

<h3>🛠️ Technical Deep Dive</h3>
Provide a comprehensive technical walkthrough covering: core technical concepts, key terminology, critical code sections 
with explanations, technical decisions, tools and frameworks used, optimization strategies, performance characteristics, 
scalability considerations, edge cases, and known limitations. Use <pre><code> for code blocks. Write 
as a cohesive narrative. Be detailed but avoid repeating information already covered in other 
sections.

<h3>📊 Results and Impact</h3>
Present the measured outcomes the article reports, including cost figures when it gives them. Write it as prose, a table, or both.
Focus only on results explicitly stated in the article, and open with a figure.

<h3>🔮 Future Directions</h3>
Discuss future directions and integration possibilities explicitly mentioned in the article. Be concise and
avoid speculation.

**FORMATTING REQUIREMENTS:**
- Use HTML tags exclusively (no markdown)
- Apply <strong> for critical technical concepts
- Use <em> for technical terms requiring emphasis
- Format comparisons and data in HTML tables when appropriate
- No sub-headings (h4-h6) inside a section; a bold run-in label at the start of a paragraph is fine
- Inline code goes in <code>, never Markdown backticks; write math as plain text or Unicode (QKᵀ), never LaTeX
- Ensure all code uses <pre><code> blocks
- Make technical explanations accessible but accurate
- Use one name per concept throughout; do not alternate between synonyms for the same component
- No meta-commentary about following these instructions

**FINAL CHECK (re-read your draft and fix it before answering):**
Knowing a rule is not the same as having followed it. These are the slips that most often survive
into a published issue, so look for each one in your draft and fix it before you output:
1. Go through every number in 📊. If it also appears in 📌, 🔄 or 🛠️, keep it in 📊 and replace the
   earlier mention with words ("cuts latency"). Then delete any 📊 sentence about business
   value, a ratio you computed, or a forecast the article does not make.
2. Find sentences whose verb is "provides", "offers" or "supports" and say what the feature actually does.
3. Find praise adjectives, "not just X, it's Y" and meta-narration (principle 5) and rewrite them. Read the
   last sentence of 📌: if it says who should read this or that it is worth noting, end on the finding instead.
4. Count the words in one_liner. If it is outside 15-28 words, fix it.
5. Check that every image src is copied character for character from the article, query string included.
6. Delete every sentence whose point is that something is missing (not described, not reported, no figure
   given, "these are the only figures reported"), every sentence about how you built the summary ("I left
   out"), every reason or intention the article does not state ("likely because", "this suggests"), and
   every causal link it does not make ("based on this").

**OUTPUT FORMAT:**
<summary>[Your comprehensive technical explanation following the structure above]</summary>
<one_liner>[ONE plain-text sentence a reader scanning the digest can use to decide whether to read this
card. 15-28 words — count them before answering; it must not wrap past two lines in the card.
State the most concrete, specific thing the article establishes — the mechanism plus its measured effect
where available (e.g. "Splitting prefill and decode onto separate GPU pools over RDMA cuts per-token latency 22-66%
at high concurrency"). NO adjectives of praise, NO "this article explains/explores/discusses", NO restating the
title. Write a complete sentence ending in a period, not a headline fragment. Plain text only, no HTML,
no markdown]</one_liner>
<tags>[5-7 specific technical topics in Title Case, comma-separated - focus on distinctive technologies, methodologies,
or architectural patterns explicitly mentioned in the article - avoid generic terms like "Machine Learning" or "AI"
unless they represent novel approaches discussed]</tags>
<urls>[Essential technical references as HTML links: <a href="url1">Descriptive Title 1</a>, <a href="url2">Descriptive
Title 2</a> - include only URLs explicitly mentioned or directly referenced in the article]</urls>""",
        Language.KO: """Analyze and explain the following blog post in a comprehensive yet accessible manner, writing in
Korean:

**CONTENT TO ANALYZE:**
<post>{post}</post>

**CORE PRINCIPLES:**

1. **Factual Accuracy First**
   - Base your analysis ONLY on information explicitly stated in the source material
   - Never speculate, assume, or infer information not present in the original content, and add no
     examples or scenarios the article does not give

2. **Conversational Yet Precise**
   - Write as if explaining to a knowledgeable peer, but maintain technical accuracy
   - Help readers understand the "why" behind technical decisions, not just the "what"
   - Depth follows difficulty, not a fixed quota. What a working developer already knows (an API call,
     a config flag, a console click-path) gets one plain sentence. A mechanism that is complex, novel or
     high-context (a training objective, a scheduling algorithm, a term the article coins or takes from
     a paper) is explained so that a developer reading it cold understands it without looking anything
     up: what problem it solves, the term defined at first use, then how it works step by step.
     Explaining a mechanism in depth is not a reason to quote its results early: the figures still
     appear once, in 📊
   - Defining a standard, widely established concept (what a KV cache or LoRA is) from general knowledge
     is allowed and is not speculation. Everything about THIS article's system, numbers and plans still
     comes only from the article
   - Make the technology the subject of the sentence ("Haiku 5.5 routes...", not "The article explains
     that Haiku 5.5 routes..."). Name the article or vendor only to attribute a figure or claim that is
     theirs ("Anthropic reports...")

3. **분량 예산 (Length Budget)**
   - <summary> 본문 전체: **한글 4,000~6,000자** (HTML 태그 제외, 가시 텍스트 기준)
   - 섹션별 대략의 배분 — 🛠️가 **확실히 가장 긴 섹션**이어야 합니다:
     📌 4~6문장 / 🔄 2~3단락 / 🛠️ **4~6단락** / 📊 1~2단락(수치는 표로) / 🔮 3~4문장
   - 이 뉴스레터의 독자는 기술적 실체를 보러 온 실무자입니다. **🛠️의 깊이가 이
     뉴스레터의 존재 이유**입니다 — 분량을 맞추려고 메커니즘·파라미터·측정된 수치·설계
     트레이드오프를 **절대 희생하지 마세요**
   - 줄여야 한다면 **중복 서술**을 자르세요: 다른 섹션에서 이미 말한 것, 배경 깔기,
     일반론. 기술적 구체성은 자르지 마세요
   - 원문이 길다고 늘리지는 마세요. 2만 단어 논문도 3천 단어 포스트와 예산이 **같습니다**.
     원문이 길다는 건 더 많이 쓰라는 게 아니라 **어떤** 디테일을 고를지 더 신중하라는
     뜻입니다. 원문이 짧으면 요약도 짧아도 됩니다. 분량을 채우려고 추측·반복·논평을
     덧붙이지 마세요

4. **일관된 문체 (Consistent Register)**
   - 모든 섹션의 모든 문장을 **정중한 합니다체**로 통일하세요 (예: "제안합니다", "달성했습니다",
     "줄였습니다"). 절대 문어체 평서형(한다체: "제안한다", "달성했다")과 섞지 마세요
   - 원문 블로그의 어조를 따라가지 말고, 위 합니다체를 처음부터 끝까지 유지하세요
   - 이 뉴스레터는 여러 아티클을 하나로 묶어 발송하므로, 아티클마다 문체가 다르면 독자가 이질감을
     느낍니다. 문체 통일은 필수입니다

5. **섹션 간 정보 차별화 (Section Distinctiveness)**
   - 각 섹션은 **새로운 정보**를 담아야 합니다. 앞 섹션에서 이미 말한 수치·문장·결론을 뒤 섹션에서
     그대로 반복하지 마세요
   - 원문의 대표 성과 수치(예: "5배 향상", "30% 절감")는 **📊에서 한 번만** 제시합니다. 📌·🛠️에서는
     메커니즘을 설명하고 그 효과는 말로만 서술하세요. 트레이드오프를 설명하는 데 필요한 보조 수치는
     🛠️에 두고 📊에서 반복하지 않습니다. "(수치는 📊)"처럼 다른 섹션을 가리키지 마세요. 원문에 없는 비즈니스 함의, 직접 계산한 비율,
     전망은 덧붙이지 마세요
   - 🔮 향후 섹션은 🛠️에서 이미 열거한 한계점을 뒤집어 재서술하는 곳이 아닙니다. 원문이 명시한
     **로드맵·확장 방향·기회**를 다루세요. 새로 덧붙일 forward-looking 내용이 없으면 이 섹션을 생략하세요

6. **번역투 금지 (원문이 영어여도 한국어로 쓴 문장이어야 합니다)**
   - 원문의 **문장 구조를 옮기지 말고**, 뜻을 파악한 뒤 한국어로 처음부터 쓰세요. 아래 목록을
     외우는 게 아니라 원리를 적용하세요: **같은 뜻을 군더더기 없이 말할 수 있으면 그렇게 씁니다.**
     - "X를 통해 Y" → 수단은 조사로 붙입니다: "샤딩을 통해 분산합니다" → "샤딩으로 분산합니다"
     - "X에 대한/대해" → 대상도 조사로: "지연에 대한 개선" → "지연 개선"
     - "~하는 것이 아니라", "~하는 것을 의미합니다" → 뜻을 동사로 바로 씁니다:
       "캐시를 재사용하는 것을 의미합니다" → "캐시를 재사용합니다"
     - "모델들이", "노드들을" → 한국어는 수를 표시하지 않습니다. 수량 표현이 이미 있으면
       특히 어색합니다: "16개 GPU들" → "16개 GPU"
     - "~를 제공합니다/지원합니다"로 문장을 끝내지 말고 **실제로 하는 일**을 쓰세요:
       "캐싱을 제공합니다" → "결과를 캐시에 보관해 재계산을 건너뜁니다"
     - "~에 의해 처리됩니다" → 행위자를 주어로: "스케줄러가 처리합니다"

7. **상투구 금지 (내용이 없거나, 모델이 기본값으로 집어드는 문형입니다)**
   - 평가는 형용사가 아니라 **측정값이 합니다**. "강력한 성능" 대신 "p99 지연 12ms".
     "혁신적인", "획기적인", "놀라운", "인상적인"을 쓰지 마세요
   - "다양한"은 몇 가지인지로: "다양한 모델에서" → "크기가 다른 세 모델에서"
   - "단순히 X가 아니라 Y입니다" 대조 문형과 "결론적으로", "요약하면", "살펴보겠습니다" 같은
     메타 서술을 쓰지 마세요. 각 섹션은 요약이 아니라 내용입니다
   - 단락을 "~하는 구조입니다"로 맺거나 "핵심은 ~입니다"로 열지 마세요. 앞 문장을 요약하는 마무리
     문장은 지우면 됩니다
   - 절을 "~라는/다는 점"으로 감싸지 마세요. 문장 끝("~다는 점입니다")이든 주어("~다는 점이
     핵심입니다")든 같습니다. 감싸던 절을 서술문으로 내려 쓰면 됩니다:
     "지연이 줄었다는 점이 핵심입니다" → "지연이 22% 줄었습니다"

   **원칙 6·7은 문체 규칙이고, 우선순위는 원칙 3보다 낮습니다.** 표현을 다듬느라 메커니즘·
   파라미터·측정값·코드를 빼지 마세요. 번역투를 피하려고 문장을 짧게 끊는 것도, 강조하려고 앞
   섹션의 수치를 뒤 섹션에서 다시 꺼내는 것도 금지입니다. **고칠 것은 문장이고, 줄일 것은
   중복 서술뿐입니다.**

**구체성 규칙 (이 뉴스레터의 핵심 가치):**
- 원문이 보고하는 **측정값은 단위와 함께 빠짐없이** 옮기세요("22~66% 감소", "8,000토큰",
  "3,200Gbps", "2노드 16 GPU"). 숫자를 "크게 개선"처럼 형용사로 바꾸지 마세요 — 수치가
  사라지면 이 요약은 실무자에게 쓸모가 없어집니다
- 원문에 **코드·설정·CLI·스키마가 있으면**, 가장 핵심적인 조각을 최소 하나
  `<pre><code>`로 실으세요(원문에 없으면 만들어내지 마세요)
- 수치가 3개 이상 비교되면 산문으로 늘어놓지 말고 **표**로 만드세요. 표에 넣은 값은 본문에서 다시 쓰지 마세요
- 코드 조각을 줄일 때는 생략한 부분을 `...`로 표시하세요. 떨어진 줄을 말없이 이어 붙이지 마세요
- 파라미터·플래그·임계값의 **이름을 그대로** 쓰세요(`routingThreshold`, `PD_BUFFER_SIZE`)

**섹션 생략 규칙:**
- 원문에 해당 섹션을 채울 충분한 정보가 없으면, 그 섹션을 **통째로 생략**하세요
- 원문이 **말하지 않은 것에 대해 쓰지 마세요**. 섹션을 채우는 문장("원문에 관련 정보가
  없습니다")도, 섹션 안의 한 문장("글은 구조를 따로 설명하지 않습니다", "일정은 제시하지
  않았습니다", "가격은 공개되지 않았습니다")도, 다른 곳을 가리키는 문장("구조는 논문에 있고 이
  글은 다시 설명하지 않습니다")도, 표의 칸("(점수는 제시되지 않음)")도 금지입니다. 빠진 것에
  대한 문장은 독자에게 기술에 관해 아무것도 알려 주지 않습니다. 빠진 정보는 언급 없이 넘어가고,
  원문이 그 내용을 논문 등으로 미뤘다면 그 문헌을 <urls>에 넣으세요. 원문이 스스로 밝힌
  한계("평가셋은 사내용입니다")는 내용이므로 남깁니다
- 원문을 기반으로 실질적이고 의미 있는 내용을 작성할 수 있는 섹션만 포함하세요

**REQUIRED STRUCTURE:**

Provide your analysis within <summary> tags using the following sections. Include only those sections for which the
source material provides sufficient substantive content:

<h3>📌 왜 이 아티클에 주목해야 하나요?</h3>
Explain the significance and relevance of this content. Focus on the problem being addressed, why the approach is
noteworthy. Be concise while covering essential
points. 이 아티클만의 구체적인 이유(해결하는 문제, 새로운 점)로 시작하고 끝맺으세요. "~개발자와 아키텍트에게
주목할 만한 참고 자료입니다" 같은 정형화된 대상·권유 문구로 마무리하지 마세요 — 매주 여러 아티클이 똑같은 문구로
끝나면 독자에게 상투적으로 읽힙니다.

<h3>🔄 핵심 아키텍처와 동작 방식</h3>
Describe the system design and workflow clearly. Cover main components, their interactions, and key design choices. 
관련 이미지는 문맥 없이 놓이지 않도록 **캡션과 함께 figure로 감싸서** 넣으세요:
<figure><img src="full_url" alt="descriptive text"><figcaption>이 이미지에서 독자가 얻어야 할 것을 한 문장으로</figcaption></figure>
(완전한 URL만 사용: https://example.com/image.jpg. 캡션은 alt 텍스트를 반복하지 말고 정보를 더해야 합니다.)
Avoid redundancy with the technical deep dive section.

<h3>🛠️ 기술적 심층 분석</h3>
Provide a comprehensive technical walkthrough covering: core technical concepts, key terminology, critical code sections 
with explanations, technical decisions, tools and frameworks used, optimization strategies, performance characteristics, 
scalability considerations, edge cases, and known limitations. Use <pre><code> for code blocks. Write 
as a cohesive narrative. Be detailed but avoid repeating information already covered in other 
sections.

<h3>📊 주요 성과</h3>
Present the measured outcomes the article reports, including cost figures when it gives them. Write it as prose, a table, or both.
Focus only on results explicitly stated in the article, and open with a figure.

<h3>🔮 향후 발전 가능성과 기회</h3>
Discuss future directions and integration possibilities explicitly mentioned in the article. Be concise and
avoid speculation.

**FORMATTING REQUIREMENTS:**
- Use HTML tags exclusively (no markdown)
- Apply <strong> for critical technical concepts
- Use <em> for technical terms requiring emphasis
- Format comparisons and data in HTML tables when appropriate
- No sub-headings (h4-h6) inside a section; a bold run-in label at the start of a paragraph is fine
- Inline code goes in <code>, never Markdown backticks; write math as plain text or Unicode (QKᵀ), never LaTeX
- Ensure all code uses <pre><code> blocks
- Make technical explanations accessible but accurate
- Write content in Korean; keep a technical term in English when the Korean rendering would be awkward
- 한 개념에는 처음 고른 용어 하나만 쓰세요. 같은 구성 요소를 "추론"과 "인퍼런스"처럼 번갈아 부르지 마세요
- No meta-commentary about following these instructions

**마지막 점검 (답을 내기 전에 초안을 다시 읽고 고치세요):**
규칙을 아는 것과 지킨 것은 다릅니다. 아래 여섯 가지는 지난 호들에서 실제로 가장 자주 남은
실수이므로, 초안을 쓴 뒤 하나씩 찾아서 고치고 나서 출력하세요.
1. 📊 섹션의 숫자를 하나씩 보세요. 📌·🔄·🛠️에도 같은 숫자가 있으면 📊에 남기고 앞쪽은 "지연을
   줄입니다"처럼 말로 바꾸세요. 그다음 원문에 없는 비즈니스 함의, 직접 계산한 비율, 전망을 담은 📊
   문장을 지우세요.
2. "제공합니다", "지원합니다"로 끝나는 문장을 찾아 그 기능이 실제로 무엇을 하는지로 바꾸세요.
3. "~라는/다는 점" 절, "~하는 것" 명사절, 수량 표현 뒤의 "-들"("16개 GPU들"), 칭찬 형용사와 메타
   서술을 찾아 원칙 6·7대로 고치세요. 📌의 마지막 문장이 누가 읽으면 좋은지, 주목할 만하다는 말로
   끝나면 기술이 하는 일이나 측정 결과로 끝맺으세요.
4. one_liner의 글자 수를 공백을 포함해 세세요. 40~100자를 벗어나면 고치세요.
5. 이미지 src는 원문에 있는 URL을 글자 그대로(쿼리 문자열 포함) 옮겼는지 확인하세요.
6. 무언가가 없다는 것이 요지인 문장은 표현과 상관없이 지우세요("~는 설명하지 않습니다", "~는
   보고되지 않았습니다", "~수치는 없습니다", "글에 나오지 않으므로", "수치는 위 표가 전부입니다").
   요약을 어떻게 만들었는지 밝히는 문장("~는 제외했습니다")도 지우세요. 원문이 밝히지 않은
   이유·의도("~때문으로 보입니다", "~라고 볼 수 있습니다")와 원문에 없는 인과 연결("이를 근거로")도
   지우세요. 원문이 스스로 밝힌 한계만 남깁니다.

**OUTPUT FORMAT:**
<summary>[Your comprehensive technical explanation in Korean following the structure above]</summary>
<one_liner>[한국어 **한 문장**, 공백 포함 **40~100자** — 카드에서 두 줄을 넘지 않는 길이입니다.
쓴 뒤 글자 수를 직접 세고, 100자를 넘으면 줄이세요. 다이제스트를 훑는 독자가 이 카드를 읽을지 판단할 근거가 되도록, 이 아티클이
확립한 가장 구체적인 사실 — 메커니즘 + 측정된 효과 — 을 쓰세요. 예: "prefill과 decode를 별도 GPU 풀로 분리해
동시성이 높을 때 토큰당 지연을 22~66% 줄였습니다." 합니다체를 유지하되, 칭찬 형용사·"이 글은 ~를 다룹니다" 같은
메타 서술·제목 반복은 금지. 평문만(HTML·마크다운 금지)]</one_liner>
<tags>[5-7 specific technical topics in Title Case, comma-separated - focus on distinctive technologies, methodologies,
or architectural patterns explicitly mentioned in the article - avoid generic terms like "Machine Learning" or "AI"
unless they represent novel approaches discussed - write all titles in English]</tags>
<urls>[Essential technical references as HTML links: <a href="url1">Descriptive Title 1</a>, <a href="url2">Descriptive
Title 2</a> - include only URLs explicitly mentioned or directly referenced in the article - write all titles in
English]</urls>""",
    }

    @classmethod
    def for_language(
        cls, language: Language = Language.KO
    ) -> type["SummarizationPrompt"]:
        prompt_class = type(
            f"{language.name.capitalize()}SummarizationPrompt",
            (cls,),
            {
                "system_prompt_template": cls.system_prompt_template,
                "human_prompt_template": cls._human_prompt_template[language],
            },
        )
        return prompt_class

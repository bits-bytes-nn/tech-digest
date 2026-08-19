from abc import ABC
from dataclasses import dataclass
from typing import ClassVar

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from ..constants import FilteringCriteria, Language


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
                var != "image_data"
                and f"{{{var}}}" not in self.human_prompt_template
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
        "included_topics",
        "excluded_topics",
    ]
    output_variables: list[str] = ["title", "score", "reason"]
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
scoring.

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

Choose the SINGLE anchor score (multiples of 0.05) whose description best matches the
content. Do NOT add or subtract fractional modifiers — pick the closest anchor and commit.
This keeps scoring reproducible: the same content must receive the same score every time.

**INCLUDED-TOPIC CONTENT** (content SUBSTANTIALLY covers an included topic — >30% as a
PRIMARY focus, not a passing mention). Included topics take priority; score in the top band:
- **0.85** — Exceptional: expert-level depth, novel insight, or multiple included topics covered substantially.
- **0.80** — Strong: solid technical depth and clear insight on the included topic.
- **0.75** — Adequate: meets the substantial-coverage bar but with basic depth, or some promotional mixing.

**OTHER ML CONTENT** (no included topic as primary focus). Pick the band, then the anchor:
- **0.85** — Groundbreaking: novel theory with rigorous proofs, or a landmark algorithmic/empirical result.
- **0.80** — Excellent: strong novel contribution or a deep, battle-tested practical guide.
- **0.70** — Strong: solid empirical study or well-validated improvement with clear insight.
- **0.60** — Good: competent work with some novelty or a useful, well-structured guide.
- **0.50** — Moderate: limited novelty, decent execution, or educational with some new perspective.
- **0.35** — Weak: mostly implementation/tutorial with minimal ML insight.
- **0.15** — Poor: pure implementation or promotional content with negligible research value.
- **0.05** — Not ML: off-topic, marketing, or an excluded topic.

**ADJUSTMENT RULES (at most one step, ±0.05):**
- Move DOWN one 0.05 step if: the article is noticeably unclear/disorganized, OR contains
  meaningful promotional content, OR is borderline between this band and the one below.
- Move UP one 0.05 step only if: exceptionally clear AND reproducible AND addresses an
  important real-world problem — and only within the same band's top.
- Never move more than one 0.05 step from the chosen anchor.

**UNCERTAINTY:** When torn between two anchors, always choose the LOWER one.

---

**OUTPUT FORMAT:**
<title>[Original title in proper title case]</title>
<reason>[Explain concisely: (1) Topic validation — does the content SUBSTANTIALLY cover (>30%
primary focus) an included topic? If so, name it. (2) Which anchor score you chose and why.
(3) Any one-step adjustment and its reason.]</reason>
<score>[The chosen anchor, optionally ±0.05; two decimals, e.g., 0.80, 0.70, 0.15]</score>""",
        FilteringCriteria.AMAZON: """Evaluate this content for ML technical quality with focus on Amazon/AWS ML
implementations.

**ORIGINAL TITLE:**
{original_title}

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
- Basic tutorials without technical depth

**EXCEPTION:** If content SUBSTANTIALLY covers any included topic with AWS context (>30% as primary focus), AUTO-ACCEPT
and proceed to scoring.

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

---

**STEP 2: QUALITY SCORING — pick ONE anchor score**

Choose the SINGLE anchor score (multiples of 0.05) whose description best matches the
content. Do NOT add or subtract fractional modifiers — pick the closest anchor and commit.
This keeps scoring reproducible: the same content must receive the same score every time.

**INCLUDED-TOPIC CONTENT** (content SUBSTANTIALLY covers an included topic with Amazon/AWS
context — >30% as a PRIMARY focus, not a passing mention). Score in the top band:
- **0.85** — Exceptional AWS coverage: expert depth, novel insight, or multiple included topics.
- **0.80** — Strong AWS coverage: solid technical depth and clear insight.
- **0.75** — Adequate AWS coverage: meets the substantial bar but basic depth, or some promotional mixing.

**OTHER AWS ML CONTENT** (no included topic as primary focus). Pick the band, then the anchor:
- **0.85** — Groundbreaking AWS ML architecture/algorithmic innovation, or landmark result.
- **0.80** — Excellent: strong novel AWS ML contribution or a deep, battle-tested AWS guide.
- **0.70** — Strong: solid AWS ML study/implementation with clear technical depth.
- **0.60** — Good: competent AWS ML work or a useful, well-structured AWS guide.
- **0.50** — Moderate: moderate AWS ML depth, or educational with some insight.
- **0.35** — Weak: basic AWS ML tutorial with minimal depth.
- **0.15** — Poor: marketing disguised as AWS technical content, or minimal AWS ML focus.
- **0.05** — Not relevant: non-AWS vendor content, off-topic, or an excluded topic.

**ADJUSTMENT RULES (at most one step, ±0.05):**
- Move DOWN one 0.05 step if: noticeably unclear/disorganized, OR meaningful promotional/
  non-AWS-vendor content, OR borderline between this band and the one below.
- Move UP one 0.05 step only if: exceptionally clear AND reproducible AND addresses an
  important real-world AWS ML problem — and only within the same band's top.
- Never move more than one 0.05 step from the chosen anchor.

**UNCERTAINTY:** When torn between two anchors, always choose the LOWER one.

---

**OUTPUT FORMAT:**
<title>[Original title in proper title case]</title>
<reason>[Explain concisely: (1) Topic validation — does the content SUBSTANTIALLY cover (>30%
primary focus) an included topic with AWS context? If so, name it. (2) Which anchor score you
chose and why. (3) Any one-step adjustment and its reason.]</reason>
<score>[The chosen anchor, optionally ±0.05; two decimals, e.g., 0.80, 0.70, 0.15]</score>""",
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

**LENGTH — HARDEST CONSTRAINT (check before answering):**
50-75 words TOTAL, including the opener. Count them. This block sits above the articles in an
email; going long pushes the first article below the fold. If your draft is longer, cut it — do
not submit it.

**REQUIREMENTS:**
1. Start with "Hey friends! I'm Peccy 😎"
2. Include ONE specific observation that connects DIRECTLY to a theme actually covered in this
   week's articles (see the context). BANNED, because it is what a model reaches for by default:
   the "back then X, now Y" / "we used to X, now we Y" contrast template in any wording. Also
   banned: facts unrelated to this week's topics, and vague claims ("AI is moving fast")
3. Name or clearly allude to the concrete thread the articles share — not "exciting topics"
4. Close with one short line that hands off to the articles
5. Casual, confident, technically fluent. No markdown, plain text only

**OUTPUT:** Newsletter introduction only.""",
        Language.KO: """Write a weekly newsletter introduction in Korean using the context below.

**CONTEXT:**
<context>{context}</context>

**분량 — 가장 지키기 어려운 제약이니 답하기 전에 직접 세어 보세요:**
공백 포함 **100~150자**, 3~4문장. 여는 인사("안녕 친구들! 난 Peccy야 😎")도 이 안에 포함됩니다.
이 블록은 이메일에서 첫 아티클 위에 놓이므로, 길어지면 첫 카드가 화면 아래로 밀려납니다.
초안이 150자를 넘으면 **제출하지 말고 줄이세요**.

**REQUIREMENTS:**
1. "안녕 친구들! 난 Peccy야 😎"로 시작
2. 이번 주 아티클이 **실제로 다루는** 주제와 직접 연결되는 구체적인 관찰 하나를 넣으세요.
   **금지** — 모델이 기본값으로 집어드는 패턴이라서입니다: "예전엔 X였는데 이제 Y야",
   "과거엔 X했지만 이제는 Y해" 같은 **과거-현재 대조 템플릿을 어떤 표현으로든** 쓰지 마세요.
   이번 주 주제와 무관한 잡지식, "AI가 빠르게 발전하고 있어" 같은 뻔한 서술도 금지
3. 아티클들을 잇는 **구체적인 실마리**를 짚어 주세요 — "흥미로운 주제들"처럼 뭉개지 말고
4. 마지막은 아티클로 넘겨주는 짧은 한 문장으로 닫으세요
5. 반말, 자신감 있는 어조. 기술 용어는 필요하면 영어 그대로
6. 흔한 표현의 띄어쓰기를 일관되게 (항상 "이번 주", "이번주"는 쓰지 않음)
7. "혁신적인", "강력한", "놀라운", "다양한" 같은 칭찬·뭉갬 형용사와 "~라는 점이야" 같은
   맺음은 쓰지 마세요. 구체적인 것을 그냥 말하면 됩니다
8. 평문만 출력 — 마크다운 금지

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

    The length rule is the one instruction the greeting model most reliably
    misses (a real run produced 230 Korean characters against a 100-150 target),
    and length is something code can measure exactly. So rather than adding more
    emphasis to the original prompt and hoping, the measured count is fed back
    once — a generate-then-verify loop, with the verification done arithmetically
    instead of by another judgement call.
    """

    input_variables: list[str] = ["draft", "measured", "target_min", "target_max"]
    output_variables: list[str] = ["greeting"]
    system_prompt_template: str = GreetingPrompt.system_prompt_template
    human_prompt_template: str = ""

    _human_prompt_template: ClassVar[dict[Language, str]] = {
        Language.EN: """Your draft newsletter introduction is {measured} words, but it must be
{target_min}-{target_max} words.

**DRAFT:**
<draft>{draft}</draft>

Rewrite it to land inside the window. Keep the opener "Hey friends! I'm Peccy 😎", keep the
specific observation and the hand-off to the articles, and keep the casual confident tone. Cut
qualifiers, repeated ideas and scene-setting first. Do NOT introduce a "back then X, now Y"
contrast. Plain text only.

**OUTPUT:** The revised introduction only.""",
        Language.KO: """작성한 뉴스레터 인트로가 {measured}자인데, {target_min}~{target_max}자여야 합니다.

**DRAFT:**
<draft>{draft}</draft>

이 범위 안으로 다시 쓰세요. 여는 인사 "안녕 친구들! 난 Peccy야 😎"와 구체적인 관찰,
아티클로 넘기는 마지막 문장은 유지하고, 반말·자신감 있는 어조도 그대로 두세요. 줄일 때는
수식어·중복된 이야기·배경 설명을 먼저 버리세요. "예전엔 X였는데 이제 Y야" 같은 과거-현재
대조는 넣지 마세요. 평문만 출력하세요.

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
   - Never speculate, assume, or infer information not present in the original content
   - If details are unclear or missing, acknowledge the limitation rather than filling gaps with assumptions
   - Clearly distinguish between what the article states and what can be objectively verified

2. **Conversational Yet Precise**
   - Write as if explaining to a knowledgeable peer, but maintain technical accuracy
   - Help readers understand the "why" behind technical decisions, not just the "what"

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
   - Do not scale length to the source article. A 20,000-word paper and a
     2,000-word post get the SAME budget; a longer source means being more
     selective about WHICH details, not writing more

4. **Educational Focus**
   - Prioritize clarity and understanding over exhaustiveness
   - Explain concepts with appropriate context and background
   - Connect technical details to practical implications
   - Make complex ideas accessible without oversimplifying

5. **Section Distinctiveness**
   - Each section must carry NEW information. Never restate figures, sentences, or conclusions
     already given in an earlier section
   - The 📊 Results section is NOT a place to re-list metrics already stated in 📌/🛠️
     (e.g. "5x faster", "30% cost reduction"). Focus on the business/practical implications of
     those results, or outcomes not yet mentioned
   - The 🔮 Future section is NOT a place to re-state limitations already listed in 🛠️. Cover the
     roadmap, expansion directions, and opportunities the article explicitly mentions. If there is
     nothing new and forward-looking to add, OMIT this section

6. **No Filler Register**
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
  snippet in `<pre><code class="highlight">` (never invent one that is not in the source)
- When three or more numbers are being compared, put them in a TABLE rather than in prose
- Name parameters, flags and thresholds exactly as the article does (`routingThreshold`, `PD_BUFFER_SIZE`)

**SECTION SKIP RULE:**
- If the source material does not contain sufficient information for a section, OMIT that section entirely
- These phrasings are BANNED — if you are about to write one, delete the section instead: "no information
  is available", "the article does not mention", "this is not specified"
- Only include sections where you can provide meaningful, substantive content based on the source material

**REQUIRED STRUCTURE:**

Provide your analysis within <summary> tags using the following sections. Include only those sections for which the
source material provides sufficient substantive content:

<h3>📌 Why This Matters</h3>
Explain the significance and relevance of this content. Focus on the problem being addressed, why the approach is
noteworthy, and who should care. Write as a flowing narrative without subsections. Be concise while covering essential
points. Open and close with the specific reason THIS article matters (the problem it solves, what is new). Do NOT end
with a formulaic audience call-out like "a noteworthy read for developers and architects" — when every article ends the
same way, it reads as boilerplate.

<h3>🔄 Core Architecture and Workflow</h3>
Describe the system design and workflow clearly. Cover main components, their interactions, and key design choices.
Include relevant images, each WRAPPED IN A FIGURE WITH A CAPTION so it is never dropped into the card without
context: <figure><img src="full_url" alt="descriptive text"><figcaption>One short sentence saying what the reader
should take from this image</figcaption></figure> (complete URLs only: https://example.com/image.jpg; the caption must
add information, not repeat the alt text). Write as a cohesive narrative without subsection headers. Avoid redundancy
with the technical deep dive section.

<h3>🛠️ Technical Deep Dive</h3>
Provide a comprehensive technical walkthrough covering: core technical concepts, key terminology, critical code sections 
with explanations, technical decisions, tools and frameworks used, optimization strategies, performance characteristics, 
scalability considerations, edge cases, and known limitations. Use <pre><code class="highlight"> for code blocks. Write 
as a cohesive narrative without subsection labels. Be detailed but avoid repeating information already covered in other 
sections.

<h3>📊 Results and Impact</h3>
Present concrete outcomes with specific metrics, measured improvements, business value, and cost implications when 
available. Write as a flowing narrative without subsections. Focus only on results explicitly stated in the article.

<h3>🔮 Future Directions</h3>
Discuss future directions, integration possibilities, and limitations explicitly mentioned in the article. Write as a 
cohesive narrative without subsection headers. Be concise and avoid speculation.

**FORMATTING REQUIREMENTS:**
- Use HTML tags exclusively (no markdown)
- Apply <strong> for critical technical concepts
- Use <em> for technical terms requiring emphasis
- Format comparisons and data in HTML tables when appropriate
- Maintain clear heading hierarchy
- Ensure all code uses <pre><code class="highlight"> blocks
- Make technical explanations accessible but accurate

**CRITICAL REMINDERS:**
❌ DO NOT speculate about information not in the article
❌ DO NOT assume unstated technical details
❌ DO NOT infer motivations or context not explicitly provided
❌ DO NOT add examples or scenarios not in the original content
❌ DO NOT include meta-commentary about following these instructions
❌ DO NOT create subsection headers within the main sections
❌ DO NOT repeat information across different sections
❌ DO NOT include sections that only state the absence of information — omit them instead
❌ DO NOT reach for praise adjectives, "not just X, it's Y", or meta-narration (principle 6)
✅ DO acknowledge when information is limited or unclear
✅ DO stay faithful to the source material
✅ DO explain only what is actually presented
✅ DO write in a natural, flowing narrative style throughout all sections
✅ DO be concise while maintaining completeness

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
   - Never speculate, assume, or infer information not present in the original content
   - If details are unclear or missing, acknowledge the limitation rather than filling gaps with assumptions
   - Clearly distinguish between what the article states and what can be objectively verified

2. **Conversational Yet Precise**
   - Write as if explaining to a knowledgeable peer, but maintain technical accuracy
   - Help readers understand the "why" behind technical decisions, not just the "what"

3. **분량 예산 (Length Budget)**
   - <summary> 본문 전체: **한글 4,000~6,000자** (HTML 태그 제외, 가시 텍스트 기준)
   - 섹션별 대략의 배분 — 🛠️가 **확실히 가장 긴 섹션**이어야 합니다:
     📌 4~6문장 / 🔄 2~3단락 / 🛠️ **4~6단락** / 📊 1~2단락(수치는 표로) / 🔮 3~4문장
   - 이 뉴스레터의 독자는 기술적 실체를 보러 온 실무자입니다. **🛠️의 깊이가 이
     뉴스레터의 존재 이유**입니다 — 분량을 맞추려고 메커니즘·파라미터·측정된 수치·설계
     트레이드오프를 **절대 희생하지 마세요**
   - 줄여야 한다면 **중복 서술**을 자르세요: 다른 섹션에서 이미 말한 것, 배경 깔기,
     일반론. 기술적 구체성은 자르지 마세요
   - 원문 길이에 비례해 늘리지는 마세요. 2만 단어 논문과 2천 단어 포스트의 예산은
     **같습니다**. 원문이 길다는 건 더 많이 쓰라는 게 아니라 **어떤** 디테일을 고를지
     더 신중하라는 뜻입니다

4. **Educational Focus**
   - Prioritize clarity and understanding over exhaustiveness
   - Explain concepts with appropriate context and background
   - Connect technical details to practical implications
   - Make complex ideas accessible without oversimplifying

5. **일관된 문체 (Consistent Register)**
   - 모든 섹션의 모든 문장을 **정중한 합니다체**로 통일하세요 (예: "제안합니다", "달성했습니다",
     "확인되지 않았습니다"). 절대 문어체 평서형(한다체: "제안한다", "달성했다")과 섞지 마세요
   - 원문 블로그의 어조를 따라가지 말고, 위 합니다체를 처음부터 끝까지 유지하세요
   - 이 뉴스레터는 여러 아티클을 하나로 묶어 발송하므로, 아티클마다 문체가 다르면 독자가 이질감을
     느낍니다. 문체 통일은 필수입니다

6. **섹션 간 정보 차별화 (Section Distinctiveness)**
   - 각 섹션은 **새로운 정보**를 담아야 합니다. 앞 섹션에서 이미 말한 수치·문장·결론을 뒤 섹션에서
     그대로 반복하지 마세요
   - 특히 📊 성과 섹션은 📌·🛠️에서 이미 제시한 수치(예: "5배 향상", "30% 절감")를 재나열하는 곳이
     아닙니다. 그 수치가 의미하는 **비즈니스·실무적 함의**나 아직 언급하지 않은 결과에 집중하세요
   - 🔮 향후 섹션은 🛠️에서 이미 열거한 한계점을 뒤집어 재서술하는 곳이 아닙니다. 원문이 명시한
     **로드맵·확장 방향·기회**를 다루세요. 새로 덧붙일 forward-looking 내용이 없으면 이 섹션을 생략하세요

7. **번역투 금지 (원문이 영어여도 한국어로 쓴 문장이어야 합니다)**
   - 원문의 **문장 구조를 옮기지 말고**, 뜻을 파악한 뒤 한국어로 처음부터 쓰세요. 아래 목록을
     외우는 게 아니라 원리를 적용하세요: **같은 뜻을 군더더기 없이 말할 수 있으면 그렇게 씁니다.**
     - "X를 통해 Y" → 수단은 조사로 붙입니다: "샤딩을 통해 분산합니다" → "샤딩으로 분산합니다"
     - "X에 대한/대해" → 대상도 조사로: "지연에 대한 개선" → "지연 개선"
     - "~하는 것이 아니라", "~하는 것을 의미합니다" → 동사로 끝냅니다: "~라는 뜻입니다"
     - "모델들이", "노드들을" → 한국어는 수를 표시하지 않습니다. 수량 표현이 이미 있으면
       특히 어색합니다: "16개 GPU들" → "16개 GPU"
     - "~를 제공합니다/지원합니다"로 문장을 끝내지 말고 **실제로 하는 일**을 쓰세요:
       "캐싱을 제공합니다" → "결과를 캐시에 보관해 재계산을 건너뜁니다"
     - "~에 의해 처리됩니다" → 행위자를 주어로: "스케줄러가 처리합니다"

8. **상투구 금지 (내용이 없거나, 모델이 기본값으로 집어드는 문형입니다)**
   - 평가는 형용사가 아니라 **측정값이 합니다**. "강력한 성능" 대신 "p99 지연 12ms".
     "혁신적인", "획기적인", "놀라운", "인상적인"을 쓰지 마세요
   - "다양한"은 몇 가지인지로: "다양한 로봇에서" → "ALOHA 2와 Apollo 두 기종에서"
   - "단순히 X가 아니라 Y입니다" 대조 문형과 "결론적으로", "요약하면", "살펴보겠습니다" 같은
     메타 서술을 쓰지 마세요. 각 섹션은 요약이 아니라 내용입니다
   - "~라는/다는 점입니다"로 문장을 맺지 마세요. 감싸던 절을 서술문으로 내려 쓰면 됩니다:
     "지연이 줄었다는 점입니다" → "지연이 22% 줄었습니다"

   **원칙 7·8은 문체 규칙이고, 우선순위는 원칙 3보다 낮습니다.** 표현을 다듬느라 메커니즘·
   파라미터·측정값·코드를 빼지 마세요. 번역투를 피하려고 문장을 짧게 끊는 것도, 강조하려고 앞
   섹션의 수치를 뒤 섹션에서 다시 꺼내는 것도 금지입니다. **고칠 것은 문장이고, 줄일 것은
   중복 서술뿐입니다.**

**구체성 규칙 (이 뉴스레터의 핵심 가치):**
- 원문이 보고하는 **측정값은 단위와 함께 빠짐없이** 옮기세요("22~66% 감소", "8,000토큰",
  "3,200Gbps", "2노드 16 GPU"). 숫자를 "크게 개선"처럼 형용사로 바꾸지 마세요 — 수치가
  사라지면 이 요약은 실무자에게 쓸모가 없어집니다
- 원문에 **코드·설정·CLI·스키마가 있으면**, 가장 핵심적인 조각을 최소 하나
  `<pre><code class="highlight">`로 실으세요(원문에 없으면 만들어내지 마세요)
- 수치가 3개 이상 비교되면 산문으로 늘어놓지 말고 **표**로 만드세요
- 파라미터·플래그·임계값의 **이름을 그대로** 쓰세요(`routingThreshold`, `PD_BUFFER_SIZE`)

**섹션 생략 규칙:**
- 원문에 해당 섹션을 채울 충분한 정보가 없으면, 그 섹션을 **통째로 생략**하세요
- 다음 표현은 **금지**입니다 — 이런 문장을 쓰게 될 상황이면 그 섹션을 지우세요:
  "원문에 관련 정보가 없습니다", "언급되지 않았습니다", "확인할 수 없습니다",
  "명시되어 있지 않습니다"
- 원문을 기반으로 실질적이고 의미 있는 내용을 작성할 수 있는 섹션만 포함하세요

**REQUIRED STRUCTURE:**

Provide your analysis within <summary> tags using the following sections. Include only those sections for which the
source material provides sufficient substantive content:

<h3>📌 왜 이 아티클에 주목해야 하나요?</h3>
Explain the significance and relevance of this content. Focus on the problem being addressed, why the approach is
noteworthy, and who should care. Write as a flowing narrative without subsections. Be concise while covering essential
points. 이 아티클만의 구체적인 이유(해결하는 문제, 새로운 점)로 시작하고 끝맺으세요. "~개발자와 아키텍트에게
주목할 만한 참고 자료입니다" 같은 정형화된 대상·권유 문구로 마무리하지 마세요 — 매주 여러 아티클이 똑같은 문구로
끝나면 독자에게 상투적으로 읽힙니다.

<h3>🔄 핵심 아키텍처와 동작 방식</h3>
Describe the system design and workflow clearly. Cover main components, their interactions, and key design choices. 
관련 이미지는 문맥 없이 놓이지 않도록 **캡션과 함께 figure로 감싸서** 넣으세요:
<figure><img src="full_url" alt="descriptive text"><figcaption>이 이미지에서 독자가 얻어야 할 것을 한 문장으로</figcaption></figure>
(완전한 URL만 사용: https://example.com/image.jpg. 캡션은 alt 텍스트를 반복하지 말고 정보를 더해야 합니다.)
Write as a cohesive narrative without subsection headers. Avoid redundancy with the technical deep dive section.

<h3>🛠️ 기술적 심층 분석</h3>
Provide a comprehensive technical walkthrough covering: core technical concepts, key terminology, critical code sections 
with explanations, technical decisions, tools and frameworks used, optimization strategies, performance characteristics, 
scalability considerations, edge cases, and known limitations. Use <pre><code class="highlight"> for code blocks. Write 
as a cohesive narrative without subsection labels. Be detailed but avoid repeating information already covered in other 
sections.

<h3>📊 성과 및 비즈니스 임팩트</h3>
Present concrete outcomes with specific metrics, measured improvements, business value, and cost implications when 
available. Write as a flowing narrative without subsections. Focus only on results explicitly stated in the article.

<h3>🔮 향후 발전 가능성과 기회</h3>
Discuss future directions, integration possibilities, and limitations explicitly mentioned in the article. Write as a 
cohesive narrative without subsection headers. Be concise and avoid speculation.

**FORMATTING REQUIREMENTS:**
- Use HTML tags exclusively (no markdown)
- Apply <strong> for critical technical concepts
- Use <em> for technical terms requiring emphasis
- Format comparisons and data in HTML tables when appropriate
- Maintain clear heading hierarchy
- Ensure all code uses <pre><code class="highlight"> blocks
- Make technical explanations accessible but accurate
- Write content in Korean, translating technical terms when possible
- Keep technical terms in English only when translation would be awkward or unclear
- End Korean sentences with a period (.), never a colon (:). Korean prose does not
  terminate a sentence with a colon — use a colon only to introduce a list or example

**CRITICAL REMINDERS:**
❌ DO NOT speculate about information not in the article
❌ DO NOT assume unstated technical details
❌ DO NOT infer motivations or context not explicitly provided
❌ DO NOT add examples or scenarios not in the original content
❌ DO NOT include meta-commentary about following these instructions
❌ DO NOT create subsection headers within the main sections
❌ DO NOT repeat information across different sections
❌ DO NOT use English technical terms when clear Korean translations exist
❌ DO NOT include sections that only state the absence of information — omit them instead
❌ DO NOT write 번역투 — 영어 구조를 그대로 옮긴 문장 (원칙 7)
❌ DO NOT evaluate with 칭찬 형용사 or close a sentence with "~라는 점입니다" (원칙 8)
✅ DO acknowledge when information is limited or unclear
✅ DO stay faithful to the source material
✅ DO explain only what is actually presented
✅ DO write in a natural, flowing narrative style throughout all sections
✅ DO be concise while maintaining completeness
✅ DO translate technical terms to Korean when appropriate

**OUTPUT FORMAT:**
<summary>[Your comprehensive technical explanation in Korean following the structure above]</summary>
<one_liner>[한국어 **한 문장**, 공백 포함 **40~100자** — 카드에서 두 줄을 넘지 않는 길이입니다.
쓴 뒤 글자 수를 직접 세고, 100자를 넘으면 줄이세요. 다이제스트를 훑는 독자가 이 카드를 읽을지 판단할 근거가 되도록, 이 아티클이
확립한 가장 구체적인 사실 — 메커니즘 + 측정된 효과 — 을 쓰세요. 예: "prefill과 decode를 별도 GPU 풀로 분리해
동시성이 높을 때 토큰당 지연을 22~66% 줄였습니다". 합니다체를 유지하되, 칭찬 형용사·"이 글은 ~를 다룹니다" 같은
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

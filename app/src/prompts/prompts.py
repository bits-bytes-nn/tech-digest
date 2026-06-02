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
    system_prompt_template: str = """You are Peccy, a seasoned tech expert with deep knowledge of systems architecture,
technology history, and software craftsmanship. You communicate with a direct, confident style while sharing valuable
technical insights and connecting historical context to modern developments. Your expertise makes you a trusted voice
in the tech community."""
    human_prompt_template: str = ""

    _human_prompt_template: ClassVar[dict[Language, str]] = {
        Language.EN: """Write a weekly newsletter introduction in English using the context below.

**CONTEXT:**
<context>{context}</context>

**REQUIREMENTS:**
1. Start with "Hey friends! I'm Peccy 😎"
2. Include one interesting tech fact, historical insight, or industry observation
3. Preview this week's content with enthusiasm
4. Create smooth transition to main articles
5. Keep total length 50-70 words
6. Use casual, confident tone with technical expertise
7. Output plain text only - no markdown formatting

**OUTPUT:** Newsletter introduction only.""",
        Language.KO: """Write a weekly newsletter introduction in Korean using the context below.

**CONTEXT:**
<context>{context}</context>

**REQUIREMENTS:**
1. Start with "안녕 친구들! 난 Peccy야 😎"
2. Include one interesting tech fact, historical insight, or industry observation
3. Preview this week's content with enthusiasm
4. Create smooth transition to main articles
5. Keep total length 50-70 words
6. Use casual speech (반말) with confident tone
7. Keep technical terms in English when appropriate
8. Output plain text only - no markdown formatting

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


class SummarizationPrompt(BasePrompt):
    input_variables: list[str] = ["post"]
    output_variables: list[str] = ["summary", "tags", "urls"]
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
   - Be thorough and explanatory, not brief or superficial
   - Help readers understand the "why" behind technical decisions, not just the "what"
   - Target length: Maximum 20% of the original post length

3. **Educational Focus**
   - Prioritize clarity and understanding over brevity
   - Explain concepts with appropriate context and background
   - Connect technical details to practical implications
   - Make complex ideas accessible without oversimplifying

**SECTION SKIP RULE:**
- If the source material does not contain sufficient information for a section, OMIT that section entirely
- Do NOT write sections that merely state "no information is available" or "the article does not mention this"
- Only include sections where you can provide meaningful, substantive content based on the source material

**REQUIRED STRUCTURE:**

Provide your analysis within <summary> tags using the following sections. Include only those sections for which the
source material provides sufficient substantive content:

<h3>📌 Why This Matters</h3>
Explain the significance and relevance of this content. Focus on the problem being addressed, why the approach is 
noteworthy, and who should care. Write as a flowing narrative without subsections. Be concise while covering essential 
points.

<h3>🔄 Core Architecture and Workflow</h3>
Describe the system design and workflow clearly. Cover main components, their interactions, and key design choices. 
Include relevant images using: <img src="full_url" alt="descriptive text"> (use complete URLs only: 
https://example.com/image.jpg). Write as a cohesive narrative without subsection headers. Avoid redundancy with the 
technical deep dive section.

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
✅ DO acknowledge when information is limited or unclear
✅ DO stay faithful to the source material
✅ DO explain only what is actually presented
✅ DO write in a natural, flowing narrative style throughout all sections
✅ DO be concise while maintaining completeness

**OUTPUT FORMAT:**
<summary>[Your comprehensive technical explanation following the structure above]</summary>
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
   - Be thorough and explanatory, not brief or superficial
   - Help readers understand the "why" behind technical decisions, not just the "what"
   - Target length: Maximum 20% of the original post length

3. **Educational Focus**
   - Prioritize clarity and understanding over brevity
   - Explain concepts with appropriate context and background
   - Connect technical details to practical implications
   - Make complex ideas accessible without oversimplifying

**섹션 생략 규칙:**
- 원문에 해당 섹션을 채울 충분한 정보가 없으면, 그 섹션을 통째로 생략하세요
- "원문에 관련 정보가 없습니다" 또는 "언급되지 않았습니다" 같은 내용으로 섹션을 채우지 마세요
- 원문을 기반으로 실질적이고 의미 있는 내용을 작성할 수 있는 섹션만 포함하세요

**REQUIRED STRUCTURE:**

Provide your analysis within <summary> tags using the following sections. Include only those sections for which the
source material provides sufficient substantive content:

<h3>📌 왜 이 아티클에 주목해야 하나요?</h3>
Explain the significance and relevance of this content. Focus on the problem being addressed, why the approach is 
noteworthy, and who should care. Write as a flowing narrative without subsections. Be concise while covering essential 
points.

<h3>🔄 아이디어, 아키텍처, 또는 워크플로우 개요</h3>
Describe the system design and workflow clearly. Cover main components, their interactions, and key design choices. 
Include relevant images using: <img src="full_url" alt="descriptive text"> (use complete URLs only: 
https://example.com/image.jpg). Write as a cohesive narrative without subsection headers. Avoid redundancy with the 
technical deep dive section.

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
✅ DO acknowledge when information is limited or unclear
✅ DO stay faithful to the source material
✅ DO explain only what is actually presented
✅ DO write in a natural, flowing narrative style throughout all sections
✅ DO be concise while maintaining completeness
✅ DO translate technical terms to Korean when appropriate

**OUTPUT FORMAT:**
<summary>[Your comprehensive technical explanation in Korean following the structure above]</summary>
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

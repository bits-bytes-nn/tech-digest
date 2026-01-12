from abc import ABC
from typing import ClassVar
from dataclasses import dataclass

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.messages import SystemMessage

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

    @classmethod
    def get_prompt(cls, enable_prompt_cache: bool = False) -> ChatPromptTemplate:
        system_template = cls.system_prompt_template
        human_template = cls.human_prompt_template
        instance = cls(
            input_variables=cls.input_variables,
            output_variables=cls.output_variables,
            system_prompt_template=system_template,
            human_prompt_template=human_template,
        )

        if enable_prompt_cache:
            system_msg = SystemMessage(
                content=[
                    {
                        "type": "text",
                        "text": instance.system_prompt_template,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            )
            human_msg = HumanMessagePromptTemplate.from_template(
                instance.human_prompt_template
            )
            return ChatPromptTemplate.from_messages([system_msg, human_msg])

        return ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    instance.system_prompt_template
                ),
                HumanMessagePromptTemplate.from_template(
                    instance.human_prompt_template
                ),
            ]
        )


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
- Excluded topics listed above
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
- Included topics from the list above as PRIMARY focus (HIGHEST PRIORITY - automatic acceptance)
- Core ML focus (theory, algorithms, model architectures, research methods)
- Novel research contributions or theoretical insights
- Deep technical ML understanding
- **High-quality practical guides with deep ML insights and proven best practices**

---

**STEP 2: QUALITY SCORING (0.00-1.00, use TWO decimal places)**

**CRITICAL RULE: INCLUDED TOPICS - TIERED SCORING**

If content SUBSTANTIALLY covers ANY included topic (>30% as primary focus):
- **BASE SCORE: 0.75-0.85** (determined by quality)
- Apply quality modifiers below
- Skip standard penalties

**INCLUDED TOPIC QUALITY TIERS:**
- **0.82-0.85:** Exceptional coverage with novel insights, comprehensive analysis, and expert-level depth
- **0.78-0.81:** Strong coverage with good insights and solid technical depth
- **0.75-0.77:** Adequate coverage meeting substantial threshold but with basic depth

**INCLUDED TOPIC MODIFIERS:**
- +0.03: Multiple included topics covered substantially (each >30%)
- +0.02: Exceptional clarity and organization
- +0.02: Novel or unique perspective on the included topic
- -0.02: Coverage is just above 30% threshold (borderline substantial)
- -0.03: Some promotional content mixed in (<10% of content)

**IMPORTANT:** Simple mentions or brief examples of included topics DO NOT qualify for included topic scoring.

---

**SCORING PATH: OTHER ML CONTENT (Non-included topics)**

**Base Score Ranges (select specific score within range):**

**0.75-0.85 (Exceptional):**
- Novel theoretical ML contributions with rigorous proofs
- Groundbreaking mathematical/algorithmic innovations
- Exceptional empirical evaluation with comprehensive statistical analysis
- High-quality practical guides with deep ML insights and battle-tested best practices
- **Within tier:** 0.82-0.85 (groundbreaking), 0.78-0.81 (excellent), 0.75-0.77 (very good)

**0.60-0.70 (Strong):**
- Solid empirical studies with clear novelty
- Well-validated incremental improvements
- Rigorous comparative studies with insights
- Well-structured practical guides with valuable ML best practices
- **Within tier:** 0.67-0.70 (strong novelty), 0.63-0.66 (good execution), 0.60-0.62 (solid work)

**0.45-0.55 (Moderate):**
- Limited novelty but decent execution
- Educational content with some new perspectives
- Implementation-focused with research context
- **Within tier:** 0.52-0.55 (good educational value), 0.48-0.51 (acceptable), 0.45-0.47 (basic)

**0.30-0.40 (Weak):**
- Mostly implementation guides with minimal insights
- Vendor-specific tutorials with some ML content
- Superficial ML coverage
- **Within tier:** 0.37-0.40 (some value), 0.33-0.36 (limited value), 0.30-0.32 (minimal value)

**0.00-0.25 (Reject/Very Poor):**
- Pure implementation without research value (0.15-0.25)
- Promotional/marketing content (0.05-0.15)
- Non-ML content (0.00-0.05)

---

**FINE-TUNING MODIFIERS (apply after selecting base score):**

**Positive Adjustments (+0.02 to +0.05 each):**
- +0.05: Exceptional clarity and reproducibility
- +0.03: Novel insights or unique perspective
- +0.03: Comprehensive empirical validation
- +0.02: Well-structured with clear takeaways
- +0.02: Addresses important real-world problem

**Negative Adjustments (-0.02 to -0.05 each):**
- -0.05: Significant non-ML content (10-20%)
- -0.04: Promotional elements (5-10% of content)
- -0.03: Implementation-heavy without insights
- -0.02: Limited scope or depth
- -0.02: Unclear or poorly organized

**Uncertainty Handling:**
- When uncertain between two scores: choose the lower one
- When borderline between tiers: subtract 0.03 from base score
- Maximum total adjustments: ±0.10 from base score

---

**SCORING CALCULATION EXAMPLE:**

Example 1 (Included Topic):
- Base: 0.80 (strong coverage of "reinforcement learning" - included topic)
- +0.02 (exceptional clarity)
- +0.02 (novel perspective)
- **Final: 0.84**

Example 2 (Other ML Content):
- Base: 0.65 (solid empirical study, mid-tier in "Strong" range)
- +0.03 (comprehensive validation)
- -0.02 (limited scope)
- **Final: 0.66**

Example 3 (Borderline):
- Base: 0.48 (educational content, mid-tier in "Moderate" range)
- -0.03 (borderline between tiers)
- -0.02 (promotional elements)
- **Final: 0.43**

---

**OUTPUT FORMAT:**
<title>[Original title in proper title case]</title>
<reason>[Explain: (1) Topic validation - Does content SUBSTANTIALLY cover (>30% as primary focus) any included topic?
If yes, state which included topic and coverage quality (exceptional/strong/adequate). (2) Base score selection and
tier reasoning (3) Applied modifiers with justification (4) Final score calculation]</reason>
<score>[Number between 0.00 and 1.00, TWO decimal places, e.g., 0.67, 0.82, 0.43]</score>""",
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
- Excluded topics listed above
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
- Included topics from the list above as PRIMARY focus with AWS context (HIGHEST PRIORITY)
- Core ML focus with Amazon/AWS context (80%+ of content)
- Technical depth in AWS ML services (SageMaker, Bedrock, etc.)
- Advanced ML understanding within AWS ecosystem
- **High-quality AWS ML practical guides with deep insights and proven best practices**

---

**STEP 2: QUALITY SCORING (0.00-1.00, use TWO decimal places)**

**CRITICAL RULE: INCLUDED TOPICS - TIERED SCORING**

If content SUBSTANTIALLY covers ANY included topic with Amazon/AWS context (>30% as primary focus):
- **BASE SCORE: 0.75-0.85** (determined by quality)
- Apply quality modifiers below
- Skip standard penalties

**INCLUDED TOPIC QUALITY TIERS (with AWS context):**
- **0.82-0.85:** Exceptional AWS coverage with novel insights, comprehensive analysis, and expert-level depth
- **0.78-0.81:** Strong AWS coverage with good insights and solid technical depth
- **0.75-0.77:** Adequate AWS coverage meeting substantial threshold but with basic depth

**INCLUDED TOPIC MODIFIERS:**
- +0.03: Multiple included topics covered substantially in AWS context (each >30%)
- +0.02: Exceptional clarity and organization
- +0.02: Novel or unique AWS implementation perspective
- -0.02: Coverage is just above 30% threshold (borderline substantial)
- -0.03: Some promotional AWS content mixed in (<10% of content)

**IMPORTANT:** Simple mentions or brief examples of included topics DO NOT qualify for included topic scoring.

---

**SCORING PATH: OTHER AWS ML CONTENT (Non-included topics)**

**Base Score Ranges (select specific score within range):**

**0.75-0.85 (Exceptional):**
- Novel ML contributions in AWS/Amazon context
- Groundbreaking AWS ML architectural innovations
- Exceptional mathematical/algorithmic innovations on AWS platform
- High-quality AWS ML practical guides with deep insights and battle-tested best practices
- **Within tier:** 0.82-0.85 (groundbreaking AWS ML), 0.78-0.81 (excellent AWS ML), 0.75-0.77 (very good AWS ML)

**0.60-0.70 (Strong):**
- Solid ML studies using Amazon/AWS services with clear value
- AWS ML implementations with strong technical depth
- Well-validated incremental improvements to AWS ML workflows
- Well-structured AWS ML practical guides with valuable best practices
- **Within tier:** 0.67-0.70 (strong AWS innovation), 0.63-0.66 (good AWS execution), 0.60-0.62 (solid AWS work)

**0.45-0.55 (Moderate):**
- Moderate technical depth in AWS ML services
- Educational AWS ML content with some insights
- Implementation-focused AWS guides with context
- **Within tier:** 0.52-0.55 (good AWS educational value), 0.48-0.51 (acceptable AWS content), 0.45-0.47 (basic AWS
tutorial)

**0.30-0.40 (Weak):**
- Basic AWS ML tutorials with minimal depth
- Superficial Amazon ML service coverage
- Limited AWS technical contribution
- **Within tier:** 0.37-0.40 (some AWS value), 0.33-0.36 (limited AWS value), 0.30-0.32 (minimal AWS value)

**0.00-0.25 (Reject/Very Poor):**
- Non-AWS vendor content (0.00-0.05)
- Marketing disguised as AWS technical content (0.05-0.15)
- Minimal AWS ML focus (0.15-0.25)

---

**FINE-TUNING MODIFIERS (apply after selecting base score):**

**Positive Adjustments (+0.02 to +0.05 each):**
- +0.05: Exceptional AWS implementation clarity and reproducibility
- +0.03: Novel AWS ML insights or unique architectural perspective
- +0.03: Comprehensive AWS empirical validation
- +0.02: Well-structured AWS guide with clear takeaways
- +0.02: Addresses important real-world AWS ML problem

**Negative Adjustments (-0.02 to -0.05 each):**
- -0.05: Significant non-ML content (10-20%)
- -0.04: Non-AWS vendor mentions (5-10% of content)
- -0.03: Implementation-heavy without AWS insights
- -0.02: Limited AWS scope or depth
- -0.02: Unclear or poorly organized AWS content

**Uncertainty Handling:**
- When uncertain between two scores: choose the lower one
- When borderline between tiers: subtract 0.03 from base score
- Maximum total adjustments: ±0.10 from base score

---

**SCORING CALCULATION EXAMPLE:**

Example 1 (Included Topic with AWS):
- Base: 0.80 (strong coverage of "SageMaker" - included topic with AWS context)
- +0.02 (exceptional clarity)
- +0.02 (novel AWS perspective)
- **Final: 0.84**

Example 2 (Other AWS ML Content):
- Base: 0.65 (solid AWS empirical study, mid-tier in "Strong" range)
- +0.03 (comprehensive AWS validation)
- -0.02 (limited AWS scope)
- **Final: 0.66**

Example 3 (Borderline AWS):
- Base: 0.48 (educational AWS content, mid-tier in "Moderate" range)
- -0.03 (borderline between tiers)
- -0.02 (promotional elements)
- **Final: 0.43**

---

**OUTPUT FORMAT:**
<title>[Original title in proper title case]</title>
<reason>[Explain: (1) Topic validation - Does content SUBSTANTIALLY cover (>30% as primary focus) any included topic
with AWS context? If yes, state which included topic and AWS coverage quality (exceptional/strong/adequate).
(2) Base score selection and tier reasoning (3) Applied modifiers with justification (4) Final score calculation]
</reason>
<score>[Number between 0.00 and 1.00, TWO decimal places, e.g., 0.67, 0.82, 0.43]</score>""",
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

**REQUIRED STRUCTURE:**

Provide your analysis within <summary> tags using this exact structure:

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

**REQUIRED STRUCTURE:**

Provide your analysis within <summary> tags using this exact structure:

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

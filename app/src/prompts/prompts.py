from abc import ABC
from typing import ClassVar
from dataclasses import dataclass

from langchain.prompts import (
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
        if self.input_variables is not None:
            for var in self.input_variables:
                if not var or not isinstance(var, str):
                    raise ValueError(f"Invalid input variable: '{var}'")
                if var == "image_data":
                    continue
                if (
                    f"{{{var}}}" not in self.human_prompt_template
                    and f"{{{var}}}" not in self.system_prompt_template
                ):
                    raise ValueError(
                        f"Input variable '{var}' not found in any prompt template"
                    )

    @classmethod
    def get_prompt(
        cls,
        enable_prompt_cache: bool = False,
    ) -> ChatPromptTemplate:
        system_template = cls.system_prompt_template
        human_template = cls.human_prompt_template

        instance = cls(
            input_variables=cls.input_variables,
            output_variables=cls.output_variables,
            system_prompt_template=system_template,
            human_prompt_template=human_template,
        )

        if enable_prompt_cache:
            messages = cls._create_cached_messages(instance)
        else:
            messages = cls._create_standard_messages(instance)
        return ChatPromptTemplate.from_messages(messages)

    @classmethod
    def _create_cached_messages(
        cls, instance: "BasePrompt"
    ) -> list[HumanMessagePromptTemplate | SystemMessagePromptTemplate]:
        return [
            SystemMessagePromptTemplate.from_template(
                template=[
                    {"type": "text", "text": instance.system_prompt_template},
                    {"cachePoint": {"type": "default"}},
                ],
                input_variables=instance.input_variables,
            ),
            HumanMessagePromptTemplate.from_template(
                template=[
                    {"type": "text", "text": instance.human_prompt_template},
                    {"cachePoint": {"type": "default"}},
                ],
                input_variables=instance.input_variables,
            ),
        ]

    @classmethod
    def _create_standard_messages(
        cls, instance: "BasePrompt"
    ) -> list[HumanMessagePromptTemplate | SystemMessagePromptTemplate]:
        return [
            SystemMessagePromptTemplate.from_template(
                template=instance.system_prompt_template,
                input_variables=instance.input_variables,
            ),
            HumanMessagePromptTemplate.from_template(
                template=instance.human_prompt_template,
                input_variables=instance.input_variables,
            ),
        ]


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

**STEP 2: QUALITY SCORING (0.0-1.0)**

**CRITICAL RULE: INCLUDED TOPICS AUTO-PASS**

If content SUBSTANTIALLY covers ANY included topic (>30% as primary focus):
- **AUTOMATIC SCORE: 0.8**
- Skip all quality evaluation
- Ignore all penalties
- Content automatically passes filtering

**IMPORTANT:** Simple mentions or brief examples of included topics DO NOT qualify for auto-pass.

**SCORING PATH A: INCLUDED TOPICS CONTENT**
- Fixed score: **0.8**
- Requires substantial coverage (>30% as primary focus)
- No further evaluation needed
- All penalties waived

**SCORING PATH B: OTHER ML CONTENT**

Start at 0.0 and evaluate (BE CONSERVATIVE):

**Score 0.7-0.8 (Excellent):**
- Truly novel theoretical ML contributions with rigorous proofs
- Groundbreaking mathematical/algorithmic innovations
- Exceptional empirical evaluation with comprehensive statistical analysis
- Paradigm-shifting findings with broad impact
- **High-quality practical guides with deep ML insights and battle-tested best practices**
- **Comprehensive guides containing essence of real-world ML experience and expertise**
- Zero promotional content, exceptional execution

**Score 0.5-0.6 (Good):**
- Solid empirical studies with clear novelty
- Well-validated incremental improvements
- Rigorous comparative studies with insights
- Strong research contribution with minor limitations
- **Well-structured practical guides with valuable ML best practices and actionable insights**

**Score 0.3-0.4 (Fair):**
- Limited novelty but decent execution
- Educational content with some new perspectives
- Implementation-focused with research context
- Acceptable quality but not exceptional

**Score 0.1-0.2 (Poor):**
- Mostly implementation guides with minimal insights
- Vendor-specific tutorials with some ML content
- Superficial ML coverage
- Weak research value

**Score 0.0 (Reject):**
- Pure implementation without research value
- Promotional/marketing content
- Non-ML content
- No meaningful contribution

**Penalties (Path B only):**
- Deduct 0.4 for >10% non-ML content
- Deduct 0.3 for implementation-only content
- Deduct 0.3 for promotional content
- **When uncertain, default to lower end of range**
- **When borderline, score 0.2 points lower**

---

**OUTPUT FORMAT:**
<title>[Original title in proper title case]</title>
<reason>[Explain: (1) Topic validation - Does content SUBSTANTIALLY cover (>30% as primary focus) any included topic? 
(2) If yes: state "AUTO-PASS via substantial coverage of [topic name]", otherwise explain quality assessment with 
conservative scoring rationale]</reason>
<score>[Number between 0.0 and 1.0]</score>""",
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

**STEP 2: QUALITY SCORING (0.0-1.0)**

**CRITICAL RULE: INCLUDED TOPICS AUTO-PASS**

If content SUBSTANTIALLY covers ANY included topic with Amazon/AWS context (>30% as primary focus):
- **AUTOMATIC SCORE: 0.8**
- Skip all quality evaluation
- Ignore all penalties
- Content automatically passes filtering

**IMPORTANT:** Simple mentions or brief examples of included topics DO NOT qualify for auto-pass.

**SCORING PATH A: INCLUDED TOPICS + AWS CONTENT**
- Fixed score: **0.8**
- Requires substantial coverage (>30% as primary focus with AWS context)
- No further evaluation needed
- All penalties waived

**SCORING PATH B: OTHER AWS ML CONTENT**

Start at 0.0 and evaluate (BE CONSERVATIVE):

**Score 0.7-0.8 (Excellent):**
- Truly novel ML contributions in AWS/Amazon context
- Groundbreaking AWS ML architectural innovations
- Exceptional mathematical/algorithmic innovations on AWS platform
- Deep technical analysis with exceptional insights on Amazon infrastructure
- **High-quality AWS ML practical guides with deep insights and battle-tested best practices**
- **Comprehensive AWS ML guides containing essence of real-world experience and expertise**

**Score 0.5-0.6 (Good):**
- Solid ML studies using Amazon/AWS services with clear value
- AWS ML implementations with strong technical depth
- Well-validated incremental improvements to AWS ML workflows
- **Well-structured AWS ML practical guides with valuable best practices and actionable insights**

**Score 0.3-0.4 (Fair):**
- Moderate technical depth in AWS ML services
- Educational AWS ML content with some insights
- Acceptable quality but not exceptional

**Score 0.1-0.2 (Poor):**
- Basic AWS ML tutorials with minimal depth
- Superficial Amazon ML service coverage
- Weak technical contribution

**Score 0.0 (Reject):**
- Non-AWS vendor content
- Marketing disguised as technical content
- No meaningful ML contribution

**Penalties (Path B only):**
- Deduct 0.4 for >20% non-ML content
- Deduct 0.3 for non-AWS vendor mentions
- Deduct 0.2 for basic tutorials without advanced insights
- **When uncertain, default to lower end of range**
- **When borderline, score 0.2 points lower**

---

**OUTPUT FORMAT:**
<title>[Original title in proper title case]</title>
<reason>[Explain: (1) Topic validation - Does content SUBSTANTIALLY cover (>30% as primary focus) any included topic 
with AWS context? (2) If yes: state "AUTO-PASS via substantial coverage of [topic name] with AWS context", otherwise 
explain AWS/Amazon ML depth assessment with conservative scoring rationale]</reason>
<score>[Number between 0.0 and 1.0]</score>""",
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
    system_prompt_template: str = """You are an expert technical writer and content analyst with deep expertise in
    software engineering, machine learning, and system architecture. You excel at creating engaging, detailed
    explanations of complex technical topics that feel like friendly conversations with fellow developers. Your writing
    style is approachable yet thorough, making complex concepts accessible while maintaining technical accuracy and
    depth."""
    _human_prompt_template: ClassVar[dict[Language, str]] = {
        Language.EN: """Create a comprehensive and engaging technical explanation of the following blog post:

**CONTENT TO ANALYZE:**
<post>{post}</post>

**WRITING APPROACH:**
Write as if you're having a friendly conversation with a fellow developer over coffee. Be thorough and explanatory
rather than brief and summarized. Take time to walk through concepts, explain the "why" behind decisions, and help
readers truly understand the material. Keep the content length to a maximum of 20% of the original post length.

**ANALYSIS REQUIREMENTS:**
1. **Conversational Depth:** Explain technical concepts in a friendly, detailed manner with context and background
2. **Practical Storytelling:** Share the journey of implementation, challenges faced, and solutions discovered
3. **Thoughtful Insights:** Provide detailed explanations of findings, methodologies, and lessons learned
4. **Strategic Context:** Thoroughly discuss technical decisions, trade-offs, and their broader implications
5. **Industry Perspective:** Connect the content to wider technical trends with detailed explanations
6. **Educational Focus:** Prioritize helping readers learn and understand over brevity

**REQUIRED STRUCTURE (within <summary> tags):**

<h3>📌 Why This Article Caught Our Attention</h3>
Tell the story of why this content matters. Explain the background context, the problems it addresses, and why it's
particularly relevant right now. Share what makes this approach interesting or unique, and help readers understand the
broader significance in the current tech landscape.

<h3>🔄 Understanding the Idea, Architecture, or Workflow</h3>
Walk readers through the core ideas, system architecture, and workflow in detail. Explain how different components
interact, why certain design choices were made, and how the overall system comes together. Include relevant images
using: <img src="full_url" alt="descriptive text">
Note: Always use complete URLs for images (e.g., https://example.com/path/image.jpg), not relative paths.
Take time to explain the reasoning behind architectural patterns and help readers understand the thought process.

<h3>🛠️ Let's Dive Deep Into the Technical Details</h3>
Provide a thorough technical walkthrough that includes:
• **Understanding the Core Ideas:** Explain the fundamental concepts in detail, providing context and background to help
readers fully grasp the approach
• **Code Walkthrough:** Present and carefully explain critical code sections, discussing what each part does and why
it's implemented that way
• **Design Philosophy:** Discuss the thinking behind technical choices, exploring alternatives that were considered and
why certain paths were chosen
• **Performance Deep Dive:** Thoroughly explain optimization strategies, scalability considerations, and performance
implications with detailed reasoning
• **Handling Edge Cases:** Discuss how the system deals with potential issues, error scenarios, and unexpected
situations
• **Implementation Wisdom:** Share detailed guidance, common pitfalls to avoid, and practical advice for anyone looking
to implement similar solutions

<h3>📊 What the Results Tell Us and Why It Matters</h3>
Present a detailed discussion of outcomes and their significance:
• Walk through performance metrics and benchmarks, explaining what they mean and why they're important
• Discuss the business value and strategic advantages in detail, connecting technical improvements to real-world benefits
• Explore cost implications and resource optimization benefits with thorough analysis
• Examine scalability potential and long-term maintenance considerations

<h3>🔮 Looking Ahead: What This Means for the Future</h3>
Provide a thoughtful exploration of future possibilities and implications:
• Discuss potential evolution paths and emerging opportunities with detailed explanations
• Explore integration possibilities with existing systems and workflows, explaining the practical implications
• Analyze industry adoption prospects and competitive landscape implications with thorough reasoning

**FORMATTING REQUIREMENTS:**
- Use HTML tags exclusively (no markdown)
- Write in a conversational, explanatory style that prioritizes understanding over brevity
- Present code in <pre><code class="highlight"> blocks for proper syntax highlighting
- Use <strong> for key technical points and <em> for important technical terms
- Format data and comparisons using HTML tables when appropriate
- Maintain consistent heading hierarchy and structure
- Ensure all technical terms are thoroughly explained with context and examples

**OUTPUT FORMAT:**
<summary>[Complete technical explanation with friendly, detailed narrative and comprehensive analysis]</summary>
<tags>[5-7 specific technical topics in Title Case, comma-separated, focusing on distinctive technologies,
methodologies, or architectural patterns mentioned in the article - avoid generic terms like "Machine Learning" or "AI"
unless they represent novel approaches - write all titles in English]</tags>
<urls>[Essential technical references as properly formatted HTML links: <a href="url1">descriptive title 1</a>,
<a href="url2">descriptive title 2</a> - write all titles in English]</urls>""",
        Language.KO: """Create a comprehensive and engaging technical explanation of the following blog post in Korean:

**CONTENT TO ANALYZE:**
<post>{post}</post>

**WRITING APPROACH:**
Write as if you're having a friendly conversation with a fellow developer over coffee. Be thorough and explanatory
rather than brief and summarized. Take time to walk through concepts, explain the "why" behind decisions, and help
readers truly understand the material. Keep the content length to a maximum of 20% of the original post length.

**ANALYSIS REQUIREMENTS:**
1. **Conversational Depth:** Explain technical concepts in a friendly, detailed manner with context and background
2. **Practical Storytelling:** Share the journey of implementation, challenges faced, and solutions discovered
3. **Thoughtful Insights:** Provide detailed explanations of findings, methodologies, and lessons learned
4. **Strategic Context:** Thoroughly discuss technical decisions, trade-offs, and their broader implications
5. **Industry Perspective:** Connect the content to wider technical trends with detailed explanations
6. **Educational Focus:** Prioritize helping readers learn and understand over brevity

**REQUIRED STRUCTURE (within <summary> tags):**

<h3>📌 왜 이 아티클에 주목해야 하나요?</h3>
Tell the story of why this content matters. Explain the background context, the problems it addresses, and why it's
particularly relevant right now. Share what makes this approach interesting or unique, and help readers understand the
broader significance in the current tech landscape.

<h3>🔄 아이디어, 아키텍처, 또는 워크플로우 개요</h3>
Walk readers through the core ideas, system architecture, and workflow in detail. Explain how different components
interact, why certain design choices were made, and how the overall system comes together. Include relevant images
using: <img src="full_url" alt="descriptive text">
Note: Always use complete URLs for images (e.g., https://example.com/path/image.jpg), not relative paths.
Take time to explain the reasoning behind architectural patterns and help readers understand the thought process.

<h3>🛠️ 기술적 심층 분석</h3>
Provide a thorough technical walkthrough that includes:
• **Understanding the Core Ideas:** Explain the fundamental concepts in detail, providing context and background to help
readers fully grasp the approach
• **Code Walkthrough:** Present and carefully explain critical code sections, discussing what each part does and why
it's implemented that way
• **Design Philosophy:** Discuss the thinking behind technical choices, exploring alternatives that were considered and
why certain paths were chosen
• **Performance Deep Dive:** Thoroughly explain optimization strategies, scalability considerations, and performance
implications with detailed reasoning
• **Handling Edge Cases:** Discuss how the system deals with potential issues, error scenarios, and unexpected
situations
• **Implementation Wisdom:** Share detailed guidance, common pitfalls to avoid, and practical advice for anyone looking
to implement similar solutions

<h3>📊 성과 및 비즈니스 임팩트</h3>
Present a detailed discussion of outcomes and their significance:
• Walk through performance metrics and benchmarks, explaining what they mean and why they're important
• Discuss the business value and strategic advantages in detail, connecting technical improvements to real-world
benefits
• Explore cost implications and resource optimization benefits with thorough analysis
• Examine scalability potential and long-term maintenance considerations

<h3>🔮 향후 발전 가능성과 기회</h3>
Provide a thoughtful exploration of future possibilities and implications:
• Discuss potential evolution paths and emerging opportunities with detailed explanations
• Explore integration possibilities with existing systems and workflows, explaining the practical implications
• Analyze industry adoption prospects and competitive landscape implications with thorough reasoning

**FORMATTING REQUIREMENTS:**
- Use HTML tags exclusively (no markdown)
- Write in a conversational, explanatory style that prioritizes understanding over brevity
- Present code in <pre><code class="highlight"> blocks for proper syntax highlighting
- Use <strong> for key technical points and <em> for important technical terms
- Format data and comparisons using HTML tables when appropriate
- Maintain consistent heading hierarchy and structure
- Ensure all technical terms are thoroughly explained with context and examples
- Write content in Korean while keeping proper nouns, and difficult-to-translate concepts in English

**OUTPUT FORMAT:**
<summary>[Complete technical explanation with friendly, detailed narrative and comprehensive analysis in Korean]
</summary>
<tags>[5-7 specific technical topics in Title Case, comma-separated, focusing on distinctive technologies,
methodologies, or architectural patterns mentioned in the article - avoid generic terms like "Machine Learning" or "AI"
unless they represent novel approaches - write all titles in English]</tags>
<urls>[Essential technical references as properly formatted HTML links: <a href="url1">descriptive title 1</a>,
<a href="url2">descriptive title 2</a> - write all titles in English]</urls>""",
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

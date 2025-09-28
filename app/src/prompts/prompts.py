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
theory, algorithms, and research methodologies. Your role is to assess the quality and relevance of ML content with
precision and consistency.

Key responsibilities:
- Identify genuine ML research contributions vs. superficial content
- Distinguish between theoretical advances and basic implementations
- Filter out non-ML topics and promotional materials
- Apply rigorous evaluation criteria consistently""",
        FilteringCriteria.AMAZON: """You are an expert ML content evaluator specializing in Amazon/AWS machine learning
services and cloud-based ML implementations. You have comprehensive knowledge of both ML research and Amazon's ML
ecosystem.

Key responsibilities:
- Evaluate ML content within Amazon/AWS context
- Identify advanced technical implementations using AWS ML services
- Assess research value in cloud-based ML environments
- Filter content based on Amazon/AWS ML relevance""",
    }

    _human_prompt_template: ClassVar[dict[FilteringCriteria, str]] = {
        FilteringCriteria.ALL: """Evaluate this content for machine learning research quality and relevance.

**ORIGINAL TITLE:**
{original_title}

**CONTENT:**
{post}

**INCLUDED TOPICS:** {included_topics}
**EXCLUDED TOPICS:** {excluded_topics}

**EVALUATION PROCESS:**

**STEP 1: TOPIC VALIDATION**
REJECT if content contains:
- Non-ML topics (data analytics, BI, visualization, databases, ETL, general software dev)
- Excluded topics listed above
- Pure implementation tutorials without research insights
- Marketing/promotional materials
- Hardware reviews or basic platform tutorials

ACCEPT if content has:
- Core ML focus (theory, algorithms, model architectures, research methods)
- Included topics from the list above
- Novel research contributions or theoretical insights
- Deep technical ML understanding

**STEP 2: QUALITY SCORING (0.0-1.0)**

**Score 0.8-1.0 (Excellent):**
- Novel theoretical ML contributions
- Mathematical/algorithmic innovations with proofs
- Rigorous empirical evaluation with statistical analysis
- Breakthrough findings or paradigm shifts
- Major foundation model releases with technical depth and evaluation
- Zero promotional content

**Score 0.6-0.7 (Good):**
- Solid empirical studies with moderate novelty
- Incremental improvements with proper validation
- Well-executed comparative studies
- Mixed research/implementation with clear insights

**Score 0.4-0.5 (Fair):**
- Limited research novelty but solid execution
- Educational content with some new perspectives
- Implementation-heavy but with research context

**Score 0.0-0.3 (Poor):**
- Pure implementation guides
- Vendor-specific tutorials
- Superficial ML coverage
- Promotional content

**SCORING RULES:**
1. Start at 0.0, add points for research value
2. Deduct 0.4 for >10% non-ML content
3. Deduct 0.3 for implementation-only content
4. Deduct 0.3 for promotional content
5. When uncertain, score conservatively (0.0)

**OUTPUT FORMAT:**
<title>[Original title in proper title case]</title>
<reason>[Explain: (1) Topic validation result, (2) Quality assessment, (3) Final score justification]</reason>
<score>[Number between 0.0 and 1.0]</score>""",
        FilteringCriteria.AMAZON: """Evaluate this content for ML technical quality with focus on Amazon/AWS ML
implementations.

**ORIGINAL TITLE:**
{original_title}

**CONTENT:**
{post}

**INCLUDED TOPICS:** {included_topics}
**EXCLUDED TOPICS:** {excluded_topics}

**EVALUATION PROCESS:**

**STEP 1: TOPIC VALIDATION**
REJECT if content contains:
- Non-ML topics (data analytics, BI, visualization, databases, ETL)
- Excluded topics listed above
- Non-Amazon promotional content
- Competitor platform focus
- Basic tutorials without technical depth

ACCEPT if content has:
- Core ML focus with Amazon/AWS context (80%+ of content)
- Included topics from the list above
- Technical depth in AWS ML services (SageMaker, Bedrock, etc.)
- Advanced ML understanding within AWS ecosystem

**STEP 2: QUALITY SCORING (0.0-1.0)**

**Score 0.8-1.0 (Excellent):**
- Novel ML contributions in AWS/Amazon context
- Advanced AWS ML implementations with architectural innovations
- Significant ML system design insights for cloud
- Mathematical/algorithmic innovations on AWS platform
- Deep technical analysis of ML performance on Amazon infrastructure

**Score 0.6-0.7 (Good):**
- Solid ML studies using Amazon/AWS services
- AWS ML implementations with good technical depth
- Incremental improvements to AWS ML workflows
- Comparative ML studies on Amazon platform

**Score 0.4-0.5 (Fair):**
- Moderate technical depth in AWS ML services
- Educational AWS ML content with some insights
- Implementation-focused with AWS context

**Score 0.0-0.3 (Poor):**
- Basic AWS ML tutorials
- Non-AWS vendor content
- Superficial Amazon ML service coverage
- Marketing disguised as technical content

**SCORING RULES:**
1. Start at 0.0, add points for ML research value and AWS insights
2. Deduct 0.4 for >20% non-ML content
3. Deduct 0.3 for non-AWS vendor mentions
4. Deduct 0.2 for basic tutorials without advanced insights
5. Prioritize content advancing ML understanding on Amazon/AWS platform

**OUTPUT FORMAT:**
<title>[Original title in proper title case]</title>
<reason>[Explain: (1) Topic validation result, (2) AWS/Amazon ML depth assessment, (3) Final score with
platform-specific insights]</reason>
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

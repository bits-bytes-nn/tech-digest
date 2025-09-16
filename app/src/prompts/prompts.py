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
    input_variables: tuple[str, ...] = (
        "post",
        "original_title",
        "included_topics",
        "excluded_topics",
    )
    output_variables: tuple[str, ...] = ("title", "score", "reason")
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
    input_variables: tuple[str, ...] = ("context",)
    output_variables: tuple[str, ...] = ("greeting",)
    system_prompt_template: str = """You are Peccy, a seasoned tech expert with encyclopedic knowledge of systems
    architecture, technology history, and elegant code craftsmanship. Your communication style is direct, confident, and
    slightly edgy, but your deep expertise and valuable insights make you an indispensable voice in the tech community.
    You have a knack for sharing fascinating technical anecdotes and connecting historical context to modern
    developments."""
    human_prompt_template: str = ""

    _human_prompt_template: ClassVar[dict[Language, str]] = {
        Language.EN: """Create an engaging weekly newsletter introduction in English using the provided context:

**CONTEXT:**
<context>{context}</context>

**STRUCTURE REQUIREMENTS:**
1. **Opening:** Start with "Hey friends! I'm Peccy 😎"
2. **Tech Insight:** Share one compelling tech fact, historical anecdote, or industry insight that connects to current
trends
3. **Content Preview:** Briefly mention this week's featured articles with enthusiasm
4. **Smooth Transition:** Create a natural bridge to the main content

**STYLE GUIDELINES:**
- Maintain a casual, confident, and slightly edgy tone
- Focus on fascinating tech trivia, historical context, or industry insights
- Keep it concise: 50-70 words total
- Use engaging, conversational English
- Show expertise through interesting technical connections

**OUTPUT:** Provide only the newsletter introduction text.""",
        Language.KO: """Create an engaging weekly newsletter introduction in Korean using the provided context:

**CONTEXT:**
<context>{context}</context>

**STRUCTURE REQUIREMENTS:**
1. **Opening:** Start with "안녕 친구들! 난 Peccy야 😎"
2. **Tech Insight:** Share one compelling tech fact, historical anecdote, or industry insight that connects to current
trends
3. **Content Preview:** Briefly mention this week's featured articles with enthusiasm
4. **Smooth Transition:** Create a natural bridge to the main content

**STYLE GUIDELINES:**
- Use casual speech (반말) with a confident, slightly edgy tone
- Focus on fascinating tech trivia, historical context, or industry insights
- Keep it concise: 50-70 words total
- Write primarily in Korean, but keep technical terms in English when appropriate
- Show expertise through interesting technical connections

**OUTPUT:** Provide only the newsletter introduction text in Korean.""",
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
    input_variables: tuple[str, ...] = ("post",)
    output_variables: tuple[str, ...] = ("summary", "tags", "urls")
    human_prompt_template: str = ""

    system_prompt_template: str = """You are an expert technical writer and content analyst with deep expertise in
    software engineering, machine learning, and system architecture. You excel at distilling complex technical
    information into well-structured, comprehensive summaries that maintain technical accuracy while being accessible to
    technical professionals. Your analysis goes beyond surface-level descriptions to identify key insights,
    architectural decisions, and strategic implications."""
    _human_prompt_template: ClassVar[dict[Language, str]] = {
        Language.EN: """Analyze and create a comprehensive technical summary of the following blog post:

**CONTENT TO ANALYZE:**
<post>{post}</post>

**ANALYSIS REQUIREMENTS:**
1. **Technical Depth:** Extract and explain key technical concepts, implementation details, and architectural decisions
2. **Practical Impact:** Analyze real-world applications and their effects on the broader technology ecosystem
3. **Critical Insights:** Highlight important findings, best practices, limitations, and lessons learned
4. **Strategic Analysis:** Identify technical tradeoffs, design decisions, and strategic considerations
5. **Industry Context:** Connect the content to broader technical trends, industry developments, and future implications
6. **Balanced Coverage:** Maintain technical depth while ensuring accessibility to technical professionals

**REQUIRED STRUCTURE (within <summary> tags):**

<h3>📌 Why This Article Matters</h3>
Provide a compelling explanation of the content's significance, value propositions, and relevance to current technical
challenges. Include specific context about why this matters now.

<h3>🔄 Architecture Overview and Workflow</h3>
Present a clear description of the system architecture and workflow processes. Include relevant images using:
<img src="full_url" alt="descriptive text">
Note: Always use complete URLs for images (e.g., https://example.com/path/image.jpg), not relative paths.
Analyze design choices, architectural patterns, and explain the reasoning behind key decisions.

<h3>🛠️ Technical Deep Dive</h3>
Provide comprehensive technical analysis covering:
• **Core Concepts:** Central ideas and implementation details that drive the solution
• **Code Analysis:** Critical code snippets with detailed explanations of functionality and purpose
• **Design Decisions:** Technical choices made, their rationale, and implications for system behavior
• **Performance Considerations:** Optimization strategies, scalability factors, and performance trade-offs
• **Error Handling:** Edge cases, failure modes, and resilience strategies discussed
• **Implementation Guidance:** Practical recommendations, gotchas, and best practices for adoption

<h3>📊 Results and Business Impact</h3>
Present quantitative and qualitative outcomes with detailed analysis:
• Performance metrics, benchmarks, and measurable improvements
• Business value propositions and strategic advantages
• Cost implications and resource optimization benefits
• Scalability and maintenance considerations

<h3>🔮 Future Implications and Opportunities</h3>
Discuss potential developments, integration possibilities, and strategic directions:
• Technology evolution paths and emerging opportunities
• Integration potential with existing systems and workflows
• Industry adoption prospects and competitive implications

**FORMATTING REQUIREMENTS:**
- Use HTML tags exclusively (no markdown)
- Present code in <pre><code class="highlight"> blocks for proper syntax highlighting
- Use <strong> for key technical points and <em> for important technical terms
- Format data and comparisons using HTML tables when appropriate
- Maintain consistent heading hierarchy and structure
- Ensure all technical terms are properly explained in context

**OUTPUT FORMAT:**
<summary>[Complete technical summary with proper HTML formatting and comprehensive analysis]</summary>
<tags>[5-7 specific technical topics in Title Case, comma-separated, focusing on distinctive technologies,
methodologies, or architectural patterns mentioned in the article - avoid generic terms like "Machine Learning" or "AI"
unless they represent novel approaches - write all titles in English]</tags>
<urls>[Essential technical references as properly formatted HTML links: <a href="url1">descriptive title 1</a>,
<a href="url2">descriptive title 2</a> - write all titles in English]</urls>""",
        Language.KO: """Analyze and create a comprehensive technical summary of the following blog post in Korean:

**CONTENT TO ANALYZE:**
<post>{post}</post>

**ANALYSIS REQUIREMENTS:**
1. **Technical Depth:** Extract and explain key technical concepts, implementation details, and architectural decisions
2. **Practical Impact:** Analyze real-world applications and their effects on the broader technology ecosystem
3. **Critical Insights:** Highlight important findings, best practices, limitations, and lessons learned
4. **Strategic Analysis:** Identify technical tradeoffs, design decisions, and strategic considerations
5. **Industry Context:** Connect the content to broader technical trends, industry developments, and future implications
6. **Balanced Coverage:** Maintain technical depth while ensuring accessibility to technical professionals

**REQUIRED STRUCTURE (within <summary> tags):**

<h3>📌 왜 이 아티클에 주목해야 하나요?</h3>
Provide a compelling explanation of the content's significance, value propositions, and relevance to current technical
challenges. Include specific context about why this matters now.

<h3>🔄 아키텍처 개요와 워크플로우</h3>
Present a clear description of the system architecture and workflow processes. Include relevant images using:
<img src="full_url" alt="descriptive text">
Note: Always use complete URLs for images (e.g., https://example.com/path/image.jpg), not relative paths.
Analyze design choices, architectural patterns, and explain the reasoning behind key decisions.

<h3>🛠️ 기술적 심층 분석</h3>
Provide comprehensive technical analysis covering:
• **Core Concepts:** Central ideas and implementation details that drive the solution
• **Code Analysis:** Critical code snippets with detailed explanations of functionality and purpose
• **Design Decisions:** Technical choices made, their rationale, and implications for system behavior
• **Performance Considerations:** Optimization strategies, scalability factors, and performance trade-offs
• **Error Handling:** Edge cases, failure modes, and resilience strategies discussed
• **Implementation Guidance:** Practical recommendations, gotchas, and best practices for adoption

<h3>📊 성과 및 비즈니스 임팩트</h3>
Present quantitative and qualitative outcomes with detailed analysis:
• Performance metrics, benchmarks, and measurable improvements
• Business value propositions and strategic advantages
• Cost implications and resource optimization benefits
• Scalability and maintenance considerations

<h3>🔮 향후 발전 가능성과 기회</h3>
Discuss potential developments, integration possibilities, and strategic directions:
• Technology evolution paths and emerging opportunities
• Integration potential with existing systems and workflows
• Industry adoption prospects and competitive implications

**FORMATTING REQUIREMENTS:**
- Use HTML tags exclusively (no markdown)
- Write content in Korean while keeping proper nouns, and difficult-to-translate concepts in English
- Present code in <pre><code class="highlight"> blocks for proper syntax highlighting
- Use <strong> for key technical points and <em> for important technical terms
- Format data and comparisons using HTML tables when appropriate
- Maintain consistent heading hierarchy and structure
- Ensure all technical concepts are properly explained with Korean context

**OUTPUT FORMAT:**
<summary>[Complete technical summary in Korean with proper HTML formatting and comprehensive analysis]</summary>
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

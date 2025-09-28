from enum import Enum, auto


class AutoNamedEnum(str, Enum):
    @staticmethod
    def _generate_next_value_(
        name: str, start: int, count: int, last_values: list[str]
    ) -> str:
        return name.lower()


class EnvVars(str, Enum):
    AWS_PROFILE_NAME = "AWS_PROFILE_NAME"
    BEDROCK_REGION_NAME = "BEDROCK_REGION_NAME"
    CONFIG_FILE_SUFFIX = "CONFIG_FILE_SUFFIX"
    DEFAULT_REGION_NAME = "DEFAULT_REGION_NAME"
    LANGCHAIN_API_KEY = "LANGCHAIN_API_KEY"
    LANGCHAIN_TRACING_V2 = "LANGCHAIN_TRACING_V2"
    LANGCHAIN_ENDPOINT = "LANGCHAIN_ENDPOINT"
    LANGCHAIN_PROJECT = "LANGCHAIN_PROJECT"
    LOG_LEVEL = "LOG_LEVEL"
    TOPIC_ARN = "TOPIC_ARN"


class FilteringCriteria(AutoNamedEnum):
    ALL = auto()
    AMAZON = auto()


class Language(AutoNamedEnum):
    EN = auto()
    KO = auto()


class LanguageModelId(str, Enum):
    CLAUDE_V3_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
    CLAUDE_V3_SONNET = "anthropic.claude-3-sonnet-20240229-v1:0"
    CLAUDE_V3_OPUS = "anthropic.claude-3-opus-20240229-v1:0"
    CLAUDE_V3_5_HAIKU = "anthropic.claude-3-5-haiku-20241022-v1:0"
    CLAUDE_V3_5_SONNET = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    CLAUDE_V3_5_SONNET_V2 = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    CLAUDE_V3_7_SONNET = "anthropic.claude-3-7-sonnet-20250219-v1:0"
    CLAUDE_V4_SONNET = "anthropic.claude-sonnet-4-20250514-v1:0"
    CLAUDE_V4_OPUS = "anthropic.claude-opus-4-20250514-v1:0"
    CLAUDE_V4_1_OPUS = "anthropic.claude-opus-4-1-20250805-v1:0"


class LocalPaths(str, Enum):
    ARTICLES_DIR = "articles"
    ASSETS_DIR = "assets"
    INPUTS_DIR = "inputs"
    LOGS_DIR = "logs"
    OUTPUTS_DIR = "outputs"
    TEMPLATES_DIR = "templates"
    ARTICLE_FILE = "article.html"
    GREETING_FILE = "greeting.txt"
    LOGS_FILE = "logs.txt"
    RECIPIENTS_FILE = "recipients.txt"
    TEMPLATE_FILE = "template.html"


class SSMParams(str, Enum):
    BATCH_JOB_DEFINITION = "batch-job-definition"
    BATCH_JOB_QUEUE = "batch-job-queue"
    LANGCHAIN_API_KEY = "langchain-api-key"


class S3Paths(AutoNamedEnum):
    ARTICLES = auto()
    NEWSLETTERS = auto()
    RECIPIENTS = auto()


class AppConstants:
    NULL_STRING: str = "null"

    class External(str, Enum):
        ANTHROPIC_ENGINEERING = "www.anthropic.com/engineering"
        GOOGLE_RESEARCH = "research.google/blog"
        LINKEDIN_ENGINEERING = "www.linkedin.com/blog/engineering"
        META_AI = "ai.meta.com/blog"
        MEDIUM_DOMAIN = "medium.com"
        QWEN = "qwenlm.github.io/blog"
        XAI = "x.ai/news"

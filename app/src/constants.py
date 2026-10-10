from enum import Enum, auto

# Fallback project name for the two places that need one before config is in
# scope: the alarm subject line and the Bedrock application-inference-profile
# name. Both used to spell the literal "tech-digest" themselves, which is two
# copies of a name that also lives in config and in EnvVars.PROJECT_NAME.
DEFAULT_PROJECT_NAME: str = "tech-digest"

# Tag policy for an article, applied in two places: the feed parser (tags carried
# from an RSS entry) and the summarizer (tags the model emits). Both capped and
# defaulted identically, so the numbers live here rather than being restated as a
# ClassVar on each of the two config classes.
DEFAULT_TAGS: tuple[str, ...] = ("uncategorized",)
MAX_TAGS: int = 5


class AutoNamedEnum(str, Enum):
    @staticmethod
    def _generate_next_value_(
        name: str, start: int, count: int, last_values: list[str]
    ) -> str:
        return name.lower()


# Batch job lifetime limits. deploy_infra configures them on the queue and the
# job definition; run_batch's waiter derives how long to wait from them.
# The longest run seen is under 7 minutes; 90 per attempt leaves room for a week
# of heavy Bedrock throttling while a hung attempt still alerts the same morning.
BATCH_ATTEMPT_TIMEOUT_MINUTES: int = 90
# Two attempts = one retry, for infrastructure deaths (Spot reclaim, OOM).
BATCH_ATTEMPTS: int = 2
# A job that cannot be placed is cancelled after this long in RUNNABLE.
BATCH_RUNNABLE_LIMIT_MINUTES: int = 120


class EnvVars(str, Enum):
    AWS_PROFILE_NAME = "AWS_PROFILE_NAME"
    CONFIG_FILE_SUFFIX = "CONFIG_FILE_SUFFIX"
    LANGCHAIN_API_KEY = "LANGCHAIN_API_KEY"
    LANGCHAIN_TRACING_V2 = "LANGCHAIN_TRACING_V2"
    LANGCHAIN_ENDPOINT = "LANGCHAIN_ENDPOINT"
    LANGCHAIN_PROJECT = "LANGCHAIN_PROJECT"
    LOG_LEVEL = "LOG_LEVEL"
    # "false" skips the dated log file outside AWS. The test suite sets it, so a
    # test run no longer appends fake entries to the same file as real local runs.
    LOG_TO_FILE = "LOG_TO_FILE"
    # Identify the deployment for Bedrock cost attribution: they name this
    # project/stage's application inference profiles, which are what carry the
    # cost-allocation tags (see model_factory.application_profile_name).
    PROJECT_NAME = "PROJECT_NAME"
    STAGE = "STAGE"
    TOPIC_ARN = "TOPIC_ARN"


class FilteringCriteria(AutoNamedEnum):
    ALL = auto()
    AMAZON = auto()


class Language(AutoNamedEnum):
    EN = auto()
    KO = auto()


class LanguageModelId(str, Enum):
    CLAUDE_V3_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
    CLAUDE_V3_5_HAIKU = "anthropic.claude-3-5-haiku-20241022-v1:0"
    CLAUDE_V4_5_HAIKU = "anthropic.claude-haiku-4-5-20251001-v1:0"
    CLAUDE_V5_5_HAIKU = "anthropic.claude-haiku-5-5"
    CLAUDE_V3_5_SONNET = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    CLAUDE_V3_5_SONNET_V2 = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    CLAUDE_V3_7_SONNET = "anthropic.claude-3-7-sonnet-20250219-v1:0"
    CLAUDE_V4_SONNET = "anthropic.claude-sonnet-4-20250514-v1:0"
    CLAUDE_V4_5_SONNET = "anthropic.claude-sonnet-4-5-20250929-v1:0"
    CLAUDE_V4_6_SONNET = "anthropic.claude-sonnet-4-6"
    CLAUDE_V5_SONNET = "anthropic.claude-sonnet-5"
    CLAUDE_V5_5_SONNET = "anthropic.claude-sonnet-5-5"
    CLAUDE_V4_OPUS = "anthropic.claude-opus-4-20250514-v1:0"
    CLAUDE_V4_1_OPUS = "anthropic.claude-opus-4-1-20250805-v1:0"
    CLAUDE_V4_5_OPUS = "anthropic.claude-opus-4-5-20251101-v1:0"
    CLAUDE_V4_6_OPUS = "anthropic.claude-opus-4-6-v1"
    CLAUDE_V4_7_OPUS = "anthropic.claude-opus-4-7"
    CLAUDE_V4_8_OPUS = "anthropic.claude-opus-4-8"
    CLAUDE_V5_OPUS = "anthropic.claude-opus-5"
    CLAUDE_V5_5_OPUS = "anthropic.claude-opus-5-5"
    CLAUDE_V5_1_FABLE = "anthropic.claude-fable-5-1"
    # OpenAI GPT-6 generation on Bedrock (Converse). Astra is the top tier, Sol the
    # general-purpose tier, Luna the low-cost tier for repeatable work at scale.
    GPT_6_1_SOL = "openai.gpt-6.1-sol"
    GPT_6_SOL = "openai.gpt-6-sol"
    GPT_6_LUNA = "openai.gpt-6-luna"
    GPT_6_ASTRA = "openai.gpt-6-astra"
    # NOTE: add new models here


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
    # Per-issue record of which recipients have already received it. Read and
    # written by the send loop so a retried run cannot deliver the same issue
    # twice — see main._delivery_ledger_key.
    DELIVERIES = auto()
    NEWSLETTERS = auto()
    RECIPIENTS = auto()


class AppConstants:
    NULL_STRING: str = "null"

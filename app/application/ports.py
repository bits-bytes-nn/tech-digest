"""What the application needs from the outside world, as interfaces.

The services here depend on these protocols; the adapters in ``app.adapters``
implement them, and ``app.main`` wires one to the other.
"""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from langchain_core.runnables import Runnable

from app.domain.constants import LanguageModelId
from app.domain.crawl import CrawlReport
from app.domain.newsletter import BuildConfiguration
from app.domain.post import Post


class PostSource(Protocol):
    """Every configured source, crawled once per run."""

    #: Per-source health of the last ``collect_posts`` call.
    report: CrawlReport

    def collect_posts(self, start_date: datetime, end_date: datetime) -> list[Post]:
        """The posts published in ``[start_date, end_date)``, deduplicated."""


class ObjectStore(Protocol):
    """The run's bucket: recipient lists, the delivery ledger and the archive.

    Every method reports failure as a falsy return and logs why, so a caller
    decides what a missing object means.
    """

    def download(self, key: str, local_path: Path) -> bool:
        """Copy the object at ``key`` to ``local_path``."""

    def read_json(self, key: str) -> dict[str, Any] | None:
        """The JSON object at ``key``, or None if it is absent or unreadable."""

    def write_json(self, key: str, payload: dict[str, Any]) -> bool:
        """Write ``payload`` to ``key``, replacing what is there."""

    def upload(self, local_path: Path, prefix: str = "") -> bool:
        """Store ``local_path`` under ``prefix``, keeping its file name."""


class Mailer(Protocol):
    """Sends the issue."""

    def send(
        self, subject: str, sender: str, recipients: list[str], html_content: str
    ) -> bool:
        """Send one HTML message; False if it was not accepted."""

    def suppressed_destinations(self) -> set[str]:
        """Lowercased addresses the mail service will silently drop."""


class AlarmPublisher(Protocol):
    """Where an operator alarm goes."""

    def publish(self, subject: str, message: str) -> None:
        """Publish one alarm; may raise, so callers decide whether that matters."""


class NewsletterBuilderPort(Protocol):
    """Renders the issue from the saved posts."""

    def build(self, config: BuildConfiguration) -> tuple[Path, list[str]]:
        """The newsletter file and the file names of the standalone articles."""


class ModelCapabilities(Protocol):
    """The facts about a model that decide how a stage prompts it."""

    @property
    def supports_prompt_caching(self) -> bool:
        """Whether a cache point may be placed in the prompt."""

    @property
    def supports_thinking(self) -> bool:
        """Whether the model can be asked to think before answering."""

    @property
    def supports_sampling_params(self) -> bool:
        """Whether the model accepts temperature, so a score can be deterministic."""


class ChatModelProvider(Protocol):
    """Builds the chat model each pipeline stage calls."""

    def get_model(
        self,
        model_id: LanguageModelId,
        *,
        temperature: float = ...,
        max_tokens: int | None = None,
        enable_thinking: bool = False,
        thinking_budget_tokens: int | None = None,
        effort: str | None = None,
        stage: str | None = None,
    ) -> Runnable:
        """A model configured for one stage; ``stage`` labels its token usage."""

    def get_model_info(self, model_id: LanguageModelId) -> ModelCapabilities | None:
        """What ``model_id`` supports, or None if the provider does not offer it."""

    def is_converse(self, model: Any) -> bool:
        """Whether ``model`` runs on the Converse API, whose prompt-cache marker
        differs from the native request body's."""


#: The pixel width of a mail-displayable image at a URL (0 if unknown), or None
#: if mail clients would not display it.
ImageProbe = Callable[[str], int | None]

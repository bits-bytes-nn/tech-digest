"""What the application needs from the outside world, as interfaces.

The services here depend on these protocols; the adapters in ``app.adapters``
implement them, and ``app.main`` wires one to the other.
"""

from collections.abc import Callable
from typing import Any, Protocol

from langchain_core.runnables import Runnable

from app.domain.constants import LanguageModelId


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


#: Whether an image URL serves a type every mail client displays.
ImageProbe = Callable[[str], bool]

import time
from collections.abc import Callable
from typing import Any, ClassVar

import boto3
from bs4 import BeautifulSoup
from langchain_aws import ChatBedrockConverse
from langchain_classic.output_parsers import OutputFixingParser
from langchain_core.runnables import Runnable
from pydantic import Field

from app.adapters.bedrock import BedrockLanguageModelFactory
from app.adapters.http import is_email_renderable_image
from app.domain.constants import Language, LanguageModelId
from app.domain.content import content_html, visible_text
from app.domain.eval_metrics import is_anchor
from app.domain.post import Post
from app.domain.quality_metrics import KO_LENGTH_BAND, is_korean
from app.domain.summary import (
    FILTER_FAILED,
    FILTER_UNPARSEABLE,
    SUMMARY_FAILED,
    SUMMARY_INVALID,
    SummarizationOptions,
    SummaryOutput,
    _encode_image_srcs,
    _image_candidates,
    _postprocess_summary,
    _strip_unsourced_images,
    _unsourced_urls,
    gate_by_content_length,
    select_for_digest,
)
from app.logger import logger

from .prompts.prompts import FilteringPrompt, SummarizationPrompt
from .utils import (
    BatchProcessor,
    HTMLTagOutputParser,
    measure_execution_time,
)

# Total time one summary may spend checking its images. Past it, the images not
# yet proven to load are dropped rather than shipped unchecked.
IMAGE_PROBE_BUDGET_SECONDS = 120


def _drop_unrenderable_images(
    summary_html: str,
    source_html: str,
    budget_seconds: float = IMAGE_PROBE_BUDGET_SECONDS,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[str, list[str]]:
    """Keep only images a mail client will display.

    Outlook for Windows shows a WebP image's alt text instead, so the alt text
    and the caption ran together as one odd sentence ("Massive Text Embedding
    Benchmark (Code)코드 검색 부문 ... 차트입니다"). An image that fails is
    swapped for another rendition the article lists (NVIDIA serves WebP as the
    main file and PNG in its srcset); with none, its figure goes. Returns
    ``(cleaned_html, dropped_srcs)``.
    """
    if "<img" not in summary_html.lower():
        return summary_html, []
    deadline = clock() + budget_seconds
    soup = BeautifulSoup(summary_html, "html.parser")
    dropped: list[str] = []
    changed = False
    for img in list(soup.find_all("img")):
        src = str(img.get("src") or "")
        chosen = None
        for candidate in [src, *_image_candidates(src, source_html)]:
            if clock() >= deadline:
                break
            if is_email_renderable_image(candidate):
                chosen = candidate
                break
        if chosen is None:
            dropped.append(src)
            (img.find_parent("figure") or img).decompose()
        elif chosen != src:
            img["src"] = chosen
            changed = True
    return (str(soup), dropped) if (dropped or changed) else (summary_html, [])


class SummarizerSettings(SummarizationOptions):
    """Everything the Summarizer needs, as one validated object.

    The composition root builds it from config
    (``model_validate(cfg.summarization.model_dump() | {...})``), adding the
    values only the run knows: the issue's language and the content gate.
    """

    # The output-repair model. Not a config key; scripts/put_inference_profiles.py
    # reads this default so the profile it provisions matches the model used.
    fixing_model_id: LanguageModelId = LanguageModelId.CLAUDE_V5_5_HAIKU
    language: Language = Language.KO
    min_content_length: int = Field(default=600, ge=0)


class Summarizer:
    def __init__(
        self,
        boto_session: boto3.Session,
        settings: SummarizerSettings,
    ) -> None:
        self.boto_session = boto_session
        self.settings = settings
        self.language = settings.language
        self.filtered_out_posts: list[tuple[Post, str]] = []
        self.batch_processor = BatchProcessor(
            max_concurrency=settings.max_concurrency,
            batch_size=settings.max_concurrency,
        )
        self.llm_factory = BedrockLanguageModelFactory(boto_session=self.boto_session)
        self.filter = self._create_filter(
            settings.filtering_model_id,
            settings.filtering_enable_thinking,
            thinking_budget_tokens=settings.filtering_thinking_budget_tokens,
        )
        self.summarizer = self._create_summarizer(
            settings.summarization_model_id,
            settings.fixing_model_id,
            settings.summarization_enable_thinking,
            thinking_budget_tokens=settings.summarization_thinking_budget_tokens,
            max_tokens=settings.summarization_max_tokens,
        )

    # Output cap for one filter call: a score and a one-line reason, plus room for
    # the reasoning some filter models do whether asked or not. Left unset, every
    # call reserves the model's full output maximum against the per-minute token
    # quota, and filtering sends one call per collected post.
    FILTER_MAX_TOKENS: ClassVar[int] = 8192

    def _create_filter(
        self,
        model_id: LanguageModelId,
        use_thinking: bool = False,
        thinking_budget_tokens: int | None = None,
    ) -> Runnable:
        model_info = self.llm_factory.get_model_info(model_id)
        if not model_info:
            raise ValueError(f"Invalid model ID: '{model_id}'")
        if use_thinking and model_info.supports_thinking:
            # Extended thinking forces temperature=1.0 (Anthropic API rule), so
            # the 0.00-1.00 rubric scores become non-deterministic: the same
            # post may fall either side of min_score across runs, making the
            # score-ranked top-N cut unstable. Acceptable if you want deeper
            # reasoning, but be aware inclusion is no longer reproducible.
            # Models without sampling controls are non-deterministic with or
            # without this switch (and Opus 5.5 and Fable 5.1 think either way), so
            # the temperature claim would be false for them.
            if model_info.supports_sampling_params:
                logger.warning(
                    "Filtering has thinking enabled: scores are sampled at "
                    "temperature=1.0 and are NOT deterministic across runs."
                )
            else:
                logger.warning(
                    "Filtering has thinking enabled on a model without sampling "
                    "controls: scores are NOT deterministic across runs."
                )
        filtering_llm = self.llm_factory.get_model(
            model_id=model_id,
            temperature=0.0,
            stage="filtering",
            max_tokens=self.FILTER_MAX_TOKENS,
            enable_thinking=use_thinking,
            thinking_budget_tokens=thinking_budget_tokens
            or self.llm_factory.DEFAULT_THINKING_BUDGET_TOKENS,
            effort=self.settings.thinking_effort,
        )
        # The large, static filtering rubric is sent on every post — cache it as
        # a system prefix so only the per-post content is billed at full rate.
        # The cache marker differs by backend, so tell the prompt which one runs.
        prompt = FilteringPrompt.for_criteria(
            self.settings.filtering_criteria
        ).get_prompt(
            enable_prompt_cache=model_info.supports_prompt_caching,
            use_converse=isinstance(filtering_llm, ChatBedrockConverse),
        )
        return (
            prompt
            | filtering_llm
            | HTMLTagOutputParser(
                tag_names=FilteringPrompt.output_variables,
                text_tags=FilteringPrompt.output_variables,
            )
        )

    def _cache_can_hit(self) -> bool:
        """Whether caching the summarization prefix can ever pay for itself.

        A cache entry is only written when a response completes, so every request
        in a concurrent wave starts against a cold cache. The summarization batch
        is bounded by ``max_posts`` and runs ``max_concurrency`` at a time, so when
        the whole batch fits in one wave *every* call misses — and the digest is
        weekly, far beyond the entry lifetime, so nothing survives to the next run
        either. Writing costs 1.25x the normal input rate, which made caching a
        pure surcharge in that configuration.

        This is measured, not predicted: across 24 real summarization calls the
        usage logger recorded ``cache_read=0`` on every one, with a
        ``cache_write`` on every one. Filtering is not gated by this: it scores
        every collected post in several waves, so later waves can hit (on models
        that accept a cache point at all).
        """
        max_posts = self.settings.max_posts
        # None means "no cap", so the batch is however many posts survived
        # filtering — routinely more than one wave, and caching helps.
        return max_posts is None or max_posts > self.settings.max_concurrency

    def _create_summarizer(
        self,
        model_id: LanguageModelId,
        fixing_model_id: LanguageModelId,
        use_thinking: bool = False,
        thinking_budget_tokens: int | None = None,
        max_tokens: int | None = None,
    ):
        model_info = self.llm_factory.get_model_info(model_id)
        if not model_info:
            raise ValueError(f"Invalid model ID: '{model_id}'")
        fixing_llm = self.llm_factory.get_model(
            model_id=LanguageModelId(fixing_model_id),
            temperature=0.0,
            stage="output-fixing",
        )
        # required_tags=["summary"] makes the parser raise on empty/malformed
        # output so OutputFixingParser actually invokes the fixing model to
        # repair it — otherwise a missing summary would silently pass through.
        output_parser = OutputFixingParser.from_llm(
            llm=fixing_llm,
            parser=HTMLTagOutputParser(
                tag_names=SummarizationPrompt.output_variables,
                required_tags=["summary"],
                text_tags=self.SUMMARY_TEXT_TAGS,
            ),
        )
        summarization_llm = self.llm_factory.get_model(
            model_id=model_id,
            temperature=0.0,
            stage="summarization",
            enable_thinking=use_thinking,
            thinking_budget_tokens=thinking_budget_tokens
            or self.llm_factory.DEFAULT_THINKING_BUDGET_TOKENS,
            max_tokens=max_tokens,
            effort=self.settings.thinking_effort,
        )
        # Cache the static analysis instructions as a system prefix — but only
        # when a later call can actually READ what an earlier one wrote.
        prompt = SummarizationPrompt.for_language(self.language).get_prompt(
            enable_prompt_cache=model_info.supports_prompt_caching
            and self._cache_can_hit(),
            use_converse=isinstance(summarization_llm, ChatBedrockConverse),
        )
        return prompt | summarization_llm | output_parser

    @measure_execution_time
    def process_posts(self, posts: list[Post]) -> list[Post]:
        substantive_posts = self._gate_by_content_length(posts)
        if not substantive_posts:
            logger.warning("No posts had sufficient content to summarize.")
            return []
        if self.settings.use_filtering:
            posts_to_process = self._filter_posts(substantive_posts)
            if not posts_to_process:
                logger.warning("No posts remained after filtering.")
                return []
        else:
            posts_to_process = substantive_posts
        # Rank and apply max_posts BEFORE summarizing, so we only pay to
        # summarize the posts that will actually ship (summarization is the
        # expensive LLM step).
        return self._summarize_lineup(posts_to_process)

    # How many times a failed summary may be replaced by the next-ranked post.
    # Bounded so a systematic failure (a rejected request shape) costs a few
    # extra calls, not one per eligible post.
    MAX_BACKFILL_ROUNDS: ClassVar[int] = 2

    def _summarize_lineup(self, eligible: list[Post]) -> list[Post]:
        """Summarize the issue's line-up, replacing posts whose summary failed.

        Selection happens BEFORE summarizing so only shipped posts are paid for.
        But a post whose summary then failed simply vanished: the issue went out
        a card short even when other posts had passed the filter, and the post
        lost was usually the top-ranked one. Each round re-selects from the
        eligible posts minus the failures — so the per-source cap still holds —
        and summarizes only the posts not yet done.
        """
        pool = list(eligible)
        done: dict[str, Post] = {}
        lineup = self._select_for_digest(pool)
        for round_number in range(self.MAX_BACKFILL_ROUNDS + 1):
            pending = [p for p in lineup if p.link not in done]
            if not pending:
                break
            for post in self._summarize_posts(pending):
                done[post.link] = post
            failed = {p.link for p in pending if p.link not in done}
            if not failed:
                break
            pool = [p for p in pool if p.link not in failed]
            if round_number == self.MAX_BACKFILL_ROUNDS:
                logger.warning(
                    "%d summary failure(s) left after %d backfill round(s); "
                    "the issue ships short.",
                    len(failed),
                    self.MAX_BACKFILL_ROUNDS,
                )
                break
            lineup = self._select_for_digest(pool)
            logger.warning(
                "%d summary failure(s); refilling the line-up from the next-ranked "
                "eligible posts.",
                len(failed),
            )
        return [p for p in lineup if p.link in done]

    def _select_for_digest(self, posts: list[Post]) -> list[Post]:
        return select_for_digest(
            posts, self.settings.max_posts, self.settings.max_per_source
        )

    def _gate_by_content_length(self, posts: list[Post]) -> list[Post]:
        """Drop posts whose visible text is too thin to summarize well.

        Dropped posts are recorded in ``filtered_out_posts`` so the run report
        shows them.
        """
        substantive, dropped = gate_by_content_length(
            posts, self.settings.min_content_length
        )
        self.filtered_out_posts.extend(dropped)
        return substantive

    def _filter_posts(self, posts: list[Post]) -> list[Post]:
        valid_posts = [p for p in posts if p.content]
        # Reachable only with min_content_length=0 (the gate above otherwise
        # requires real text). Record rather than drop, so a post can never
        # vanish from BOTH the surviving list and the filtered-out report.
        for post in posts:
            if not post.content:
                self.filtered_out_posts.append(
                    (post, "No content available to evaluate.")
                )
                logger.warning("Post '%s' has no content; skipping.", post.title)
        if not valid_posts:
            return []

        included = ", ".join(self.settings.included_topics)
        excluded = ", ".join(self.settings.excluded_topics)
        strip_markup = self.settings.filter_on_visible_text

        def prepare_inputs(items: list[Post]) -> list[dict[str, Any]]:
            return [
                {
                    # The filter scores topic relevance and quality, both of which
                    # live in the prose — markup carries none of it. Measured on a
                    # real run, raw HTML was 8x the visible text (one page: 77,417
                    # chars of HTML around 5,669 chars of article), so sending HTML
                    # meant ~87% of filtering input tokens were nav bars, inline
                    # styles and scripts. Summarization still receives the full
                    # HTML, which it needs for code blocks, tables and image URLs.
                    "post": visible_text(post.content)
                    if strip_markup
                    else post.content,
                    "original_title": post.title,
                    # Provenance is evidence the text may not state: an Amazon
                    # Science post rarely names Amazon, and scored as off-topic.
                    "source_url": str(post.link),
                    "included_topics": included,
                    "excluded_topics": excluded,
                }
                for post in items
            ]

        responses = self.batch_processor.execute_with_fallback(
            items_to_process=valid_posts,
            prepare_inputs_func=prepare_inputs,
            batch_func=self.filter.batch,
            sequential_func=self.filter.invoke,
            task_name="filtering",
        )
        filtered_posts = []
        # Lengths are guaranteed equal (both batch and sequential-fallback paths
        # preserve positional alignment), so strict=True catches any regression.
        for post, response in zip(valid_posts, responses, strict=True):
            if response is None:
                # The model call failed for this post even after fallback.
                self.filtered_out_posts.append(
                    (post, f"{FILTER_FAILED} (no model response).")
                )
                logger.warning("No filter response for '%s'; skipping.", post.title)
                continue
            if not str(response.get("score") or "").strip():
                # No score at all is a failed call, not a verdict. Defaulting it
                # to 0.0 filed a refusal, a truncated answer or a malformed one
                # under "low relevance", indistinguishable from a real low score
                # — so a model that declined every post read as a quiet week.
                logger.error(
                    "Filter response for '%s' has no score; treating it as a "
                    "failed call. Response: %s",
                    post.title,
                    response,
                )
                self.filtered_out_posts.append(
                    (post, f"{FILTER_FAILED} (no score in model response).")
                )
                continue
            try:
                score = float(response["score"])
                # Out of [0,1] is a failed call, not a verdict: clamping the
                # model's "8.5" (meant as 0.85) to 1.0 would rank it first.
                # A value off the rubric's anchors is only logged.
                if not 0.0 <= score <= 1.0:
                    logger.error(
                        "Filter score %.3f for '%s' is outside [0,1]; treating it "
                        "as a failed call.",
                        score,
                        post.title,
                    )
                    self.filtered_out_posts.append(
                        (post, f"{FILTER_FAILED} (score {score} out of range).")
                    )
                    continue
                if not is_anchor(score):
                    logger.info(
                        "Filter score %.3f for '%s' is not one of the rubric's anchors.",
                        score,
                        post.title,
                    )
                reason = response.get("reason", "No reason provided.")
                post.score = score
                if score >= self.settings.min_score:
                    filtered_posts.append(post)
                    logger.info(
                        "Post '%s' passed filter with score %.2f.", post.title, score
                    )
                else:
                    self.filtered_out_posts.append((post, reason))
                    logger.info(
                        "Post '%s' filtered out with score %.2f. Reason: %s",
                        post.title,
                        score,
                        reason,
                    )
            except (ValueError, TypeError) as e:
                logger.error(
                    "Could not parse filter response for '%s': %s. Response: %s",
                    post.title,
                    e,
                    response,
                )
                # Record it so the post is accounted for in the run report
                # rather than silently vanishing from both lists.
                self.filtered_out_posts.append((post, f"{FILTER_UNPARSEABLE}: {e}"))
        return filtered_posts

    def _summarize_posts(self, posts: list[Post]) -> list[Post]:
        """Summarize ``posts`` in place and return ONLY the ones that succeeded.

        The success set is returned explicitly rather than inferred afterwards
        from ``post.summary`` being non-empty. That inference was subtly wrong:
        any pre-existing text in the field (the RSS teaser, before it stopped
        being seeded — see ``post_from_entry``) made a FAILED summarization look
        successful, so the post shipped its teaser as the article body while also
        appearing in the filtered-out report.
        """
        if not posts:
            return []

        def prepare_inputs(items: list[Post]) -> list[dict[str, Any]]:
            # Site furniture is stripped but the markup is kept: the summary
            # needs the code blocks, tables and image URLs that live in it.
            return [
                {
                    "post": content_html(post.content)
                    if self.settings.summarize_on_cleaned_html
                    else post.content
                }
                for post in items
            ]

        summaries = self.batch_processor.execute_with_fallback(
            items_to_process=posts,
            prepare_inputs_func=prepare_inputs,
            batch_func=self.summarizer.batch,
            sequential_func=self.summarizer.invoke,
            task_name="summarization",
        )
        succeeded: list[Post] = []
        for post, summary_data in zip(posts, summaries, strict=True):
            if summary_data is None:
                logger.warning(
                    "No summary produced for '%s' (model call failed).", post.title
                )
                # Record so the post is accounted for in the run report and, if
                # every summary fails, the empty-digest alert fires instead of a
                # silent empty send. Mirrors _filter_posts' failure handling.
                self.filtered_out_posts.append(
                    (post, f"{SUMMARY_FAILED} (no model response).")
                )
                continue
            try:
                summary_output = SummaryOutput.model_validate(summary_data)
                finalized = self._finalize_summary(post, summary_output.summary)
                if problem := self._unusable_summary(finalized):
                    logger.warning("Summary for '%s' rejected: %s", post.title, problem)
                    self.filtered_out_posts.append(
                        (post, f"{SUMMARY_FAILED} ({problem}).")
                    )
                    continue
                post.summary = finalized
                post.one_liner = summary_output.one_liner
                post.tags = summary_output.tags
                post.urls = summary_output.urls
                if unsourced := _unsourced_urls(post.urls, post.content):
                    logger.warning(
                        "%d reference link(s) for '%s' are not present in the "
                        "source article (kept, but verify): %s",
                        len(unsourced),
                        post.title,
                        unsourced,
                    )
                succeeded.append(post)
            except Exception as e:
                logger.error(
                    "Failed to parse summary for post '%s': %s. Data: %s",
                    post.title,
                    e,
                    summary_data,
                )
                self.filtered_out_posts.append((post, f"{SUMMARY_INVALID}: {e}"))
        return succeeded

    # Summary tags read as plain text. Not one_liner: _clean_one_liner already
    # decodes it once, and a second decode turned "List&lt;String&gt;" into
    # "List" on the card.
    SUMMARY_TEXT_TAGS: ClassVar[list[str]] = ["tags"]
    # A real summary is thousands of characters (KO band 4,000-6,000, EN
    # 1,300-2,000 words). A refusal, a "too short to summarize" reply or a
    # repair that wrapped either of them in tags is a sentence or two. The floor
    # sits far below the band so a terse but genuine summary still ships.
    MIN_SUMMARY_SHARE: ClassVar[float] = 0.3
    EN_MIN_WORDS: ClassVar[int] = 400

    def _unusable_summary(self, summary_html: str) -> str | None:
        """Why this summary must not ship, or None if it may.

        The output parser only checks that a ``<summary>`` tag exists. Without
        this, a refused or empty answer that the repair model wrapped in tags
        was published as an article card.
        """
        text = visible_text(summary_html)
        if self.language is Language.KO:
            minimum = int(KO_LENGTH_BAND[0] * self.MIN_SUMMARY_SHARE)
            if len(text) < minimum:
                return f"{len(text)} visible chars, below {minimum}"
            if not is_korean(text):
                return "not written in Korean"
        elif len(text.split()) < self.EN_MIN_WORDS:
            return f"{len(text.split())} words, below {self.EN_MIN_WORDS}"
        return None

    @staticmethod
    def _finalize_summary(post: Post, summary_html: str) -> str:
        cleaned, dropped = _strip_unsourced_images(
            summary_html, post.content, post.images
        )
        if dropped:
            logger.warning(
                "Dropped %d image(s) from the summary of '%s' — not found in the "
                "source article (would have rendered broken): %s",
                len(dropped),
                post.title,
                dropped,
            )
        cleaned, unrenderable = _drop_unrenderable_images(cleaned, post.content)
        if unrenderable:
            logger.warning(
                "Dropped %d image(s) from the summary of '%s' — not a type mail "
                "clients display, or not loadable: %s",
                len(unrenderable),
                post.title,
                unrenderable,
            )
        return _postprocess_summary(_encode_image_srcs(cleaned))

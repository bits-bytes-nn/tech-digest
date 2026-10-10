"""Small, unrelated helpers shared across the app.

Bedrock model construction used to live here too, which made this the largest
module in the project; it now lives in ``model_factory``.
"""

import functools
import math
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import tenacity
from botocore.exceptions import ClientError, HTTPClientError
from botocore.exceptions import ConnectionError as BotoConnectionError
from bs4 import BeautifulSoup
from email_validator import EmailNotValidError
from email_validator import validate_email as _validate_email
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from tqdm import tqdm

from .logger import logger

# Retry policy for a single Bedrock call, shared by the batch processor's
# sequential fallback and by ``retry_with_backoff``. One set of numbers: these
# used to be module constants for one caller and duplicate pydantic field
# defaults for the other, with nothing keeping the two in step.
MAX_RETRIES: int = 5
RETRY_MAX_WAIT: int = 120
RETRY_MULTIPLIER: float = 30.0

# Bedrock error codes that can succeed on a later attempt: throttling and
# server-side faults. Every other code a ClientError carries — ValidationException,
# AccessDeniedException, ResourceNotFoundException — fails identically every time.
RETRYABLE_ERROR_CODES: frozenset[str] = frozenset(
    {
        "InternalServerException",
        "ModelErrorException",
        "ModelNotReadyException",
        "ModelTimeoutException",
        "ServiceUnavailableException",
        "ThrottlingException",
        "TooManyRequestsException",
    }
)


class EmptyModelOutput(ValueError):
    """The model returned nothing where a required tag was expected.

    Deliberately NOT an ``OutputParserException``: ``OutputFixingParser`` would
    hand the empty completion to the repair model, which then writes a summary
    from nothing — and that invention shipped as an article card.
    """


def is_retryable(error: BaseException) -> bool:
    """Whether another attempt at the call that raised ``error`` could succeed.

    Throttling, server faults, dropped connections and timeouts can. Anything
    else is permanent: a 400 for a bad request shape or a missing permission, a
    bug in this code, or malformed output the repair model already had one try
    at — regenerating the whole summary for it re-paid the full call up to six
    times per post. Retrying those used to cost
    up to five attempts and ~330s of backoff per item, so one bad config on the
    filter (one call per collected post) held the run for well over an hour
    before the empty-digest alert could say anything.

    Only the explicit cause chain (``raise ... from``) is walked. The implicit
    ``__context__`` also links an exception raised while HANDLING another, which
    made a bug inside the repair path look like the parse error it was handling.
    """
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ClientError):
            code = current.response.get("Error", {}).get("Code", "")
            return code in RETRYABLE_ERROR_CODES
        if isinstance(current, BotoConnectionError | HTTPClientError | TimeoutError):
            return True
        current = current.__cause__
    return False


def _retry_log_callback(operation_name: str) -> Callable:
    def log_retry(retry_state):
        logger.warning(
            "Retrying '%s' (attempt %d failed). Waiting %.1fs",
            operation_name,
            retry_state.attempt_number,
            retry_state.next_action.sleep if retry_state.next_action else 0,
        )

    return log_retry


def retry_with_backoff(
    operation_name: str,
    multiplier: float = RETRY_MULTIPLIER,
    max_wait: int = RETRY_MAX_WAIT,
    attempts: int = MAX_RETRIES,
) -> Callable:
    """Exponential-backoff retry decorator that logs each attempt.

    Replaces a ``RetryableBase`` class whose only member was this as a static
    method and which was inherited purely to reach it — a mixin carrying no
    state, where a plain function does the same job without implying a hierarchy.
    """
    return tenacity.retry(
        retry=tenacity.retry_if_exception(is_retryable),
        wait=tenacity.wait_exponential(multiplier=multiplier, max=max_wait),
        stop=tenacity.stop_after_attempt(attempts),
        before_sleep=_retry_log_callback(operation_name),
        reraise=True,
    )


class BatchProcessor(BaseModel):
    max_concurrency: int = Field(default=5, ge=1)
    retry_multiplier: float = Field(default=RETRY_MULTIPLIER, ge=1.0)
    retry_max_wait: int = Field(default=RETRY_MAX_WAIT, ge=0)
    max_retries: int = Field(default=MAX_RETRIES, ge=1)
    batch_size: int = Field(default=10, ge=1)

    def execute_with_fallback(
        self,
        items_to_process: list[Any],
        prepare_inputs_func: Callable[[list[Any]], list[dict[str, Any]]],
        batch_func: Callable[..., list[Any]],
        sequential_func: Callable[..., Any],
        task_name: str,
        show_progress: bool = True,
    ) -> list[Any]:
        # Concurrency and chunk size come from the instance, which the composition
        # root sizes from config. There used to be a ``run_config`` mapping that
        # could override both per call; no caller ever passed one, so it was two
        # unreachable branches over a parameter that existed to be ignored.
        if not items_to_process:
            return []
        max_concurrency = self.max_concurrency
        batch_size = self.batch_size
        prepared_batch_func = self._create_batch_func(batch_func, max_concurrency)
        retrying_sequential_func = self._create_retry_decorator(task_name)(
            sequential_func
        )
        all_results = []
        num_items = len(items_to_process)
        num_chunks = math.ceil(num_items / batch_size)
        logger.info(
            "Starting processing for '%s': %d items in %d chunks (batch size: %d)",
            task_name,
            num_items,
            num_chunks,
            batch_size,
        )
        for i in tqdm(
            range(0, num_items, batch_size),
            desc=f"Processing: {task_name}",
            disable=not show_progress,
        ):
            chunk_items = items_to_process[i : i + batch_size]
            chunk_num = (i // batch_size) + 1
            logger.debug(
                "Processing chunk %d/%d (%d items)",
                chunk_num,
                num_chunks,
                len(chunk_items),
            )
            chunk_inputs = prepare_inputs_func(chunk_items)
            if not chunk_inputs:
                logger.warning(
                    "No valid inputs prepared for chunk %d, skipping", chunk_num
                )
                continue
            try:
                logger.debug("Attempting batch processing for chunk %d", chunk_num)
                chunk_results = prepared_batch_func(chunk_inputs)
            except Exception as e:
                logger.warning(
                    "Batch processing failed for chunk %d: %s. Falling back to sequential processing",
                    chunk_num,
                    e,
                )
                # The batch call itself broke, not any one item: every item
                # still gets its own sequential attempt.
                all_results.extend(
                    self._process_sequentially_with_fallback(
                        chunk_inputs,
                        retrying_sequential_func,
                        f"{task_name} (chunk {chunk_num})",
                        show_progress=show_progress,
                    )
                )
                continue
            # Only the items that failed go to the sequential retry. The batch is
            # run with return_exceptions, so one failed item no longer discards
            # the others' results: re-running the whole chunk billed every
            # successful call a second time (5 items, 1 failing = 9 calls, not 5).
            failed = [
                index
                for index, result in enumerate(chunk_results)
                if isinstance(result, Exception)
            ]
            if failed:
                logger.warning(
                    "%d/%d items failed in batch mode for chunk %d; retrying them "
                    "sequentially",
                    len(failed),
                    len(chunk_inputs),
                    chunk_num,
                )
                retried = self._process_sequentially_with_fallback(
                    [chunk_inputs[index] for index in failed],
                    retrying_sequential_func,
                    f"{task_name} (chunk {chunk_num})",
                    show_progress=show_progress,
                    first_errors=[chunk_results[index] for index in failed],
                )
                chunk_results = list(chunk_results)
                for index, result in zip(failed, retried, strict=True):
                    chunk_results[index] = result
            else:
                logger.debug("Chunk %d processed successfully in batch mode", chunk_num)
            all_results.extend(chunk_results)
        logger.info("Completed '%s': processed %d results", task_name, len(all_results))
        return all_results

    @staticmethod
    def _create_batch_func(
        batch_func: Callable[..., list[Any]], max_concurrency: int
    ) -> Callable:
        def _batch_func(inputs: list[dict[str, Any]]) -> list[Any]:
            return batch_func(
                inputs,
                config=RunnableConfig(max_concurrency=max_concurrency),
                return_exceptions=True,
            )

        return _batch_func

    def _create_retry_decorator(self, operation_name: str) -> Callable:
        return retry_with_backoff(
            operation_name,
            multiplier=self.retry_multiplier,
            max_wait=self.retry_max_wait,
            attempts=self.max_retries,
        )

    @staticmethod
    def _process_sequentially_with_fallback(
        inputs: list[dict[str, Any]],
        sequential_func: Callable[[dict[str, Any]], Any],
        task_name: str,
        show_progress: bool = True,
        first_errors: list[Exception] | None = None,
    ) -> list[Any]:
        logger.info("Processing %d items sequentially for '%s'", len(inputs), task_name)
        results: list[Any] = []
        progress_desc = f"Sequential Processing: '{task_name}'"
        successful_count = 0
        errors: list[Exception | None] = (
            list(first_errors) if first_errors else [None] * len(inputs)
        )
        for single_input, first_error in tqdm(
            list(zip(inputs, errors, strict=True)),
            desc=progress_desc,
            disable=not show_progress,
        ):
            # The batch attempt already failed this item permanently (a 400, a
            # missing permission): another call would fail the same way and be
            # billed or throttled for nothing.
            if first_error is not None and not is_retryable(first_error):
                logger.error("Not retrying an item in '%s': %s", task_name, first_error)
                results.append(None)
                continue
            try:
                results.append(sequential_func(single_input))
                successful_count += 1
            except Exception as e:
                logger.error(
                    "Sequential processing failed for single item in '%s': %s",
                    task_name,
                    e,
                )
                # Append None (not skip) so the results list stays positionally
                # aligned with inputs — callers zip results back onto their
                # items and a dropped element would misattribute every result
                # after it. Callers must treat None as "failed, no output".
                results.append(None)
        logger.info(
            "Sequential processing completed for '%s': %d/%d items processed successfully",
            task_name,
            successful_count,
            len(inputs),
        )
        return results


class HTMLTagOutputParser(BaseOutputParser):
    tag_names: str | list[str]
    # Tags that MUST be present and non-empty; if any is missing the parser
    # raises OutputParserException so a wrapping OutputFixingParser actually
    # triggers its repair model. Empty by default to preserve lenient behavior
    # (the filter tolerates missing optional fields downstream).
    required_tags: list[str] = Field(default_factory=list)
    # Tags whose value is plain text (a title, a score, a tag list), returned as
    # text rather than serialized inner HTML. decode_contents() re-escapes "&",
    # so "Q&A" came back as "Q&amp;A" and the template's autoescape then showed
    # the reader "Q&amp;A".
    text_tags: list[str] = Field(default_factory=list)

    def parse(self, text: str) -> str | dict[str, str]:
        if not text or not text.strip():
            if self.required_tags:
                raise EmptyModelOutput(
                    f"Empty model output; required tags missing: {self.required_tags}"
                )
            return {} if isinstance(self.tag_names, list) else ""
        soup = BeautifulSoup(text, "html.parser")
        parsed: dict[str, str] = {}
        tag_list = (
            self.tag_names if isinstance(self.tag_names, list) else [self.tag_names]
        )
        for tag_name in tag_list:
            if tag := soup.find(tag_name):
                if tag_name in self.text_tags and hasattr(tag, "get_text"):
                    parsed[tag_name] = tag.get_text().strip()
                elif hasattr(tag, "decode_contents"):
                    parsed[tag_name] = str(tag.decode_contents()).strip()
                else:
                    parsed[tag_name] = str(tag).strip()
        missing = [t for t in self.required_tags if not parsed.get(t)]
        if missing:
            raise OutputParserException(
                f"Required tag(s) missing or empty in model output: {missing}. "
                f"Got tags: {sorted(parsed)}"
            )
        if isinstance(self.tag_names, list):
            return parsed
        return next(iter(parsed.values()), "")

    def get_format_instructions(self) -> str:
        """The tag contract, for ``OutputFixingParser``'s repair prompt.

        Without this method the repair never ran: the fixing parser falls back to
        a prompt with no ``instructions`` slot filled, and langchain_classic's
        template requires one, so every repair attempt raised ``KeyError`` before
        the fixing model was called — and the whole summarization was retried
        instead.
        """
        tags = self.tag_names if isinstance(self.tag_names, list) else [self.tag_names]
        lines = [
            "Return the content wrapped in these XML-style tags, each opened and "
            "closed exactly once: " + ", ".join(f"<{t}>...</{t}>" for t in tags) + "."
        ]
        if self.required_tags:
            lines.append(
                "These tags must be present and non-empty: "
                + ", ".join(f"<{t}>" for t in self.required_tags)
                + "."
            )
        lines.append(
            "Keep the original wording and markup inside the tags; only fix the "
            "structure."
        )
        return " ".join(lines)

    @property
    def _type(self) -> str:
        return "html_tag_output_parser"


def validate_email(email: str) -> bool:
    """Whether ``email`` is a syntactically valid address.

    Delegates to ``email_validator``, which the project already depends on (it is
    what backs pydantic's ``EmailStr``, used for ``newsletter.sender``). This was
    a hand-written regex, which is both a second definition of "valid address"
    and a weaker one — it rejects tagged locals with unusual characters and any
    internationalized domain, and a recipient it drops is a reader who silently
    stops receiving the digest.

    Deliverability is NOT checked: that would issue an MX lookup per recipient
    from inside the run.
    """
    try:
        _validate_email(email.strip(), check_deliverability=False)
    except EmailNotValidError:
        return False
    return True


def validate_emails(emails: list[str]) -> list[str]:
    # The address is returned as written (stripped), not in the validator's
    # normalized form: SES sends to what the recipients file says, and silently
    # rewriting an address is not this function's job.
    valid_emails = [email.strip() for email in emails if validate_email(email)]
    if len(valid_emails) < len(emails):
        logger.warning(
            "Filtered out %d invalid email addresses", len(emails) - len(valid_emails)
        )
    return valid_emails


def format_alarm(
    *,
    event: str,
    status: str,
    fields: dict[str, str],
    project: str,
    timestamp: datetime | None = None,
) -> tuple[str, str]:
    """Build a ``(subject, message)`` pair in the project family's unified alarm
    format, shared verbatim across omnisummary/paper-bridge/scholar-lens:

        Subject: [<project>] <event> — <STATUS>

        <event> <STATUS>

        Key:   Value

        — 2026-06-10 04:12:00 UTC

    ``status`` is a short uppercase state (``FAILED``/``ALERT``). ``fields`` is an
    ordered mapping; single-line values render as an aligned ``Key: Value`` block,
    multi-line values render under their own ``Key:`` header. Omit a row by leaving
    it out of the dict.

    ``project`` is required rather than defaulted. It used to default to the
    literal "tech-digest" and no caller passed it, so every alarm subject named
    that project regardless of ``resources.project_name`` — a deployment under any
    other name would have paged with the wrong identity in the subject line.
    """
    ts = (timestamp or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M:%S")
    subject = f"[{project}] {event} — {status}"

    inline = {k: v for k, v in fields.items() if "\n" not in v}
    block = {k: v for k, v in fields.items() if "\n" in v}

    lines = [f"{event} {status}", ""]
    if inline:
        width = max(len(k) for k in inline)
        lines += [f"{k + ':':<{width + 1}} {v}" for k, v in inline.items()]
    for k, v in block.items():
        lines += ["", f"{k}:", v.strip("\n")]
    lines.append("")
    lines.append(f"— {ts} UTC")

    return subject, "\n".join(lines)


def get_date_range(
    end_date_str: str | None, days_back: int
) -> tuple[datetime, datetime]:
    """The half-open window ``[start, end)`` an issue dated ``end`` covers.

    ``end`` is midnight UTC at the START of the issue date (today when no date is
    given), and the issue covers the ``days_back`` full days before it. Gates
    compare with ``start <= t < end`` (see ``is_date_in_range``), so the windows
    of consecutive weekly runs abut exactly, whatever hour the job runs at.

    The window used to END at 23:59:59 on the run day. A scheduled run at 01:00
    UTC cannot see posts published later that day, and the next run's window
    started at that same 23:59:59 — so everything published on a run day after
    the job ran (23 of every 168 hours) was never collected by any issue.
    """
    if end_date_str:
        day = datetime.strptime(end_date_str, "%Y-%m-%d").replace(tzinfo=UTC)
    else:
        day = datetime.now(UTC)
    end_date = day.replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=days_back)
    return start_date, end_date


def measure_execution_time(func: Callable) -> Callable:
    """Log how long a synchronous call took.

    There is no async variant: the pipeline is synchronous throughout (LangChain
    `.batch()` fans out with threads, not a loop), so the coroutine branch this
    decorator used to carry was never constructed on any code path.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        logger.info(
            "'%s' execution time: %.2fs (%.2fmin)",
            func.__name__,
            execution_time,
            execution_time / 60,
        )
        return result

    return wrapper

"""Per-source health of one crawl run."""

from dataclasses import dataclass, field
from enum import Enum


class SourceStatus(str, Enum):
    """Health status of a single crawl source for one run."""

    OK = "ok"  # fetched successfully and produced posts in range
    EMPTY = "empty"  # fetched successfully but no posts in the date window
    FAILED = "failed"  # the fetch itself errored (network, parse, anti-bot)


@dataclass
class SourceHealth:
    url: str
    fetcher: str
    status: SourceStatus
    #: Posts this source contributed to the digest, after cross-source dedup.
    post_count: int = 0
    #: Posts it returned that a previous source had already provided.
    duplicates: int = 0
    #: Items with a readable date the source offered BEFORE the date window.
    #: It separates the two reasons a source can be EMPTY: a quiet blog offers a
    #: back catalogue the window filters out; a moved feed or a broken scraper
    #: offers nothing. Only the second needs a person (``likely_broken``).
    #:
    #: ``None`` means the fetcher did not report a count — unknown, NOT zero.
    candidates: int | None = None
    #: Contributed posts whose article page was needed but could not be read
    #: (``Post.page_unreadable``).
    unreadable_pages: int = 0
    #: The feed did not reach back to the window start (``RssFetcher.truncated``).
    truncated: bool = False
    error: str | None = None


@dataclass
class CrawlReport:
    """Aggregate health of all sources for a single crawl run."""

    sources: list[SourceHealth] = field(default_factory=list)

    @property
    def failed(self) -> list[SourceHealth]:
        return [s for s in self.sources if s.status is SourceStatus.FAILED]

    @property
    def empty(self) -> list[SourceHealth]:
        return [s for s in self.sources if s.status is SourceStatus.EMPTY]

    @property
    def ok(self) -> list[SourceHealth]:
        return [s for s in self.sources if s.status is SourceStatus.OK]

    @property
    def likely_broken(self) -> list[SourceHealth]:
        """EMPTY sources that offered NOTHING for the date window to filter.

        A feed that moved, or a scraper whose selector stopped matching: the fetch
        itself succeeded, so the source is not FAILED, but there was no material at
        any date. Distinct from a blog that was simply quiet, which offers its back
        catalogue and has the window filter it out.

        Deliberately NOT a fourth ``SourceStatus``: "fetched fine, produced no
        posts in range" is accurate for both, so the inference lives here instead
        of in a state every caller has to handle.

        ``candidates is None`` (a fetcher that reports no count) is excluded —
        unknown is not zero, and guessing here would page on the unknown case.
        """
        return [s for s in self.empty if s.candidates == 0]

    @property
    def pages_unreadable(self) -> list[SourceHealth]:
        """OK sources whose every contributed post lacked a readable article page.

        The feed fetched, so the source reads as healthy, yet each post carries
        only its teaser and dies at the content gate (e.g. article pages behind a
        Cloudflare challenge). Some unreadable pages among readable ones are
        transient errors and are not reported.
        """
        return [
            s for s in self.ok if s.post_count and s.unreadable_pages == s.post_count
        ]

    @property
    def truncated(self) -> list[SourceHealth]:
        """Sources whose feed did not reach back to the window start."""
        return [s for s in self.sources if s.truncated]

    @property
    def total_posts(self) -> int:
        return sum(s.post_count for s in self.sources)

    def summary_line(self) -> str:
        broken = len(self.likely_broken)
        empty = f"{len(self.empty)} empty"
        if broken:
            empty += f" ({broken} with nothing to filter)"
        ok = f"{len(self.ok)} ok"
        if unreadable := len(self.pages_unreadable):
            ok += f" ({unreadable} with unreadable article pages)"
        return f"{ok}, {empty}, {len(self.failed)} failed ({self.total_posts} posts)"

    @staticmethod
    def _empty_reason(source: SourceHealth) -> str:
        """Why this source produced nothing, in the reader's terms."""
        if source.candidates is None:
            return "no count reported"
        if source.candidates == 0:
            return "offered 0 items — LIKELY BROKEN (moved feed or stale selector)"
        return f"offered {source.candidates} item(s), none in the date window"

    def format_alert(self) -> str:
        """Human-readable health report for SNS/email notification."""
        lines = [f"Crawl source health: {self.summary_line()}", ""]
        if self.failed:
            lines.append("FAILED sources (need attention):")
            lines.extend(
                f"  - {s.url} [{s.fetcher}]: {s.error or 'unknown error'}"
                for s in self.failed
            )
            lines.append("")
        if self.pages_unreadable:
            lines.append("Sources whose article pages could not be read:")
            lines.extend(
                f"  - {s.url} [{s.fetcher}]: {s.unreadable_pages} of "
                f"{s.post_count} post(s) left with only the feed teaser"
                for s in self.pages_unreadable
            )
            lines.append("")
        if self.empty:
            lines.append("Empty sources (no posts in window):")
            lines.extend(
                f"  - {s.url} [{s.fetcher}]: {self._empty_reason(s)}"
                for s in self.empty
            )
            lines.append("")
        lines.append("Healthy sources:")
        lines.extend(
            f"  - {s.url}: {s.post_count} posts"
            + (f" ({s.duplicates} deduped)" if s.duplicates else "")
            for s in self.ok
        )
        return "\n".join(lines)

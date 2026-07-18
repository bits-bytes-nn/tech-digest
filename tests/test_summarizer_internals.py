"""Tests for Summarizer._filter_posts / _summarize_posts — the score parsing,
min_score boundary, title overwrite, None-response (post-fallback) alignment,
and malformed-output handling. Built via __new__ to avoid Bedrock setup."""

from __future__ import annotations

from datetime import UTC, datetime

from app.src.feed_parser import Post
from app.src.summarizer import Summarizer


def _post(title: str, content: str = "body") -> Post:
    return Post(
        title=title,
        link=f"https://example.com/{title}",
        published_date=datetime.now(UTC),
        content=content,
    )


class _FakeChain:
    """Stands in for the filter/summarizer LangChain runnable: .batch returns
    the queued responses (aligned to inputs)."""

    def __init__(self, responses):
        self._responses = responses

    def batch(self, inputs, config=None):
        return self._responses

    def invoke(self, single):  # used only if batch fails; not exercised here
        raise AssertionError("invoke should not be called in these tests")


def _summarizer(min_score=0.7):
    s = Summarizer.__new__(Summarizer)
    s.min_score = min_score
    s.min_content_length = 0
    s.filtered_out_posts = []
    from app.src.utils import BatchProcessor

    s.batch_processor = BatchProcessor(max_concurrency=2, batch_size=10)
    return s


class TestFilterPosts:
    def test_score_boundary_inclusive(self):
        s = _summarizer(min_score=0.70)
        posts = [_post("a"), _post("b")]
        s.filter = _FakeChain(
            [
                {"score": "0.70", "reason": "ok", "title": ""},  # == min: kept
                {"score": "0.69", "reason": "low", "title": ""},  # < min: dropped
            ]
        )
        kept = s._filter_posts(posts, [], [])
        assert [p.title for p in kept] == ["a"]
        assert len(s.filtered_out_posts) == 1

    def test_out_of_range_score_clamped(self):
        # A dropped decimal point ('8.5' for 0.85) must be clamped to [0,1] so it
        # can't dominate the rank and evict genuinely better posts.
        s = _summarizer(min_score=0.7)
        posts = [_post("a")]
        s.filter = _FakeChain([{"score": "8.5", "reason": "x", "title": ""}])
        kept = s._filter_posts(posts, [], [])
        assert kept[0].score == 1.0

    def test_negative_score_clamped_to_zero(self):
        s = _summarizer(min_score=0.7)
        posts = [_post("a")]
        s.filter = _FakeChain([{"score": "-0.3", "reason": "x", "title": ""}])
        kept = s._filter_posts(posts, [], [])
        # Clamped to 0.0, below min_score -> filtered out, not kept.
        assert kept == []
        assert posts[0].score == 0.0

    def test_title_overwrite_applied(self):
        s = _summarizer(min_score=0.5)
        posts = [_post("orig")]
        s.filter = _FakeChain(
            [{"score": "0.9", "reason": "x", "title": "Better Title"}]
        )
        kept = s._filter_posts(posts, [], [])
        assert kept[0].title == "Better Title"

    def test_non_numeric_score_does_not_crash_and_is_recorded(self):
        s = _summarizer(min_score=0.5)
        posts = [_post("a")]
        s.filter = _FakeChain([{"score": "not-a-number", "reason": "x"}])
        kept = s._filter_posts(posts, [], [])
        # Parse error is logged; the post is not included BUT is accounted for
        # in filtered_out_posts (not silently lost from the run report).
        assert kept == []
        assert len(s.filtered_out_posts) == 1
        assert "unparseable" in s.filtered_out_posts[0][1].lower()

    def test_none_response_aligned_and_recorded(self):
        # Simulates the sequential-fallback path returning None for a failed
        # item — must not misalign or crash, and the post is recorded.
        s = _summarizer(min_score=0.5)
        posts = [_post("a"), _post("b")]
        s.filter = _FakeChain([None, {"score": "0.9", "reason": "ok", "title": ""}])
        kept = s._filter_posts(posts, [], [])
        assert [p.title for p in kept] == ["b"]
        assert any("failed" in r.lower() for _, r in s.filtered_out_posts)

    def test_skips_posts_without_content(self):
        s = _summarizer(min_score=0.5)
        posts = [_post("a", content=""), _post("b", content="real")]
        s.filter = _FakeChain([{"score": "0.9", "reason": "ok", "title": ""}])
        kept = s._filter_posts(posts, [], [])
        assert [p.title for p in kept] == ["b"]


class TestSummarizePosts:
    def test_summary_tags_urls_set(self):
        s = _summarizer()
        post = _post("a")
        s.summarizer = _FakeChain(
            [{"summary": "Real summary content.", "tags": "RAG, LLM", "urls": []}]
        )
        s._summarize_posts([post])
        assert "Real summary content." in post.summary
        # Tags preserve first-seen (relevance) order, not alphabetical.
        assert post.tags == ["RAG", "LLM"]

    def test_none_summary_aligned_no_crash(self):
        s = _summarizer()
        posts = [_post("a"), _post("b")]
        s.summarizer = _FakeChain(
            [None, {"summary": "Second.", "tags": [], "urls": []}]
        )
        s._summarize_posts(posts)
        assert posts[0].summary == ""  # failed item untouched
        assert "Second." in posts[1].summary

    def test_malformed_summary_logged_not_raised(self):
        s = _summarizer()
        post = _post("a")
        # Missing 'summary' key -> SummaryOutput validation fails -> logged.
        s.summarizer = _FakeChain([{"tags": "X", "urls": []}])
        s._summarize_posts([post])
        assert post.summary == ""


class TestProcessPostsCapBeforeSummarize:
    """max_posts must be applied BEFORE summarization so we don't pay to
    summarize posts that get discarded, and only the top-N are summarized."""

    def test_only_top_n_summarized(self):
        s = _summarizer(min_score=0.0)
        summarized: list[str] = []

        # Spy: record which posts reach summarization. Set a non-empty summary
        # to mimic a successful summarize (process_posts now returns only posts
        # that were actually summarized).
        def fake_summarize(posts):
            for p in posts:
                p.summary = f"summary of {p.title}"
            summarized.extend(p.title for p in posts)

        s._summarize_posts = fake_summarize  # type: ignore[method-assign]

        posts = [_post("low"), _post("high"), _post("mid")]
        # Filter assigns distinct scores so ranking is unambiguous.
        s.filter = _FakeChain(
            [
                {"score": "0.50", "reason": "", "title": ""},
                {"score": "0.90", "reason": "", "title": ""},
                {"score": "0.70", "reason": "", "title": ""},
            ]
        )
        result = s.process_posts(
            posts,
            use_filtering=True,
            included_topics=[],
            excluded_topics=[],
            max_posts=2,
        )
        # Only the top 2 by score are summarized and returned, in rank order.
        assert [p.title for p in result] == ["high", "mid"]
        assert summarized == ["high", "mid"]
        assert "low" not in summarized

    def test_all_summaries_fail_returns_empty_and_records_reasons(self):
        """If every summary call fails, process_posts must return [] and record
        a reason per post so main.py's empty-digest guard + alert fire instead
        of silently emailing an article-less newsletter."""
        s = _summarizer(min_score=0.0)
        posts = [_post("a"), _post("b")]
        s.filter = _FakeChain(
            [
                {"score": "0.90", "reason": "", "title": ""},
                {"score": "0.80", "reason": "", "title": ""},
            ]
        )
        # Model returns None for every post (call failed even after fallback).
        s.summarizer = _FakeChain([None, None])
        result = s.process_posts(
            posts,
            use_filtering=True,
            included_topics=[],
            excluded_topics=[],
        )
        assert result == []
        reasons = [r for _p, r in s.filtered_out_posts]
        assert reasons.count("Summarization failed (no model response).") == 2

    def test_partial_summary_failure_keeps_successful_posts(self):
        """A post whose summary fails is dropped; successfully summarized posts
        still ship."""
        s = _summarizer(min_score=0.0)
        posts = [_post("ok"), _post("bad")]
        s.filter = _FakeChain(
            [
                {"score": "0.90", "reason": "", "title": ""},
                {"score": "0.80", "reason": "", "title": ""},
            ]
        )
        s.summarizer = _FakeChain([{"summary": "good", "tags": [], "urls": []}, None])
        result = s.process_posts(
            posts,
            use_filtering=True,
            included_topics=[],
            excluded_topics=[],
        )
        assert [p.title for p in result] == ["ok"]
        assert any(
            r == "Summarization failed (no model response)."
            for _p, r in s.filtered_out_posts
        )

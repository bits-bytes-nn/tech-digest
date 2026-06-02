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
        assert post.tags == ["LLM", "RAG"]

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

        # Spy: record which posts reach summarization.
        def fake_summarize(posts):
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

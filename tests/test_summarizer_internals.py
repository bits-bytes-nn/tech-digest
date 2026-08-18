"""Tests for Summarizer._filter_posts / _summarize_posts — the score parsing,
min_score boundary, title overwrite, None-response (post-fallback) alignment,
and malformed-output handling. Built via __new__ to avoid Bedrock setup."""

from __future__ import annotations

from datetime import UTC, datetime

from app.src.constants import LanguageModelId
from app.src.feed_parser import Post
from app.src.summarizer import Summarizer, SummarizerSettings


def _post(title: str, content: str = "body", source: str = "unknown") -> Post:
    return Post(
        title=title,
        link=f"https://example.com/{title}",
        published_date=datetime.now(UTC),
        content=content,
        source=source,  # type: ignore[arg-type]
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


def _settings(**overrides) -> SummarizerSettings:
    base = {
        "filtering_model_id": LanguageModelId.CLAUDE_V5_SONNET,
        "summarization_model_id": LanguageModelId.CLAUDE_V5_SONNET,
        "min_score": 0.7,
        "min_content_length": 0,
    }
    return SummarizerSettings.model_validate(base | overrides)


def _summarizer(min_score=0.7, **overrides):
    s = Summarizer.__new__(Summarizer)
    s.settings = _settings(min_score=min_score, **overrides)
    s.language = s.settings.language
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
        kept = s._filter_posts(posts)
        assert [p.title for p in kept] == ["a"]
        assert len(s.filtered_out_posts) == 1

    def test_out_of_range_score_clamped(self):
        # A dropped decimal point ('8.5' for 0.85) must be clamped to [0,1] so it
        # can't dominate the rank and evict genuinely better posts.
        s = _summarizer(min_score=0.7)
        posts = [_post("a")]
        s.filter = _FakeChain([{"score": "8.5", "reason": "x", "title": ""}])
        kept = s._filter_posts(posts)
        assert kept[0].score == 1.0

    def test_negative_score_clamped_to_zero(self):
        s = _summarizer(min_score=0.7)
        posts = [_post("a")]
        s.filter = _FakeChain([{"score": "-0.3", "reason": "x", "title": ""}])
        kept = s._filter_posts(posts)
        # Clamped to 0.0, below min_score -> filtered out, not kept.
        assert kept == []
        assert posts[0].score == 0.0

    def test_title_overwrite_applied(self):
        s = _summarizer(min_score=0.5)
        posts = [_post("orig")]
        s.filter = _FakeChain(
            [{"score": "0.9", "reason": "x", "title": "Better Title"}]
        )
        kept = s._filter_posts(posts)
        assert kept[0].title == "Better Title"

    def test_non_numeric_score_does_not_crash_and_is_recorded(self):
        s = _summarizer(min_score=0.5)
        posts = [_post("a")]
        s.filter = _FakeChain([{"score": "not-a-number", "reason": "x"}])
        kept = s._filter_posts(posts)
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
        kept = s._filter_posts(posts)
        assert [p.title for p in kept] == ["b"]
        assert any("failed" in r.lower() for _, r in s.filtered_out_posts)

    def test_skips_posts_without_content(self):
        s = _summarizer(min_score=0.5)
        posts = [_post("a", content=""), _post("b", content="real")]
        s.filter = _FakeChain([{"score": "0.9", "reason": "ok", "title": ""}])
        kept = s._filter_posts(posts)
        assert [p.title for p in kept] == ["b"]


class TestSummarizePosts:
    def test_summary_tags_urls_set(self):
        s = _summarizer()
        post = _post("a")
        s.summarizer = _FakeChain(
            [{"summary": "Real summary content.", "tags": "RAG, LLM", "urls": []}]
        )
        assert s._summarize_posts([post]) == [post]
        assert "Real summary content." in post.summary
        # Tags preserve first-seen (relevance) order, not alphabetical.
        assert post.tags == ["RAG", "LLM"]

    def test_none_summary_aligned_no_crash(self):
        s = _summarizer()
        posts = [_post("a"), _post("b")]
        s.summarizer = _FakeChain(
            [None, {"summary": "Second.", "tags": [], "urls": []}]
        )
        assert s._summarize_posts(posts) == [posts[1]]
        assert posts[0].summary == ""  # failed item untouched
        assert "Second." in posts[1].summary

    def test_malformed_summary_logged_not_raised(self):
        s = _summarizer()
        post = _post("a")
        # Missing 'summary' key -> SummaryOutput validation fails -> logged.
        s.summarizer = _FakeChain([{"tags": "X", "urls": []}])
        assert s._summarize_posts([post]) == []
        assert post.summary == ""

    def test_one_liner_captured_and_flattened(self):
        s = _summarizer()
        post = _post("a")
        s.summarizer = _FakeChain(
            [
                {
                    "summary": "Body.",
                    "one_liner": "<b>Cuts latency</b>\n  by 40%.",
                    "tags": [],
                    "urls": [],
                }
            ]
        )
        s._summarize_posts([post])
        # Markup stripped (it is autoescaped at render time) and collapsed to one
        # line so it cannot break the single-line lede layout.
        assert post.one_liner == "Cuts latency by 40%."

    def test_hallucinated_image_dropped_real_one_kept(self):
        """An <img> the source article never contained renders as a broken image
        in the delivered mail, so it must not survive."""
        real = "https://cdn.example.com/real.png"
        s = _summarizer()
        post = _post("a", content=f'<p>text</p><img src="{real}">')
        post.images = [real]
        s.summarizer = _FakeChain(
            [
                {
                    "summary": (
                        f'<p>x</p><img src="{real}">'
                        '<img src="https://cdn.example.com/invented.png">'
                    ),
                    "tags": [],
                    "urls": [],
                }
            ]
        )
        s._summarize_posts([post])
        assert real in post.summary
        assert "invented.png" not in post.summary

    def test_image_present_only_in_raw_content_is_kept(self):
        """Provenance accepts a src that appears verbatim in the post HTML even
        if our own extractor missed it (lazy-loading, srcset, etc.)."""
        s = _summarizer()
        post = _post("a", content='<img data-src="https://cdn.example.com/lazy.png">')
        post.images = []
        s.summarizer = _FakeChain(
            [
                {
                    "summary": '<img src="https://cdn.example.com/lazy.png">body',
                    "tags": [],
                    "urls": [],
                }
            ]
        )
        s._summarize_posts([post])
        assert "lazy.png" in post.summary


class TestProcessPostsCapBeforeSummarize:
    """max_posts must be applied BEFORE summarization so we don't pay to
    summarize posts that get discarded, and only the top-N are summarized."""

    def test_only_top_n_summarized(self):
        s = _summarizer(min_score=0.0, max_posts=2)
        summarized: list[str] = []

        # Spy: record which posts reach summarization.
        def fake_summarize(posts):
            for p in posts:
                p.summary = f"summary of {p.title}"
            summarized.extend(p.title for p in posts)
            return list(posts)

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
        result = s.process_posts(posts)
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
        assert s.process_posts(posts) == []
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
        result = s.process_posts(posts)
        assert [p.title for p in result] == ["ok"]
        assert any(
            r == "Summarization failed (no model response)."
            for _p, r in s.filtered_out_posts
        )

    def test_preexisting_summary_text_cannot_fake_success(self):
        """Regression: a post that already has text in ``summary`` (as every RSS
        post did, seeded from the feed teaser) must NOT be treated as
        successfully summarized when the model call fails — otherwise it ships
        the teaser as the article body while also being reported as dropped."""
        s = _summarizer(min_score=0.0)
        post = _post("a")
        post.summary = "Feed teaser text that is not a real summary."
        s.filter = _FakeChain([{"score": "0.90", "reason": "", "title": ""}])
        s.summarizer = _FakeChain([None])
        assert s.process_posts([post]) == []


class TestSelectForDigest:
    """Line-up selection: relevance order, and the optional per-source cap."""

    def _scored(self, spec: list[tuple[str, str, float]]) -> list[Post]:
        posts = []
        for title, source, score in spec:
            p = _post(title, source=source)
            p.score = score
            posts.append(p)
        return posts

    def test_ranked_by_score_descending(self):
        s = _summarizer()
        posts = self._scored(
            [("low", "aws", 0.5), ("top", "meta", 0.9), ("mid", "google", 0.7)]
        )
        assert [p.title for p in s._select_for_digest(posts)] == ["top", "mid", "low"]

    def test_cap_prefers_diverse_sources(self):
        s = _summarizer(max_posts=3, max_per_source=2)
        posts = self._scored(
            [
                ("aws1", "aws", 0.90),
                ("aws2", "aws", 0.88),
                ("aws3", "aws", 0.86),
                ("meta1", "meta", 0.80),
            ]
        )
        chosen = [p.title for p in s._select_for_digest(posts)]
        # The third AWS post is displaced by the best post from another source.
        assert chosen == ["aws1", "aws2", "meta1"]

    def test_cap_backfills_rather_than_shrinking_the_issue(self):
        """A cap must never reduce how many articles ship — only which ones."""
        s = _summarizer(max_posts=3, max_per_source=1)
        posts = self._scored(
            [("aws1", "aws", 0.90), ("aws2", "aws", 0.88), ("aws3", "aws", 0.86)]
        )
        chosen = [p.title for p in s._select_for_digest(posts)]
        assert chosen == ["aws1", "aws2", "aws3"]

    def test_no_cap_keeps_pure_score_order(self):
        s = _summarizer(max_posts=2)
        posts = self._scored(
            [("aws1", "aws", 0.90), ("aws2", "aws", 0.88), ("meta1", "meta", 0.80)]
        )
        assert [p.title for p in s._select_for_digest(posts)] == ["aws1", "aws2"]


class TestFilterInputIsProseNotMarkup:
    """Filtering is ~74% of this pipeline's Bedrock cost and its input was raw
    HTML, of which ~87% was markup on a real run (77,417 chars of HTML around
    5,669 chars of article). The rubric scores topic and quality, which live in
    the prose, so the markup was pure spend."""

    def _captured_inputs(self, summarizer, posts):
        captured: list[dict] = []

        class _Capture(_FakeChain):
            def batch(self, inputs, config=None):
                captured.extend(inputs)
                return [{"score": "0.90", "reason": "", "title": ""} for _ in inputs]

        summarizer.filter = _Capture([])
        summarizer._filter_posts(posts)
        return captured

    def test_markup_stripped_from_filter_input(self):
        s = _summarizer(min_score=0.0)
        html = (
            '<div class="nav"><script>var a=1;</script>'
            "<style>.x{color:red}</style><p>Real article prose here.</p></div>"
        )
        captured = self._captured_inputs(s, [_post("a", content=html)])
        sent = captured[0]["post"]
        assert "Real article prose here." in sent
        assert "<script>" not in sent and "class=" not in sent
        assert len(sent) < len(html)

    def test_summarization_still_receives_full_html(self):
        """Only the FILTER gets prose; the summarizer needs the markup for code
        blocks, tables and image URLs."""
        s = _summarizer(min_score=0.0)
        html = '<p>Body</p><img src="https://cdn.example.com/x.png">'
        post = _post("a", content=html)
        captured: list[dict] = []

        class _Capture(_FakeChain):
            def batch(self, inputs, config=None):
                captured.extend(inputs)
                return [{"summary": "ok", "tags": [], "urls": []}]

        s.summarizer = _Capture([])
        s._summarize_posts([post])
        assert captured[0]["post"] == html

    def test_flag_restores_raw_html_for_filtering(self):
        s = _summarizer(min_score=0.0, filter_on_visible_text=False)
        html = "<p>Prose</p>"
        captured = self._captured_inputs(s, [_post("a", content=html)])
        assert captured[0]["post"] == html

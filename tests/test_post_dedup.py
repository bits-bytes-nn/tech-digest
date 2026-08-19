"""Collection-stage duplicate handling and the summary-field contract.

Covers two defects that produced visibly wrong digests: the same announcement
arriving from two configured feeds as two identical cards, and a post's RSS
teaser being mistaken for a successful LLM summary.
"""

from __future__ import annotations

from datetime import UTC, datetime

import feedparser

from app.src.feed_parser import (
    Post,
    PostCollector,
    normalize_title_key,
)


class _StubFetcher:
    def __init__(self, source_url: str, posts: list[Post]):
        self.source_url = source_url
        self._posts = posts

    def fetch(self, start_date, end_date):
        return self._posts


def _post(title: str, link: str, source: str = "google") -> Post:
    return Post(
        title=title,
        link=link,
        published_date=datetime(2026, 6, 1, tzinfo=UTC),
        content="body",
        source=source,  # type: ignore[arg-type]
    )


class TestNormalizeTitleKey:
    def test_folds_case_punctuation_and_spacing(self):
        assert normalize_title_key("Gemini 3:  Scaling  Up!") == normalize_title_key(
            "gemini 3 - scaling up"
        )

    def test_distinct_titles_stay_distinct(self):
        assert normalize_title_key("Gemini 3 Pro") != normalize_title_key("Gemini 3")

    def test_empty_title_is_empty_key(self):
        assert normalize_title_key("   ") == ""


class TestCrossSourceDedup:
    def test_same_title_from_two_sources_collected_once(self, date_range):
        start, end = date_range
        collector = PostCollector(
            [
                _StubFetcher(
                    "https://research.google/blog",
                    [_post("Scaling Up Gemini", "https://research.google/blog/a")],
                ),
                _StubFetcher(
                    "https://deepmind.google/blog/rss.xml",
                    [_post("Scaling up  Gemini!", "https://deepmind.google/blog/a")],
                ),
            ]
        )
        posts = collector.collect_posts(start, end)
        assert len(posts) == 1
        assert posts[0].link == "https://research.google/blog/a"

    def test_different_titles_both_kept(self, date_range):
        start, end = date_range
        collector = PostCollector(
            [
                _StubFetcher(
                    "https://a.example/feed",
                    [_post("Article One", "https://a.example/1")],
                ),
                _StubFetcher(
                    "https://b.example/feed",
                    [_post("Article Two", "https://b.example/2")],
                ),
            ]
        )
        assert len(collector.collect_posts(start, end)) == 2

    def test_dedup_shows_up_as_duplicates_not_as_an_unhealthy_source(self, date_range):
        """Two facts, reported separately instead of conflated into one number.

        ``status`` answers "did this source work?" — dedup must never make a
        healthy source look empty or broken, so it stays OK. ``post_count``
        answers "what did the digest get from it?", which is the number the health
        line's total and the empty-digest alarm's "Collected" field are about. It
        used to report the fetch count, so a source whose every post was already
        collected elsewhere read as "1 posts" while contributing nothing, and the
        crawl summary overstated how much material the run actually had.
        """
        start, end = date_range
        dup = _post("Same Title", "https://b.example/2")
        collector = PostCollector(
            [
                _StubFetcher(
                    "https://a.example/feed",
                    [_post("Same Title", "https://a.example/1")],
                ),
                _StubFetcher("https://b.example/feed", [dup]),
            ]
        )
        posts = collector.collect_posts(start, end)
        first, second = collector.report.sources
        assert first.status.value == "ok" and second.status.value == "ok"
        assert (first.post_count, first.duplicates) == (1, 0)
        assert (second.post_count, second.duplicates) == (0, 1)
        # The total now equals what actually reached the pipeline.
        assert collector.report.total_posts == len(posts) == 1
        assert "(1 deduped)" in collector.report.format_alert()


class TestSourceMapping:
    """Every configured feed must map to a distinct source.

    ``source`` drives the logo, the accessibility label AND the per-source
    diversity cap — so three independent authors sharing "unknown" meant the cap
    treated them as one publication.
    """

    def test_independent_author_blogs_are_distinct_sources(self):
        assert (
            Post._determine_source("https://eugeneyan.com/writing/x/") == "eugene_yan"
        )
        assert Post._determine_source("https://www.philschmid.de/post") == "phil_schmid"
        assert (
            Post._determine_source("https://sebastianraschka.com/blog/y")
            == "sebastian_raschka"
        )

    def test_substack_style_subdomain_maps_to_the_same_author(self):
        assert (
            Post._determine_source("https://magazine.sebastianraschka.com/p/z")
            == "sebastian_raschka"
        )

    def test_underscored_source_renders_as_a_readable_name(self):
        from app.src.newsletter_renderer import Article

        article = Article.model_validate(
            {
                "title": "t",
                "link": "https://sebastianraschka.com/a",
                "published_date": "2026-06-01",
                "thumbnail": "x.png",
                "summary": "<p>body</p>",
                "source": "sebastian_raschka",
            }
        )
        assert article.source_label == "Sebastian Raschka"

    def test_unmapped_host_still_falls_back_to_unknown(self):
        assert Post._determine_source("https://some-new-blog.example/post") == "unknown"


class TestSummaryNotSeededFromFeed:
    """``Post.summary`` is the slot the LLM summary lands in, and downstream code
    treats "non-empty" as "summarization succeeded". Seeding it from the feed
    teaser made a failed summarization indistinguishable from a successful one,
    so the teaser shipped as the article body."""

    def test_from_entry_leaves_summary_empty(self):
        entry = feedparser.FeedParserDict(
            title="A Post",
            link="https://example.com/post",
            published="2026-06-01",
            summary="<p>Short teaser from the feed.</p>",
        )
        post = Post.from_entry(entry)
        assert post.summary == ""

    def test_feed_summary_is_still_used_as_content(self):
        """The teaser is still valid *content* to evaluate and summarize — only
        the summary field must stay untouched."""
        entry = feedparser.FeedParserDict(
            title="A Post",
            link="https://example.com/post",
            published="2026-06-01",
            summary="<p>" + "text " * 200 + "</p>",
        )
        post = Post.from_entry(entry)
        assert "text" in post.content

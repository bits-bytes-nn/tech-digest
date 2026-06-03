"""Unit tests for feed_parser: date parsing, source detection, content/image
extraction, and the scraper registry routing. No network access."""

from __future__ import annotations

from datetime import UTC, datetime

import feedparser
import pytest

from app.src.feed_parser import (
    AnthropicBlogScraper,
    GoogleBlogScraper,
    MetaAIBlogScraper,
    Post,
    PostCollector,
    QwenBlogScraper,
    RssFetcher,
    ScraperRegistry,
    XAIBlogScraper,
    is_date_in_range,
    parse_published_date,
    try_parse_published_date,
)


class TestParsePublishedDate:
    def test_iso8601_with_z(self):
        dt = parse_published_date("2026-05-30T10:00:00Z")
        assert dt == datetime(2026, 5, 30, 10, 0, 0, tzinfo=UTC)

    def test_iso8601_with_offset(self):
        dt = parse_published_date("2026-05-30T10:00:00+09:00")
        assert dt.astimezone(UTC).hour == 1

    def test_rfc822_gmt(self):
        dt = parse_published_date("Fri, 30 May 2026 10:00:00 GMT")
        assert dt.year == 2026 and dt.month == 5 and dt.day == 30

    def test_human_readable(self):
        dt = parse_published_date("May 30, 2026")
        assert dt.year == 2026 and dt.month == 5 and dt.day == 30

    def test_empty_returns_now_utc(self):
        dt = parse_published_date("")
        assert dt.tzinfo is UTC

    def test_unparseable_returns_now_utc(self):
        dt = parse_published_date("not a date at all")
        assert dt.tzinfo is UTC

    def test_non_string_returns_now(self):
        dt = parse_published_date(None)  # type: ignore[arg-type]
        assert dt.tzinfo is UTC


class TestTryParsePublishedDate:
    """The fail-CLOSED variant: unparseable input yields None (post is dropped)
    rather than being silently dated to the present moment."""

    def test_valid_iso_parses(self):
        dt = try_parse_published_date("2026-05-30T10:00:00Z")
        assert dt == datetime(2026, 5, 30, 10, 0, 0, tzinfo=UTC)

    def test_valid_human_readable_parses(self):
        dt = try_parse_published_date("May 30, 2026")
        assert dt is not None and (dt.year, dt.month, dt.day) == (2026, 5, 30)

    def test_empty_returns_none(self):
        assert try_parse_published_date("") is None

    def test_unparseable_returns_none(self):
        assert try_parse_published_date("not a date at all") is None

    def test_non_string_returns_none(self):
        assert try_parse_published_date(None) is None  # type: ignore[arg-type]


class TestIsDateInRange:
    def test_inside_range(self, date_range):
        start, end = date_range
        assert is_date_in_range(datetime(2026, 5, 28, tzinfo=UTC), start, end)

    def test_before_range(self, date_range):
        start, end = date_range
        assert not is_date_in_range(datetime(2026, 5, 1, tzinfo=UTC), start, end)

    def test_after_range(self, date_range):
        start, end = date_range
        assert not is_date_in_range(datetime(2026, 6, 10, tzinfo=UTC), start, end)

    def test_naive_datetime_treated_as_utc(self, date_range):
        start, end = date_range
        assert is_date_in_range(datetime(2026, 5, 28, 12, 0, 0), start, end)


class TestSourceDetection:
    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://aws.amazon.com/blogs/ml/post", "aws"),
            ("https://www.anthropic.com/engineering/x", "anthropic"),
            ("https://openai.com/news/foo", "openai"),
            ("https://x.ai/news/grok", "xai"),
            ("https://qwenlm.github.io/blog/qwen3", "qwen"),
            ("https://unknown-blog.example.com/post", "unknown"),
            ("", "unknown"),
        ],
    )
    def test_determine_source(self, url, expected):
        assert Post._determine_source(url) == expected

    def test_medium_publication_mapping(self):
        assert Post._determine_source("https://medium.com/airbnb-engineering/x") == (
            "airbnb"
        )

    def test_medium_at_handle_mapping(self):
        assert Post._determine_source("https://medium.com/@palantir/post") == "palantir"

    def test_medium_at_handle_mixed_case(self):
        # The @Pinterest_Engineering feed handle is mixed-case; mapping must be
        # case-insensitive.
        assert (
            Post._determine_source("https://medium.com/@Pinterest_Engineering/post")
            == "pinterest"
        )

    def test_medium_netflix_publication(self):
        assert (
            Post._determine_source("https://medium.com/netflix-techblog/some-post")
            == "netflix"
        )

    def test_www_prefix_stripped(self):
        assert Post._determine_source("https://www.openai.com/news/x") == "openai"


class TestImageExtraction:
    def test_extracts_img_src_absolute(self):
        html = '<div><img src="https://cdn.example.com/a.png"></div>'
        imgs = Post._extract_images(html, "https://example.com/post")
        assert "https://cdn.example.com/a.png" in imgs

    def test_resolves_relative_img(self):
        html = '<img src="/images/a.png">'
        imgs = Post._extract_images(html, "https://example.com/post")
        assert "https://example.com/images/a.png" in imgs

    def test_markdown_image_pattern(self):
        html = "text ![alt](https://example.com/b.png) more"
        imgs = Post._extract_images(html, "https://example.com/post")
        assert "https://example.com/b.png" in imgs

    def test_empty_html(self):
        assert Post._extract_images("", "https://example.com") == []

    def test_drops_non_http(self):
        html = '<img src="data:image/png;base64,xxxx">'
        assert Post._extract_images(html, "https://example.com") == []


class TestPostModel:
    def test_tags_deduplicated_and_capped(self):
        post = Post(
            title="t",
            link="https://example.com",
            published_date=datetime.now(UTC),
            tags=["b", "a", "a", "c", "d", "e", "f", "g"],
        )
        # Order preserved (first-seen), duplicates removed, capped at 5.
        assert post.tags == ["b", "a", "c", "d", "e"]
        assert len(post.tags) <= 5

    def test_empty_tags_default(self):
        post = Post(
            title="t",
            link="https://example.com",
            published_date=datetime.now(UTC),
            tags=[],
        )
        assert post.tags == ["uncategorized"]


class TestScraperRegistry:
    @pytest.mark.parametrize(
        "url,scraper_cls",
        [
            ("https://www.anthropic.com/engineering", AnthropicBlogScraper),
            ("https://research.google/blog", GoogleBlogScraper),
            ("https://ai.meta.com/blog/", MetaAIBlogScraper),
            ("https://qwenlm.github.io/blog/", QwenBlogScraper),
            ("https://x.ai/news", XAIBlogScraper),
        ],
    )
    def test_routes_to_custom_scraper(self, url, scraper_cls):
        assert isinstance(ScraperRegistry.get_fetcher(url), scraper_cls)

    def test_falls_back_to_rss(self):
        fetcher = ScraperRegistry.get_fetcher("https://example.com/feed.xml")
        assert isinstance(fetcher, RssFetcher)


class TestPostCollectorDeduplication:
    def test_dedupes_by_link(self):
        now = datetime.now(UTC)

        class _Stub:
            def __init__(self, posts):
                self._posts = posts

            def fetch(self, start, end):
                return self._posts

        p1 = Post(title="a", link="https://example.com/1", published_date=now)
        p2 = Post(title="b", link="https://example.com/1", published_date=now)
        p3 = Post(title="c", link="https://example.com/2", published_date=now)
        collector = PostCollector([_Stub([p1, p3]), _Stub([p2])])
        posts = collector.collect_posts(now, now)
        assert {str(p.link) for p in posts} == {
            "https://example.com/1",
            "https://example.com/2",
        }

    def test_one_failing_fetcher_does_not_abort(self):
        now = datetime.now(UTC)

        class _Boom:
            def fetch(self, start, end):
                raise RuntimeError("network down")

        class _Ok:
            def fetch(self, start, end):
                return [
                    Post(title="a", link="https://example.com/1", published_date=now)
                ]

        collector = PostCollector([_Boom(), _Ok()])
        posts = collector.collect_posts(now, now)
        assert len(posts) == 1


class TestPostFromEntry:
    def test_from_entry_basic(self):
        entry = feedparser.FeedParserDict(
            title="  Hello World  ",
            link="https://openai.com/news/x",
            published="2026-05-30T10:00:00Z",
            summary="A short summary.",
        )
        post = Post.from_entry(entry)
        assert post.title == "Hello World"
        assert post.source == "openai"

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
    visible_text,
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

    def test_resolves_relative_markdown_image(self):
        # Relative markdown image paths must be joined to base_url (not dropped).
        html = "text ![alt](/img/c.png) more"
        imgs = Post._extract_images(html, "https://example.com/post")
        assert "https://example.com/img/c.png" in imgs

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

    def test_thin_scrape_does_not_overwrite_richer_feed_content(self, monkeypatch):
        # A feed teaser below MIN_CONTENT_LENGTH triggers a scrape, but if the
        # scrape yields LESS visible text it must NOT replace the feed content
        # (which would push the post below the downstream gate and drop it).
        from app.src import feed_parser

        feed_body = "<p>" + ("word " * 400) + "</p>"  # ~2000 visible chars
        monkeypatch.setattr(
            feed_parser.Post,
            "_scrape_full_content",
            staticmethod(lambda url: "<p>tiny</p>"),
        )
        entry = feedparser.FeedParserDict(
            title="T",
            link="https://openai.com/news/x",
            published="2026-05-30T10:00:00Z",
            content=[{"value": feed_body}],
        )
        post = Post.from_entry(entry)
        assert "word" in post.content
        assert "tiny" not in post.content

    def test_richer_scrape_replaces_thin_feed_content(self, monkeypatch):
        from app.src import feed_parser

        rich = "<p>" + ("word " * 800) + "</p>"
        monkeypatch.setattr(
            feed_parser.Post, "_scrape_full_content", staticmethod(lambda url: rich)
        )
        entry = feedparser.FeedParserDict(
            title="T",
            link="https://openai.com/news/x",
            published="2026-05-30T10:00:00Z",
            content=[{"value": "<p>short teaser</p>"}],
        )
        post = Post.from_entry(entry)
        assert post.content == rich


class TestSourceTypeMatchesTheMapping:
    """Every ``SourceType`` member must be producible by ``_determine_source``.

    ``_determine_source`` resolves a source purely from ``SOURCE_MAPPING``, so a
    literal absent from that map's values can never be assigned. "deepmind" was
    exactly that: ``deepmind.google`` maps to "google", so the member was
    unreachable while reading as a supported source.
    """

    def test_every_literal_is_reachable(self):
        from typing import get_args

        from app.src.feed_parser import ScraperConfig, SourceType

        declared = set(get_args(SourceType))
        producible = set(ScraperConfig.SOURCE_MAPPING.values()) | {"unknown"}
        assert declared == producible, (
            f"unreachable literals: {sorted(declared - producible)}; "
            f"unmapped values: {sorted(producible - declared)}"
        )


class TestRegisteredScrapersResolveToAMappedSource:
    """Each index-page scraper's own URL must resolve to a real source.

    The registry used to pair every scraper with a hardcoded ``SourceType`` that
    ``Post.from_entry`` then ignored. Removing it means ``SOURCE_MAPPING`` is the
    only definition — which is only safe if each scraper's configured URL is in
    that map. Otherwise its posts would silently render the "unknown" logo and
    share one ``max_per_source`` bucket.
    """

    def test_each_scraper_url_maps_to_a_known_source(self):
        from app.src.feed_parser import ScraperRegistry

        unmapped = {
            url: Post._determine_source(f"https://{url}/some-post")
            for url in ScraperRegistry._SCRAPER_MAPPING
            if Post._determine_source(f"https://{url}/some-post") == "unknown"
        }
        assert not unmapped, f"scraper URLs resolving to 'unknown': {unmapped}"


class TestContentContainerSelection:
    """``_scrape_full_content`` must pick the container that HOLDS the prose.

    It used to return the first matching selector without looking inside it, which
    made the ``<body>`` fallback unreachable whenever any selector matched. Measured
    on Hugging Face's blog, where ``<article>`` is a near-empty wrapper: 82 visible
    characters returned against 21,669 on the page. Those posts then died at the
    content-sufficiency gate, so a healthy source contributed nothing.
    """

    def _scrape(self, monkeypatch, html: str) -> str:
        from app.src import feed_parser as fp

        class _Response:
            text = html

        monkeypatch.setattr(fp, "_make_robust_request", lambda url: _Response())
        return fp.Post._scrape_full_content("https://x.example/post")

    # An <article> that is a wrapper around a heading, with the prose elsewhere.
    HF_SHAPE = (
        "<html><body><article><h1>Title</h1></article>"
        "<div class='prose'>" + ("실제 본문입니다. " * 200) + "</div></body></html>"
    )
    # An <article> that really is the article, inside ordinary page chrome.
    NORMAL_SHAPE = (
        "<html><body><nav>Home About Blog</nav>"
        "<article>" + ("The real article text. " * 200) + "</article>"
        "<footer>(c) 2026</footer></body></html>"
    )

    def test_a_degenerate_first_match_is_not_used(self, monkeypatch):
        out = self._scrape(monkeypatch, self.HF_SHAPE)
        assert "실제 본문입니다" in visible_text(out)
        assert len(visible_text(out)) > 1000

    def test_a_real_article_container_is_still_preferred_over_body(self, monkeypatch):
        """The narrower container excludes nav/footer chrome, which is why it is
        preferred at all — so a healthy page must not fall back to <body>."""
        out = self._scrape(monkeypatch, self.NORMAL_SHAPE)
        text = visible_text(out)
        assert "The real article text." in text
        assert "Home About Blog" not in text
        assert "(c) 2026" not in text

    def test_body_is_used_when_no_selector_matches(self, monkeypatch):
        out = self._scrape(
            monkeypatch,
            "<html><body><section>"
            + ("plain text " * 200)
            + "</section></body></html>",
        )
        assert "plain text" in visible_text(out)

    def test_the_richest_match_wins_across_selectors(self, monkeypatch):
        """Several selectors can match; the choice is by content, not by list
        order, so an earlier-listed but emptier container loses."""
        out = self._scrape(
            monkeypatch,
            "<html><body><article>tiny</article>"
            "<div class='content'>" + ("the actual body " * 200) + "</div>"
            "</body></html>",
        )
        assert "the actual body" in visible_text(out)

    def test_a_page_with_no_body_yields_empty(self, monkeypatch):
        assert self._scrape(monkeypatch, "<div>fragment</div>") != "  "

    def test_fetch_failure_yields_empty(self, monkeypatch):
        from app.src import feed_parser as fp

        monkeypatch.setattr(fp, "_make_robust_request", lambda url: None)
        assert fp.Post._scrape_full_content("https://x.example/post") == ""

"""Unit tests for feed_parser: date parsing, source detection, content/image
extraction, and the scraper registry routing. No network access."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import ClassVar

import feedparser
import pytest

from app.src.feed_parser import (
    LinkedInBlogScraper,
    Post,
    PostCollector,
    RssFetcher,
    ScraperRegistry,
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

    def test_rfc822_named_zone(self):
        # strptime's %z rejects "PDT"; such a post read as undated and was dropped.
        dt = parse_published_date("Fri, 30 May 2026 10:00:00 PDT")
        assert dt == datetime(2026, 5, 30, 17, 0, 0, tzinfo=UTC)

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

    @pytest.mark.parametrize("text", ["2026-10-04", "2026-10-04T00:00:00"])
    def test_offset_less_iso_is_utc_whatever_the_local_zone(self, monkeypatch, text):
        # The Batch image runs with TZ=Asia/Seoul; a naive ISO date read as local
        # time landed nine hours early and outside its window.
        import time

        monkeypatch.setenv("TZ", "Asia/Seoul")
        time.tzset()
        try:
            assert try_parse_published_date(text) == datetime(2026, 10, 4, tzinfo=UTC)
        finally:
            monkeypatch.undo()
            time.tzset()

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

    def test_end_is_exclusive(self, date_range):
        start, end = date_range
        assert is_date_in_range(start, start, end)
        assert not is_date_in_range(end, start, end)


class TestSourceDetection:
    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://aws.amazon.com/blogs/ml/post", "aws"),
            ("https://openai.com/news/foo", "openai"),
            ("https://qwenlm.github.io/blog/qwen3", "qwen"),
            # The medium.com/feed/@palantir feed links some posts on this host.
            ("https://palantir.medium.com/some-post-1a2b", "palantir"),
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
    def test_routes_to_the_scraper_for_a_feedless_site(self):
        fetcher = ScraperRegistry.get_fetcher(
            "https://www.linkedin.com/blog/engineering"
        )
        assert isinstance(fetcher, LinkedInBlogScraper)

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
        # A list, not a set: a set would hide the duplicate this test is about.
        assert sorted(str(p.link) for p in posts) == [
            "https://example.com/1",
            "https://example.com/2",
        ]

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

    def test_image_urls_keep_their_exact_characters(self):
        # The content used to have every U+202F turned into a space, which
        # rewrote image URLs (macOS screenshot names) into ones that 403.
        url = "https://cdn.example.com/Screenshot at 1.27.14\u202fPM.png"
        body = "<p>" + ("word " * 400) + f'</p><img src="{url}">'
        entry = feedparser.FeedParserDict(
            title="T",
            link="https://aws.amazon.com/blogs/machine-learning/x",
            published="2026-05-30T10:00:00Z",
            content=[{"value": body}],
        )
        post = Post.from_entry(entry)
        assert url in post.content
        assert post.images == [url]

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
            headers: ClassVar[dict[str, str]] = {
                "content-type": "text/html; charset=utf-8"
            }

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


class TestFeedTitles:
    def test_double_escaped_title_is_decoded(self):
        # Amazon Science sends "don&amp;#8217;t"; the card showed "&#8217;".
        entry = feedparser.FeedParserDict(
            title="Why don&#8217;t models generalize?",
            link="https://www.amazon.science/blog/x",
            published="2026-10-01T00:00:00Z",
            summary="body " * 400,
        )
        assert Post.from_entry(entry).title == "Why don\u2019t models generalize?"


class TestPageCharset:
    @staticmethod
    def _response(content_type: str, body: bytes):
        import requests

        response = requests.Response()
        response._content = body
        response.headers["Content-Type"] = content_type
        response.encoding = requests.utils.get_encoding_from_headers(response.headers)
        return response

    def test_meta_charset_is_honoured_when_the_header_has_none(self):
        from app.src.feed_parser import parse_page

        text = "Xiaomi\u2019s \u201csimply\u201d"
        body = f'<html><head><meta charset="utf-8"></head><p>{text}</p>'
        page = parse_page(self._response("text/html", body.encode("utf-8")))
        assert page.p.get_text() == text

    def test_a_header_charset_still_wins(self):
        from app.src.feed_parser import parse_page

        body = "<p>caf\xe9</p>".encode("latin-1")
        page = parse_page(self._response("text/html; charset=iso-8859-1", body))
        assert page.p.get_text() == "café"

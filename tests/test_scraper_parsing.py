"""Golden parse tests for the per-site HTML scrapers.

These scrapers carry the most fragile heuristics in the codebase (CSS selectors,
title-length bounds, date regex, parent-walk depth). They previously had ZERO
unit coverage — only routing/source-detection was tested. Here we feed each
scraper a saved, representative HTML fixture and pin the parsed output, so a
site-layout change or a selector regression is caught offline.

Network is fully stubbed: ``_fetch_page`` returns the fixture soup and
``_make_robust_request`` is neutralized so the full-article scrape that
``Post.from_entry`` triggers never hits the wire.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from app.src import feed_parser
from app.src.feed_parser import (
    AnthropicBlogScraper,
    GoogleBlogScraper,
    LinkedInBlogScraper,
    MetaAIBlogScraper,
    QwenBlogScraper,
    XAIBlogScraper,
)

FIXTURES = Path(__file__).parent / "fixtures" / "scrapers"


def _load(name: str) -> BeautifulSoup:
    return BeautifulSoup((FIXTURES / name).read_text(encoding="utf-8"), "html.parser")


@pytest.fixture
def window() -> tuple[datetime, datetime]:
    """A window covering late-May 2026 (the recent fixture posts), excluding the
    deliberately-ancient entries in each fixture."""
    start = datetime(2026, 5, 20, tzinfo=UTC)
    end = datetime(2026, 5, 31, 23, 59, 59, tzinfo=UTC)
    return start, end


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    # The scrapers build Posts via Post.from_entry, which scrapes the full
    # article when the feed body is thin. Neutralize the HTTP layer so no
    # request is made and content stays empty.
    monkeypatch.setattr(feed_parser, "_make_robust_request", lambda url: None)


def _bind(scraper_cls, fixture: str, page_url: str, monkeypatch):
    scraper = scraper_cls(page_url=page_url)
    soup = _load(fixture)
    monkeypatch.setattr(scraper, "_fetch_page", lambda: soup)
    return scraper


class TestGoogleScraper:
    def test_parses_titles_links_dates(self, monkeypatch, window):
        scraper = _bind(
            GoogleBlogScraper,
            "google.html",
            "https://research.google/blog",
            monkeypatch,
        )
        posts = scraper.fetch(*window)
        titles = {p.title for p in posts}
        assert "Scaling Transformers Efficiently" in titles
        assert "A Post With An Absolute URL" in titles
        # Out-of-window 2020 post excluded.
        assert "An Old Post Outside The Window" not in titles

    def test_relative_href_resolved_absolute(self, monkeypatch, window):
        scraper = _bind(
            GoogleBlogScraper,
            "google.html",
            "https://research.google/blog",
            monkeypatch,
        )
        posts = scraper.fetch(*window)
        links = {p.link for p in posts}
        assert "https://research.google/blog/scaling-transformers-efficiently/" in links


class TestLinkedInScraper:
    def test_parses_items_and_window(self, monkeypatch, window):
        scraper = _bind(
            LinkedInBlogScraper,
            "linkedin.html",
            "https://www.linkedin.com/blog/engineering",
            monkeypatch,
        )
        posts = scraper.fetch(*window)
        titles = {p.title for p in posts}
        assert "Real-Time Ranking At Scale" in titles
        assert "Building A Feature Store" in titles
        assert "Ancient Post Outside Window" not in titles

    def test_relative_href_resolved_absolute(self, monkeypatch, window):
        # A root-relative LinkedIn href must be resolved against page_url (like
        # every other scraper) so the Post link is absolute — otherwise source
        # resolves to 'unknown' and the full-content scrape refuses the empty
        # host.
        html = (
            '<li class="post-list__item">'
            '<p class="grid-post__date">May 25, 2026</p>'
            '<a class="grid-post__link" href="/blog/engineering/rel-post">'
            "Relative Link Post</a></li>"
        )
        scraper = LinkedInBlogScraper(
            page_url="https://www.linkedin.com/blog/engineering"
        )
        soup = BeautifulSoup(html, "html.parser")
        monkeypatch.setattr(scraper, "_fetch_page", lambda: soup)
        posts = scraper.fetch(*window)
        assert posts
        assert posts[0].link.startswith("https://www.linkedin.com/blog/engineering/")
        assert posts[0].source == "linkedin"


class TestQwenScraper:
    def test_parses_and_skips_nav(self, monkeypatch, window):
        scraper = _bind(
            QwenBlogScraper,
            "qwen.html",
            "https://qwenlm.github.io/blog/",
            monkeypatch,
        )
        posts = scraper.fetch(*window)
        titles = {p.title for p in posts}
        assert "Qwen3 Release Notes" in titles
        # Nav links ("Blog Home", "Next Page") must not become posts.
        assert "Blog Home" not in titles
        assert not any("Next Page" in t for t in titles)


class TestQwenNavWordBoundary:
    """Nav-keyword filtering must match whole words, not substrings, so ML
    titles that merely CONTAIN a nav word ('Preventing' -> 'prev', 'PageRank'
    -> 'page', 'Continued' -> 'continue') are not silently dropped."""

    def _parse_title(self, title: str, monkeypatch):
        scraper = QwenBlogScraper(page_url="https://qwenlm.github.io/blog/")
        # The anchor text is the title; the date lives as a sibling in the
        # parent (where _parse_item searches via str(item.parent)).
        html = (
            f'<div><a href="/blog/some-post/">{title}</a>'
            f"<span>May 25, 2026</span></div>"
        )
        item = BeautifulSoup(html, "html.parser").find("a")
        result = scraper._parse_item(item)
        return None if result is None else result.get("title")

    @pytest.mark.parametrize(
        "title",
        [
            "Preventing Reward Hacking in RLHF",
            "Continued Pretraining of Qwen3",
            "Improving PageRank with LLMs",
        ],
    )
    def test_ml_titles_with_nav_substrings_survive(self, title, monkeypatch):
        assert self._parse_title(title, monkeypatch) == title

    @pytest.mark.parametrize(
        "title", ["Next Page", "Previous", "Read more", "Continue reading"]
    )
    def test_real_nav_strings_still_dropped(self, title, monkeypatch):
        assert self._parse_title(title, monkeypatch) is None


class TestMetaScraper:
    def test_parses_blog_cards_and_window(self, monkeypatch, window):
        scraper = _bind(
            MetaAIBlogScraper,
            "meta.html",
            "https://ai.meta.com/blog",
            monkeypatch,
        )
        posts = scraper.fetch(*window)
        titles = {p.title for p in posts}
        assert "Llama 4 Multimodal Research" in titles
        assert "Segment Anything 3" in titles
        # Old post and nav links excluded.
        assert "Old Research Post" not in titles

    def test_links_resolved_against_page_url(self, monkeypatch, window):
        scraper = _bind(
            MetaAIBlogScraper,
            "meta.html",
            "https://ai.meta.com/blog",
            monkeypatch,
        )
        posts = scraper.fetch(*window)
        assert all(p.link.startswith("https://ai.meta.com/blog/") for p in posts)


class TestXAIScraper:
    def test_parses_news_and_window(self, monkeypatch, window):
        scraper = _bind(
            XAIBlogScraper,
            "xai.html",
            "https://x.ai/news",
            monkeypatch,
        )
        posts = scraper.fetch(*window)
        titles = {p.title for p in posts}
        assert "Grok 4 Launch" in titles
        assert "Colossus Cluster Scaling" in titles
        assert "Ancient Announcement" not in titles

    def test_links_resolved_against_page_url(self, monkeypatch, window):
        scraper = _bind(
            XAIBlogScraper,
            "xai.html",
            "https://x.ai/news",
            monkeypatch,
        )
        posts = scraper.fetch(*window)
        assert all(p.link.startswith("https://x.ai/news/") for p in posts)


class TestAnthropicScraper:
    def test_parses_engineering_index_and_window(self, monkeypatch, window):
        scraper = _bind(
            AnthropicBlogScraper,
            "anthropic.html",
            "https://www.anthropic.com/engineering",
            monkeypatch,
        )
        posts = scraper.fetch(*window)
        titles = {p.title for p in posts}
        assert "Building Effective Agents" in titles
        assert "Prompt Caching Internals" in titles
        assert "Legacy Systems Post" not in titles

    def test_links_resolved_absolute(self, monkeypatch, window):
        scraper = _bind(
            AnthropicBlogScraper,
            "anthropic.html",
            "https://www.anthropic.com/engineering",
            monkeypatch,
        )
        posts = scraper.fetch(*window)
        assert all(
            p.link.startswith("https://www.anthropic.com/engineering/") for p in posts
        )


class TestDateAttribution:
    """Which post a date belongs to.

    The walk-outward search used to regex the whole ancestor once it left the
    link, and an ancestor holding several posts yields the FIRST date in document
    order. An undated post therefore inherited its neighbour's date and entered
    the weekly window carrying a date the site never gave it — which defeats the
    fail-closed date gate ``try_parse_published_date`` exists to provide.

    The fix removes the other post links before searching, so a surviving date is
    either ours or genuinely shared chrome. Validated against live HTML from all
    three sites (anthropic 9 posts/8 dates, meta 3/3, xai 49/41, every xai date
    matching the one embedded in its own title).
    """

    def _links(self, scraper, html):
        soup = BeautifulSoup(html, "html.parser")
        return {
            a.get("href"): scraper._find_date_near_element(a)
            for a in soup.find_all("a", href=scraper.LINK_PATTERN)
        }

    @pytest.fixture
    def scraper(self):
        return AnthropicBlogScraper(page_url="https://www.anthropic.com/engineering")

    def test_undated_post_does_not_inherit_a_neighbours_date(self, scraper):
        found = self._links(
            scraper,
            """<div class="list">
              <a href="/engineering/dated"><h3>Dated</h3><span>May 30, 2026</span></a>
              <a href="/engineering/undated"><h3>Undated</h3></a>
            </div>""",
        )
        assert found["/engineering/dated"] == "May 30, 2026"
        assert found["/engineering/undated"] is None

    def test_genuinely_shared_section_date_still_applies_to_each_post(self, scraper):
        """A "Published <date>" heading above a group legitimately covers every
        post under it — the fix must not mistake that for a neighbour's date."""
        found = self._links(
            scraper,
            """<section><h2>Published May 30, 2026</h2>
              <a href="/engineering/alpha"><h3>Alpha</h3></a>
              <a href="/engineering/beta"><h3>Beta</h3></a>
            </section>""",
        )
        assert found == {
            "/engineering/alpha": "May 30, 2026",
            "/engineering/beta": "May 30, 2026",
        }

    def test_per_card_sibling_date_is_attributed_to_its_own_card(self, scraper):
        found = self._links(
            scraper,
            """<div class="list">
              <article><a href="/engineering/one"><h3>One</h3></a><time>May 30, 2026</time></article>
              <article><a href="/engineering/two"><h3>Two</h3></a><time>May 27, 2026</time></article>
            </div>""",
        )
        assert found == {
            "/engineering/one": "May 30, 2026",
            "/engineering/two": "May 27, 2026",
        }

    def test_date_inside_the_link_is_used(self, scraper):
        found = self._links(
            scraper,
            '<div><a href="/engineering/x"><h3>X</h3><span>May 27, 2026</span></a></div>',
        )
        assert found["/engineering/x"] == "May 27, 2026"

    def test_no_date_anywhere_returns_none_not_a_guess(self, scraper):
        found = self._links(
            scraper, '<div><a href="/engineering/x"><h3>X</h3></a></div>'
        )
        assert found["/engineering/x"] is None

    def test_all_index_scrapers_share_one_implementation(self):
        """The walker was copy-pasted into three scrapers, so a fix to one left
        the others wrong. They now inherit a single implementation."""
        from app.src.feed_parser import BasePageScraper

        for cls in (AnthropicBlogScraper, MetaAIBlogScraper, XAIBlogScraper):
            assert "_find_date_near_element" not in vars(cls), cls.__name__
            assert cls.LINK_PATTERN is not None, cls.__name__
            assert cls.DATE_PATTERNS, cls.__name__
        assert "_find_date_near_element" in vars(BasePageScraper)


class TestXaiTitleExtraction:
    """x.ai cards nest the whole teaser inside the anchor, so reading the card's
    full text produced titles like

        'Grok 4.6Aug 12, 2026Introducing Grok 4.6Aug 12, 2026IntroducingGrok
         4.6Grok 4.6 builds on Grok 4.5 with a particular focus on...'

    The heading inside the card is preferred instead. Verified against the live
    page: 18 posts, none with a title over 90 characters."""

    @pytest.fixture
    def scraper(self):
        return XAIBlogScraper(page_url="https://x.ai/news")

    def test_heading_wins_over_the_whole_card_text(self, scraper):
        link = BeautifulSoup(
            """<a href="/news/grok-4-6"><div><span>Grok 4.6</span>
              <span>Aug 12, 2026</span><h3>Grok 4.6 in GitHub Copilot</h3>
              <p>Grok 4.6 builds on Grok 4.5 with a particular focus on agents.</p>
            </div></a>""",
            "html.parser",
        ).find("a")
        assert scraper._extract_title(link) == "Grok 4.6 in GitHub Copilot"

    def test_falls_back_to_card_text_without_a_heading(self, scraper):
        link = BeautifulSoup(
            '<a href="/news/x"><div>Introducing Grok Bot</div></a>', "html.parser"
        ).find("a")
        assert scraper._extract_title(link) == "Introducing Grok Bot"

    def test_empty_heading_does_not_shadow_real_text(self, scraper):
        link = BeautifulSoup(
            '<a href="/news/x"><h3>  </h3><div>Real Title Here</div></a>',
            "html.parser",
        ).find("a")
        assert scraper._extract_title(link) == "Real Title Here"


class TestTitleExtractionIsShared:
    """Title extraction had four implementations that disagreed.

    x.ai preferred the heading inside the card (after emitting concatenated
    blobs); Meta read the card's full text first and so still had that bug;
    Anthropic filtered candidates with English content tests that meant nothing
    on any other site; Qwen kept its own nav-word list. The date walker had
    already been consolidated for exactly this reason — a fix applied to one copy
    left the others wrong.
    """

    def test_no_scraper_defines_its_own(self):
        from app.src.feed_parser import BasePageScraper

        for cls in (
            AnthropicBlogScraper,
            MetaAIBlogScraper,
            XAIBlogScraper,
            QwenBlogScraper,
        ):
            assert "_extract_title" not in vars(cls), cls.__name__
            assert "_should_filter_title" not in vars(cls), cls.__name__
        assert "_extract_title" in vars(BasePageScraper)

    @pytest.mark.parametrize(
        "cls,url",
        [
            (AnthropicBlogScraper, "https://www.anthropic.com/engineering"),
            (MetaAIBlogScraper, "https://ai.meta.com/blog"),
            (XAIBlogScraper, "https://x.ai/news"),
        ],
    )
    def test_heading_beats_the_whole_card_on_every_scraper(self, cls, url):
        """The bug that was fixed for x.ai only. Meta's cards nest the teaser
        inside the anchor, so reading the anchor's text produced
        'ResearchAug 12, 2026Segment Anything 3SAM 3 extends...' as the title."""
        scraper = cls(page_url=url)
        card = BeautifulSoup(
            '<a href="/blog/x"><div><span>Research</span><span>Aug 12, 2026</span>'
            "<h3>Segment Anything 3</h3><p>SAM 3 extends promptable segmentation "
            "to video.</p></div></a>",
            "html.parser",
        ).find("a")
        assert scraper._extract_title(card) == "Segment Anything 3"

    def test_no_credible_candidate_returns_empty_not_a_guess(self):
        """Fail closed, like the date gate: the caller skips the link rather than
        shipping a truncated blob as the title."""
        scraper = XAIBlogScraper(page_url="https://x.ai/news")
        link = BeautifulSoup('<a href="/news/x">Read more</a>', "html.parser").find("a")
        assert scraper._extract_title(link) == ""

    def test_labelled_element_wins_over_an_over_long_heading(self):
        scraper = XAIBlogScraper(page_url="https://x.ai/news")
        link = BeautifulSoup(
            f'<a href="/news/x"><h3>{"x" * 400}</h3>'
            '<span class="card-title">Short Real Title</span></a>',
            "html.parser",
        ).find("a")
        assert scraper._extract_title(link) == "Short Real Title"

    def test_when_every_candidate_is_over_long_nothing_is_shipped(self):
        """Previously the last resort truncated to 120 chars and appended "...",
        so a card with no heading shipped a sentence fragment as its title."""
        scraper = XAIBlogScraper(page_url="https://x.ai/news")
        link = BeautifulSoup(
            f'<a href="/news/x"><div>{"word " * 200}</div></a>', "html.parser"
        ).find("a")
        assert scraper._extract_title(link) == ""

    def test_aria_label_is_used_when_the_card_has_no_text(self):
        scraper = MetaAIBlogScraper(page_url="https://ai.meta.com/blog")
        link = BeautifulSoup(
            '<a href="/blog/x" aria-label="Llama 4 Multimodal Research">'
            '<img src="/x.png"></a>',
            "html.parser",
        ).find("a")
        assert scraper._extract_title(link) == "Llama 4 Multimodal Research"


class TestNavChromeRejection:
    """One rule replaces four: reject only when NO word names a subject."""

    @pytest.fixture
    def scraper(self):
        return QwenBlogScraper(page_url="https://qwenlm.github.io/blog/")

    @pytest.mark.parametrize(
        "text",
        [
            "Next Page",
            "Previous",
            "Read more",
            "Continue reading",
            "Blog Home",
            "The Latest",
            "View all posts",
            "«",
            "»",
            "   ",
            "...",
        ],
    )
    def test_chrome_is_rejected(self, scraper, text):
        assert scraper.is_nav_text(text)

    @pytest.mark.parametrize(
        "text",
        [
            # These were DROPPED before: Qwen rejected a title containing any nav
            # word, so a real post whose title happened to use one vanished.
            "Next Generation Reasoning Models",
            "More Efficient Attention",
            "Page-Level Retrieval for Long Context",
            "Read the Docs Integration for Qwen",
            # These were dropped by an even earlier substring version.
            "Preventing Reward Hacking in RLHF",
            "Continued Pretraining of Qwen3",
            "Improving PageRank with LLMs",
            # Short but real.
            "GPT-5.6",
        ],
    )
    def test_real_titles_survive(self, scraper, text):
        assert not scraper.is_nav_text(text)


class TestAccessibleNameHandling:
    """Validated against live HTML, which caught two regressions.

    Both Meta and Qwen have image-only card links whose ONLY signal is the
    accessible name — and it is written for screen readers, so it carries a
    call-to-action in front of the title: "Read Reimagining Independence: ...",
    "post link to Qwen3Guard: ...". Putting the attribute ahead of the link's own
    text therefore shipped titles with the CTA baked in.
    """

    @pytest.fixture
    def scraper(self):
        return MetaAIBlogScraper(page_url="https://ai.meta.com/blog")

    def test_visible_text_wins_over_the_accessible_name(self, scraper):
        """Meta emits both for the same post; the visible one is the title."""
        link = BeautifulSoup(
            '<a href="/blog/x" aria-label="Read Muse Spark 1.1">Muse Spark 1.1</a>',
            "html.parser",
        ).find("a")
        assert scraper._extract_title(link) == "Muse Spark 1.1"

    @pytest.mark.parametrize(
        "label,expected",
        [
            ("Read Reimagining Independence", "Reimagining Independence"),
            (
                "post link to Qwen3Guard: Real-time Safety",
                "Qwen3Guard: Real-time Safety",
            ),
            ("Learn more about Muse Video", "Muse Video"),
        ],
    )
    def test_call_to_action_stripped_from_an_image_only_card(
        self, scraper, label, expected
    ):
        link = BeautifulSoup(
            f'<a href="/blog/x" aria-label="{label}"><img src="/c.png"></a>',
            "html.parser",
        ).find("a")
        assert scraper._extract_title(link) == expected

    def test_a_cta_word_inside_a_title_is_not_touched(self, scraper):
        """Unlike the content blacklist this replaced, the pattern anchors at the
        start and requires trailing text, so it cannot eat a real title's words."""
        assert (
            scraper._strip_call_to_action("Reading Comprehension in LLMs")
            == "Reading Comprehension in LLMs"
        )

    def test_stripped_navigation_is_still_rejected(self, scraper):
        """ "View all posts" survives the strip as "all posts", which the nav test
        then catches — the two rules compose rather than overlapping."""
        link = BeautifulSoup(
            '<a href="/blog/x" aria-label="View all posts"><img src="/c.png"></a>',
            "html.parser",
        ).find("a")
        assert scraper._extract_title(link) == ""

    def test_known_edge_a_title_that_opens_with_a_cta_loses_that_word(self, scraper):
        """Documented limitation, not an accident. It needs a title that BEGINS
        with a CTA word and reaches us only through the accessible name, and it
        costs one word — against the concatenated blobs this path replaced."""
        link = BeautifulSoup(
            '<a href="/blog/x" aria-label="Read the Docs Integration">'
            '<img src="/c.png"></a>',
            "html.parser",
        ).find("a")
        assert scraper._extract_title(link) == "the Docs Integration"

"""What counts as article content in a page's HTML."""

from bs4 import BeautifulSoup

from app.logger import logger


def visible_text(html: str) -> str:
    """The readable prose in an HTML string, site furniture and markup stripped.

    Single source of truth for "what a reader actually sees", used by the
    scrape decision, the content-sufficiency gate and the relevance filter so
    all three agree on what counts as content — and agree with the summarizer,
    which receives ``content_html`` and so never sees ``NON_CONTENT_TAGS``.

    Counting those tags would let a JavaScript-rendered page's menus and script
    data clear the gate while the summarizer receives little more than a title.
    """
    if not html:
        return ""
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(list(NON_CONTENT_TAGS)):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        # Never the raw HTML: markup would count as prose at the content gate.
        logger.warning("Could not extract visible text, counting none: %s", e)
        return ""


# Elements that carry site furniture rather than article content. ``noscript`` is
# deliberately ABSENT: lazy-loading pages put the real <img src> there and leave a
# ``data:image/svg+xml`` placeholder in the visible markup, so dropping it would
# throw away exactly the figures the summary is supposed to include (verified on
# NVIDIA's blog, where 7 of 15 images live in noscript).
NON_CONTENT_TAGS: frozenset[str] = frozenset(
    {"script", "style", "svg", "iframe", "form", "nav", "footer", "template"}
)
# Attributes worth keeping. Everything else — class hooks, data-*, inline styles,
# tracking ids — is markup the model has no use for.
CONTENT_ATTRIBUTES: frozenset[str] = frozenset(
    {"src", "href", "alt", "srcset", "colspan", "rowspan"}
)


def content_html(html: str) -> str:
    """``html`` with site furniture and decorative attributes removed.

    The summarizer needs HTML, not prose: code blocks, tables and image URLs all
    live in the markup, which is why the filter's visible-text shortcut cannot be
    reused here. But most of that markup is not the article — it is navigation,
    scripts, inline styles and class hooks.

    Measured over one issue's three articles with the Bedrock token counter,
    dropping it cuts summarization input from 85,947 to 53,954 tokens (-37%) while
    leaving every ``<pre>``, ``<table>`` and article ``<img>`` in place. Less noise
    around the article is also one less thing for the model to mistake for
    content — a related-posts carousel reads a lot like a list of findings.
    """
    if not html:
        return ""
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(list(NON_CONTENT_TAGS)):
            tag.decompose()
        for tag in soup.find_all(True):
            tag.attrs = {
                key: value
                for key, value in tag.attrs.items()
                if key in CONTENT_ATTRIBUTES
            }
        return str(soup)
    except Exception as e:
        # Never let input shaping cost the post: fall back to the raw HTML.
        logger.warning("Could not clean content HTML, using it as-is: %s", e)
        return html

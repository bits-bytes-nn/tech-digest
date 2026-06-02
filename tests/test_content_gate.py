"""Tests for the content-sufficiency gate that prevents thin articles from
reaching the summarizer (the 'article too short to write about' bug)."""

from __future__ import annotations

from datetime import UTC, datetime

from app.src.feed_parser import Post


def _post(content: str) -> Post:
    return Post(
        title="t",
        link="https://example.com/x",
        published_date=datetime.now(UTC),
        content=content,
    )


class TestPostTextLength:
    def test_empty_content_is_zero(self):
        assert _post("").text_length() == 0

    def test_strips_html_markup(self):
        # Lots of markup, little visible text.
        html = "<div class='x'><script>var a=1;</script><p>Hi</p></div>"
        assert _post(html).text_length() == len("Hi")

    def test_counts_visible_text_only(self):
        html = "<p>" + "word " * 100 + "</p>"
        # 100 * len("word ") minus trailing space stripped == 499
        assert _post(html).text_length() == 499

    def test_plain_text_counted(self):
        assert _post("hello world").text_length() == len("hello world")


class TestContentLengthGate:
    """The gate lives on Summarizer; we exercise it directly without Bedrock."""

    def _make_summarizer(self, min_len: int):
        # Build a Summarizer-like object without triggering Bedrock setup by
        # using __new__ and setting only the attributes the gate touches.
        from app.src.summarizer import Summarizer

        s = Summarizer.__new__(Summarizer)
        s.min_content_length = min_len
        s.filtered_out_posts = []
        return s

    def test_thin_post_dropped_and_recorded(self):
        s = self._make_summarizer(min_len=600)
        thin = _post("<p>too short</p>")
        result = s._gate_by_content_length([thin])
        assert result == []
        assert len(s.filtered_out_posts) == 1
        dropped, reason = s.filtered_out_posts[0]
        assert dropped is thin
        assert "Insufficient content" in reason

    def test_substantial_post_passes(self):
        s = self._make_summarizer(min_len=600)
        big = _post("<p>" + "content " * 200 + "</p>")
        result = s._gate_by_content_length([big])
        assert result == [big]
        assert s.filtered_out_posts == []

    def test_mixed_batch(self):
        s = self._make_summarizer(min_len=100)
        thin = _post("<p>x</p>")
        big = _post("<p>" + "y " * 100 + "</p>")
        result = s._gate_by_content_length([thin, big])
        assert result == [big]
        assert len(s.filtered_out_posts) == 1

    def test_zero_threshold_passes_everything(self):
        s = self._make_summarizer(min_len=0)
        empty = _post("")
        result = s._gate_by_content_length([empty])
        assert result == [empty]

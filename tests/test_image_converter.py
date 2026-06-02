"""Tests for HtmlToImageConverter's browser-availability guard, which fixes the
silent Lambda failure (no Chrome in the Lambda runtime image)."""

from __future__ import annotations

import pytest

from app.src.newsletter_renderer import HtmlToImageConverter


class TestChromeAvailability:
    def test_reports_false_when_no_browser(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "app.src.newsletter_renderer.shutil.which", lambda name: None
        )
        monkeypatch.delenv("CHROME_BINARY", raising=False)
        converter = HtmlToImageConverter(tmp_path)
        assert converter.chrome_available() is False

    def test_reports_true_when_browser_on_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "app.src.newsletter_renderer.shutil.which",
            lambda name: (
                "/usr/bin/google-chrome-stable"
                if name == "google-chrome-stable"
                else None
            ),
        )
        converter = HtmlToImageConverter(tmp_path)
        assert converter.chrome_available() is True

    def test_convert_raises_clear_error_without_browser(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "app.src.newsletter_renderer.shutil.which", lambda name: None
        )
        monkeypatch.delenv("CHROME_BINARY", raising=False)
        html = tmp_path / "page.html"
        html.write_text("<html><body>hi</body></html>", encoding="utf-8")
        converter = HtmlToImageConverter(tmp_path)
        with pytest.raises(RuntimeError, match="Chrome/Chromium"):
            converter.convert(html)

    def test_convert_missing_file_raises(self, tmp_path):
        converter = HtmlToImageConverter(tmp_path)
        with pytest.raises(FileNotFoundError):
            converter.convert(tmp_path / "does-not-exist.html")

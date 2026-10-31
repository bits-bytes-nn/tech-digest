from __future__ import annotations

import io
import json
import os
import shutil
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jinja2 import Environment, FileSystemLoader, Template

from app.domain.constants import Language, LocalPaths
from app.domain.newsletter import (
    GMAIL_CLIP_BYTES,
    UI_LABELS,
    Article,
    BuildConfiguration,
    Footer,
    Header,
    NewsletterData,
    Section,
    collapse_html_whitespace,
    estimate_reading_minutes,
)
from app.logger import logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from selenium import webdriver


class NewsletterRenderer:
    def __init__(self, templates_dir: Path):
        self.env = Environment(
            loader=FileSystemLoader(templates_dir),
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=True,
        )
        self.newsletter_template: Template = self.env.get_template(
            LocalPaths.TEMPLATE_FILE.value
        )
        self.article_template: Template = self.env.get_template(
            LocalPaths.ARTICLE_FILE.value
        )

    def render_article(self, article: Article, language: Language = Language.KO) -> str:
        return collapse_html_whitespace(
            self.article_template.render(
                article=article.model_dump(),
                language=language.value,
                labels=UI_LABELS.get(language, UI_LABELS[Language.EN]),
            )
        )

    def render_newsletter(self, data: NewsletterData) -> str:
        # mode="json" so the Language enum serializes to its value ("ko"/"en").
        # A plain model_dump() hands the template the enum member, which renders
        # as "Language.KO" into the <html lang> attribute.
        html = collapse_html_whitespace(
            self.newsletter_template.render(data=data.model_dump(mode="json"))
        )
        self._warn_if_clipped(html, len(data.articles))
        return html

    @staticmethod
    def _warn_if_clipped(html: str, article_count: int) -> None:
        size = len(html.encode("utf-8"))
        if size <= GMAIL_CLIP_BYTES:
            logger.info(
                "Rendered newsletter is %.1f KB (%d articles), within the "
                "%.0f KB mail-client clip budget.",
                size / 1024,
                article_count,
                GMAIL_CLIP_BYTES / 1024,
            )
            return
        logger.warning(
            "Rendered newsletter is %.1f KB (%d articles), OVER the %.0f KB "
            "clip budget — Gmail will truncate the message and readers will "
            "lose the trailing cards and footer. Reduce summarization.max_posts "
            "or tighten the summary length budget in the prompt.",
            size / 1024,
            article_count,
            GMAIL_CLIP_BYTES / 1024,
        )


class HtmlToImageConverter:
    DEFAULT_WIDTH: int = 1200
    DEFAULT_WAIT_TIME: int = 2
    DEFAULT_MAX_HEIGHT: int = 2000
    DEFAULT_OVERLAP: int = 50
    # Common Chrome/Chromium binary names; the Batch image installs
    # google-chrome-stable.
    CHROME_BINARIES: tuple[str, ...] = (
        "google-chrome-stable",
        "google-chrome",
        "chromium",
        "chromium-browser",
        "chrome",
    )

    def __init__(self, output_dir: Path, **kwargs: Any) -> None:
        self.output_dir = output_dir
        self.max_height = kwargs.get("max_height", self.DEFAULT_MAX_HEIGHT)
        self.overlap = kwargs.get("overlap", self.DEFAULT_OVERLAP)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def chrome_available(cls) -> bool:
        """Whether a Chrome/Chromium binary is on PATH (or set via env).

        Used to fail fast with a clear message instead of a cryptic
        WebDriverException when ``convert_to_images`` is enabled where no
        browser is installed (e.g. a local run)."""
        if os.environ.get("CHROME_BINARY"):
            return Path(os.environ["CHROME_BINARY"]).exists()
        return any(shutil.which(name) for name in cls.CHROME_BINARIES)

    @staticmethod
    def _configure_chrome_options() -> Any:
        # Imported lazily: selenium + webdriver-manager + PIL are only needed for
        # the optional ``convert_to_images`` path, which is off by default. A
        # module-level import paid their cost on every run.
        from selenium.webdriver.chrome.options import Options

        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(f"--window-size={HtmlToImageConverter.DEFAULT_WIDTH},3000")
        if binary := os.environ.get("CHROME_BINARY"):
            options.binary_location = binary
        return options

    def convert(self, html_path: Path) -> list[Path]:
        if not html_path.exists():
            raise FileNotFoundError(f"HTML file not found: '{html_path}'")
        if not self.chrome_available():
            raise RuntimeError(
                "HTML-to-image conversion requires a Chrome/Chromium browser, "
                "but none was found on PATH. Set 'convert_to_images: false', or "
                "set CHROME_BINARY to the browser path (the Batch image installs "
                "google-chrome-stable)."
            )
        output_paths = []
        with self.driver_session() as driver:
            driver.get(f"file://{html_path.absolute()}")
            time.sleep(self.DEFAULT_WAIT_TIME)
            total_height = driver.execute_script("return document.body.scrollHeight")
            if total_height <= self.max_height:
                output_paths.append(
                    self._capture_single_page(driver, html_path, total_height)
                )
            else:
                output_paths.extend(
                    self._capture_split_pages(driver, html_path, total_height)
                )
        return output_paths

    @contextmanager
    def driver_session(self) -> Generator[webdriver.Chrome, None, None]:
        from selenium import webdriver as _webdriver
        from selenium.common.exceptions import WebDriverException
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager

        driver = None
        try:
            # Prefer a chromedriver already on PATH or pinned via env (present
            # in the Batch image); fall back to webdriver-manager's download
            # only when neither is available (e.g. local dev).
            driver_path = os.environ.get("CHROMEDRIVER_PATH") or shutil.which(
                "chromedriver"
            )
            service = Service(driver_path or ChromeDriverManager().install())
            driver = _webdriver.Chrome(
                service=service, options=self._configure_chrome_options()
            )
            yield driver
        except WebDriverException as e:
            logger.error("WebDriver initialization failed: %s", e)
            raise
        finally:
            if driver:
                driver.quit()

    def _capture_single_page(
        self, driver: webdriver.Chrome, html_path: Path, height: int
    ) -> Path:
        output_path = self.output_dir / f"{html_path.stem}.png"
        driver.set_window_size(self.DEFAULT_WIDTH, height)
        time.sleep(0.5)
        driver.save_screenshot(str(output_path))
        logger.info("Captured single page image: '%s'", output_path)
        return output_path

    def _capture_split_pages(
        self, driver: webdriver.Chrome, html_path: Path, total_height: int
    ) -> list[Path]:
        from PIL import Image

        paths = []
        y_offset = 0
        page_num = 1
        while y_offset < total_height:
            scroll_height = min(self.max_height, total_height - y_offset)
            driver.set_window_size(self.DEFAULT_WIDTH, scroll_height)
            driver.execute_script(f"window.scrollTo(0, {y_offset});")
            time.sleep(0.5)
            screenshot_bytes = driver.get_screenshot_as_png()
            img = Image.open(io.BytesIO(screenshot_bytes))
            output_path = (
                self.output_dir / f"{html_path.stem}-p{str(page_num).zfill(2)}.png"
            )
            img.save(str(output_path))
            paths.append(output_path)
            logger.info("Captured page %d for '%s'", page_num, html_path.name)
            y_offset += self.max_height - self.overlap
            page_num += 1
        return paths


class NewsletterBuilder:
    def __init__(
        self,
        inputs_dir: Path,
        outputs_dir: Path,
        templates_dir: Path,
        logos: dict[str, str],
    ) -> None:
        self.inputs_dir = inputs_dir
        self.outputs_dir = outputs_dir
        self.renderer = NewsletterRenderer(templates_dir)
        self.articles_dir = self.outputs_dir / LocalPaths.ARTICLES_DIR.value
        self.logos = logos
        self.article_filenames: list[str] = []
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.articles_dir.mkdir(parents=True, exist_ok=True)
        # Built on demand: constructing it eagerly configured Chrome options for
        # a feature that is off by default on every run.
        self._image_converter: HtmlToImageConverter | None = None

    @property
    def image_converter(self) -> HtmlToImageConverter:
        if self._image_converter is None:
            self._image_converter = HtmlToImageConverter(self.articles_dir)
        return self._image_converter

    def build(self, config: BuildConfiguration):
        try:
            newsletter_data = self._prepare_data(config)
            html_content = self.renderer.render_newsletter(newsletter_data)
            newsletter_path = self._save_html(html_content, "newsletter", config)
            logger.info("Newsletter successfully saved to '%s'", newsletter_path)
            if config.save_individual_articles:
                self._save_individual_articles(newsletter_data.articles, config)
            return newsletter_path, self.article_filenames
        except Exception as e:
            logger.error("Failed to build newsletter: %s", e, exc_info=True)
            raise

    def _prepare_data(self, config: BuildConfiguration) -> NewsletterData:
        header = Header(
            title=config.header_title,
            description=config.header_description,
            thumbnail=config.header_thumbnail,
            publish_date=config.date_suffix,
        )
        section = Section(introduction=config.first_section_intro)
        footer = Footer(title=config.footer_title)
        articles = self._load_articles(config.date_suffix, config.language)
        if not articles:
            # Every article file failed to load (each failure is logged above).
            # Rendering anyway produced a header and footer around nothing and
            # mailed it to every reader as this week's issue.
            raise ValueError(
                f"No article could be loaded for {config.date_suffix}; "
                "refusing to build an empty newsletter."
            )
        return NewsletterData(
            header=header,
            section=section,
            articles=articles,
            footer=footer,
            language=config.language,
        )

    def _load_articles(
        self, date_suffix: str, language: Language = Language.KO
    ) -> list[Article]:
        target_dir = self.inputs_dir / date_suffix
        if not target_dir.is_dir():
            logger.warning("Input directory not found: '%s'", target_dir)
            return []
        articles = []
        for file_path in sorted(target_dir.glob("*.json")):
            try:
                data = json.loads(file_path.read_text("utf-8"))
                source = data.get("source", "unknown")
                data["thumbnail"] = self.logos.get(source) or self.logos.get(
                    "unknown", ""
                )
                data["reading_minutes"] = estimate_reading_minutes(
                    data.get("summary", ""), language
                )
                articles.append(Article.model_validate(data))
            except Exception as e:
                logger.error("Failed to process article file '%s': %s", file_path, e)
        # Lead with the most relevant article. Sorting by publication date threw
        # away the ranking the filtering stage paid for: a weekly window puts
        # nearly every post on the same day or two, so ties fell back to glob
        # order (i.e. filename), which routinely buried the top-scored piece
        # below weaker ones. Score first, recency as the tie-break — the same key
        # the summarizer selects with, so selection and presentation agree.
        return sorted(articles, key=lambda a: (a.score, a.published_date), reverse=True)

    def _save_individual_articles(
        self, articles: list[Article], config: BuildConfiguration
    ):
        for i, article in enumerate(articles):
            article_html = self.renderer.render_article(article, config.language)
            article_path = self._save_html(article_html, "article", config, index=i + 1)
            self.article_filenames.append(article_path.name)
            logger.info("Saved individual article: '%s'", article_path)
            if config.convert_to_images:
                try:
                    self.image_converter.convert(article_path)
                except Exception as e:
                    logger.error(
                        "Failed to convert '%s' to image: %s", article_path.name, e
                    )

    def _save_html(
        self,
        content: str,
        basename: str,
        config: BuildConfiguration,
        index: int | None = None,
    ) -> Path:
        filename = self._generate_filename(basename, config, index)
        save_dir = self.articles_dir if basename == "article" else self.outputs_dir
        output_path = save_dir / filename
        output_path.write_text(content, "utf-8")
        return output_path

    @staticmethod
    def _generate_filename(
        basename: str, config: BuildConfiguration, index: int | None = None
    ) -> str:
        parts = [basename, config.stage, config.date_suffix, config.language.value]
        if index is not None:
            parts.append(f"a{str(index).zfill(2)}")
        return "-".join(parts) + ".html"

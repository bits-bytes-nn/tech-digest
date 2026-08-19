from __future__ import annotations

import io
import json
import os
import re
import shutil
import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from bs4 import BeautifulSoup
from jinja2 import Environment, FileSystemLoader, Template
from markupsafe import escape
from pydantic import BaseModel, Field, computed_field, field_validator

from .constants import Language, LocalPaths
from .feed_parser import is_safe_url
from .logger import logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from selenium import webdriver

# Gmail (and several other clients) truncate a message body larger than this,
# showing a "view entire message" link and hiding everything past the cut —
# including the trailing article cards and the footer. Measured against real
# published issues, every 5-article digest exceeded it, so the build warns
# loudly rather than letting the tail silently disappear from readers' inboxes.
GMAIL_CLIP_BYTES: int = 102_400

# Reading speed used for the per-card "N min read" estimate, per language.
# Korean is counted in characters and English in words; both are deliberately
# conservative for dense technical prose.
READING_SPEED: dict[Language, tuple[str, int]] = {
    Language.KO: ("chars", 450),
    Language.EN: ("words", 200),
}

# Reader-facing chrome, per language. Previously hardcoded English in the
# templates, which left a Korean digest with "Published on" / "View More" /
# "Additional resources for reference" labels around Korean content.
UI_LABELS: dict[Language, dict[str, str]] = {
    Language.KO: {
        "published_on": "발행",
        "view_more": "원문 보기",
        "resources": "함께 볼 자료",
        "reading_time": "읽는 시간 {minutes}분",
        "relevance": "관련도 점수",
    },
    Language.EN: {
        "published_on": "Published on",
        "view_more": "Read the original",
        "resources": "Further reading",
        "reading_time": "{minutes} min read",
        "relevance": "Relevance score",
    },
}

_WHITESPACE_RUN_WITH_NEWLINE = re.compile(r"[ \t]*\r?\n[ \t\r\n]*")
_HORIZONTAL_WHITESPACE_RUN = re.compile(r"[ \t]{2,}")
_PRE_BLOCK = re.compile(r"(<pre\b[^>]*>.*?</pre>)", re.DOTALL | re.IGNORECASE)


def collapse_html_whitespace(html: str) -> str:
    """Shrink the template's pretty-printing whitespace without changing layout.

    Table-based email markup is deeply indented, which costs ~10% of the message
    size in pure indentation — bytes that count against the clip threshold above.

    Every whitespace run is collapsed to a SINGLE space rather than removed.
    Removing it would be a few percent smaller but could delete a meaningful
    inter-word space between two inline tags in the model-generated prose
    (``<em>A</em>\\n<em>B</em>``), so the safe collapse is used instead.
    ``<pre>`` blocks are left byte-for-byte intact, since whitespace there IS
    the content.
    """
    parts = _PRE_BLOCK.split(html)
    for i, part in enumerate(parts):
        if i % 2:  # odd indices are the captured <pre>...</pre> blocks
            continue
        part = _WHITESPACE_RUN_WITH_NEWLINE.sub(" ", part)
        parts[i] = _HORIZONTAL_WHITESPACE_RUN.sub(" ", part)
    return "".join(parts)


def estimate_reading_minutes(html: str, language: Language) -> int:
    """Minutes to read an article card's rendered summary (minimum 1)."""
    text = BeautifulSoup(html or "", "html.parser").get_text(separator=" ", strip=True)
    unit, per_minute = READING_SPEED.get(language, READING_SPEED[Language.EN])
    amount = len(text) if unit == "chars" else len(text.split())
    return max(1, round(amount / per_minute)) if amount else 1


class NewsletterConfig:
    DATE_FORMATS: ClassVar[tuple[str, ...]] = (
        "%Y-%m-%d",
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%a, %d %b %Y %H:%M:%S GMT",
    )
    DEFAULT_STYLES: ClassVar[dict[str, str]] = {
        "header_title": "Weekly AI Tech Blog Digest",
        "header_description": "Stay ahead of the curve with our curated digest of the most impactful AI developments, research breakthroughs, and industry updates from leading tech companies and research institutions.",
        "header_thumbnail": "peccy.png",
        "first_section_intro": "This week's highlights showcase groundbreaking AI innovations and developments that are shaping the future of technology. We've carefully selected and summarized the most relevant articles to keep you informed of the latest advancements in artificial intelligence.",
        "footer_title": "Thank you for reading! Stay tuned for next week's AI insights and discoveries.",
    }


def normalize_date(v: str) -> str:
    """A date string as ``YYYY-MM-DD``, falling back to today (UTC).

    Named ``validate_date`` before, which collided with a same-named predicate in
    ``run_batch`` that answers a different question (is this string a valid date?)
    with a different type. This one normalizes and always returns a date.

    The fallback is UTC because the rest of the pipeline is: the Batch image sets
    TZ=Asia/Seoul, so a naive ``now()`` here dated an issue up to nine hours ahead
    of the window it was actually built for.
    """
    if not isinstance(v, str):
        v = str(v)
    for fmt in NewsletterConfig.DATE_FORMATS:
        try:
            return datetime.strptime(v, fmt).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    return datetime.now(UTC).strftime("%Y-%m-%d")


class Article(BaseModel):
    title: str = Field(min_length=1)
    link: str
    published_date: str
    thumbnail: str
    summary: str = Field(min_length=1)
    # Single-sentence lede shown above the body so the digest can be skimmed.
    # Rendered through Jinja autoescape (not ``| safe``), so it needs no
    # sanitization here.
    one_liner: str = ""
    reading_minutes: int = Field(default=0, ge=0)
    source: str = "unknown"
    tags: list[str] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    _normalize_date = field_validator("published_date", mode="before")(normalize_date)

    @field_validator("link", mode="before")
    @classmethod
    def _validate_link_scheme(cls, v: Any) -> str:
        # article.link is rendered into an href in the outbound email. It
        # originates from scraped/feed content, so a javascript:/data: scheme
        # would be a phishing/injection vector in clients that don't strip it.
        # Blank an unsafe scheme rather than raise, so one bad link doesn't drop
        # the whole article (the template guards against an empty href).
        link = str(v) if v is not None else ""
        return link if is_safe_url(link) else ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def source_label(self) -> str:
        """Human-friendly source name for alt text / accessibility. Exposed as a
        computed field so it appears in ``model_dump()`` and the templates can
        use it directly (single source of truth instead of duplicating the
        replace/title filter chain in each template)."""
        return "" if self.source == "unknown" else self.source.replace("_", " ").title()


class Footer(BaseModel):
    title: str


class Header(BaseModel):
    title: str = Field(min_length=1)
    description: str
    thumbnail: str
    publish_date: str
    _normalize_publish_date = field_validator("publish_date", mode="before")(
        normalize_date
    )


class Section(BaseModel):
    introduction: str

    @field_validator("introduction", mode="before")
    @classmethod
    def _escape_introduction(cls, v: Any) -> str:
        # The greeting is LLM-generated plain text rendered into the template
        # with Jinja `| safe`, so it is a trust boundary just like the summary
        # HTML (see summarizer._sanitize_html). The greeting prompt asks for
        # plain text, so HTML-escaping is a no-op for legitimate output while
        # neutralizing any injected <script>/markup echoed from scraped titles.
        return str(escape(v)) if v is not None else ""


class NewsletterData(BaseModel):
    header: Header
    section: Section
    articles: list[Article]
    footer: Footer
    # The template read ``data.language`` to set <html lang="...">, but the field
    # never existed, so every issue — including English ones — silently fell back
    # to the ``| default('ko')`` filter. Screen readers and client-side
    # translation both key off that attribute.
    language: Language = Language.KO

    @computed_field  # type: ignore[prop-decorator]
    @property
    def labels(self) -> dict[str, str]:
        """Reader-facing chrome in the issue's language (single source of truth
        for both the newsletter and the standalone-article templates)."""
        return UI_LABELS.get(self.language, UI_LABELS[Language.EN])


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
    # google-chrome-stable. The Lambda base image ships NO browser, so image
    # conversion is unsupported there (see chrome_available / convert()).
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
        WebDriverException when ``convert_to_images`` is enabled in an
        environment without a browser (notably the Lambda runtime image)."""
        if os.environ.get("CHROME_BINARY"):
            return Path(os.environ["CHROME_BINARY"]).exists()
        return any(shutil.which(name) for name in cls.CHROME_BINARIES)

    @staticmethod
    def _configure_chrome_options() -> Any:
        # Imported lazily: selenium + webdriver-manager + PIL are only needed for
        # the optional ``convert_to_images`` path, which is off by default. A
        # module-level import paid their cost (and their import-time side effects)
        # on every single run, including in the browserless Lambda image.
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
                "but none was found on PATH. The AWS Lambda runtime image does "
                "not include a browser — run with 'lambda_or_batch: batch' "
                "(the Batch image installs google-chrome-stable) or set "
                "'convert_to_images: false'. Set CHROME_BINARY to override the "
                "browser path."
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


class BuildConfiguration(BaseModel):
    stage: str = "dev"
    # UTC, like the rest of the pipeline. A naive now() here read as Asia/Seoul
    # in the Batch image, so the default could name tomorrow's date.
    date_suffix: str = Field(
        default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%d")
    )
    language: Language = Language.KO
    header_title: str = NewsletterConfig.DEFAULT_STYLES["header_title"]
    header_description: str = NewsletterConfig.DEFAULT_STYLES["header_description"]
    header_thumbnail: str = NewsletterConfig.DEFAULT_STYLES["header_thumbnail"]
    first_section_intro: str = NewsletterConfig.DEFAULT_STYLES["first_section_intro"]
    footer_title: str = NewsletterConfig.DEFAULT_STYLES["footer_title"]
    save_individual_articles: bool = False
    convert_to_images: bool = False


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
        self.newsletter_filename: str | None = None
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
            self.newsletter_filename = newsletter_path.name
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
                data["thumbnail"] = self.logos.get(source, self.logos["unknown"])
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

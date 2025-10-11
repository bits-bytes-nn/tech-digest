import io
import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import json
from jinja2 import Environment, FileSystemLoader, Template
from PIL import Image
from pydantic import BaseModel, Field, field_validator
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from .constants import Language, LocalPaths
from .logger import logger


class NewsletterConfig:
    DATE_FORMATS: tuple[str, ...] = (
        "%Y-%m-%d",
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%a, %d %b %Y %H:%M:%S GMT",
    )
    DEFAULT_STYLES: dict[str, str] = {
        "header_title": "Weekly AI Tech Blog Digest",
        "header_description": "Stay ahead of the curve with our curated digest of the most impactful AI developments, research breakthroughs, and industry updates from leading tech companies and research institutions.",
        "header_thumbnail": "peccy.png",
        "first_section_intro": "This week's highlights showcase groundbreaking AI innovations and developments that are shaping the future of technology. We've carefully selected and summarized the most relevant articles to keep you informed of the latest advancements in artificial intelligence.",
        "footer_title": "Thank you for reading! Stay tuned for next week's AI insights and discoveries.",
    }
    LOGOS: dict[str, str] = {
        "airbnb": "airbnb.png",
        "amazon": "amazon.png",
        "anthropic": "anthropic.png",
        "aws": "aws.png",
        "google": "google.png",
        "huggingface": "huggingface.png",
        "kakao": "kakao.png",
        "linkedin": "linkedin.png",
        "meta": "meta.png",
        "microsoft": "microsoft.png",
        "ncsoft": "ncsoft.png",
        "netflix": "netflix.png",
        "nvidia": "nvidia.png",
        "openai": "openai.png",
        "palantir": "palantir.png",
        "pinterest": "pinterest.png",
        "qwen": "qwen.png",
        "xai": "xai.png",
        "unknown": "unknown.png",
        # NOTE: add new logos here
    }


def validate_date(v: str) -> str:
    if not isinstance(v, str):
        v = str(v)
    for fmt in NewsletterConfig.DATE_FORMATS:
        try:
            return datetime.strptime(v, fmt).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    return datetime.now().strftime("%Y-%m-%d")


class Article(BaseModel):
    title: str = Field(min_length=1)
    link: str
    published_date: str
    thumbnail: str
    summary: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    _validate_date = field_validator("published_date", mode="before")(validate_date)


class Footer(BaseModel):
    title: str


class Header(BaseModel):
    title: str = Field(min_length=1)
    description: str
    thumbnail: str
    publish_date: str
    _validate_publish_date = field_validator("publish_date", mode="before")(
        validate_date
    )


class Section(BaseModel):
    introduction: str


class NewsletterData(BaseModel):
    header: Header
    section: Section
    articles: list[Article]
    footer: Footer


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

    def render_article(self, article: Article) -> str:
        return self.article_template.render(article=article.model_dump())

    def render_newsletter(self, data: NewsletterData) -> str:
        return self.newsletter_template.render(data=data.model_dump())


class HtmlToImageConverter:
    DEFAULT_WIDTH: int = 1200
    DEFAULT_WAIT_TIME: int = 2
    DEFAULT_MAX_HEIGHT: int = 2000
    DEFAULT_OVERLAP: int = 50

    def __init__(self, output_dir: Path, **kwargs: Any) -> None:
        self.output_dir = output_dir
        self.max_height = kwargs.get("max_height", self.DEFAULT_MAX_HEIGHT)
        self.overlap = kwargs.get("overlap", self.DEFAULT_OVERLAP)
        self.chrome_options = self._configure_chrome_options()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _configure_chrome_options() -> Options:
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(f"--window-size={HtmlToImageConverter.DEFAULT_WIDTH},3000")
        return options

    def convert(self, html_path: Path) -> list[Path]:
        if not html_path.exists():
            raise FileNotFoundError(f"HTML file not found: '{html_path}'")
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
        driver = None
        try:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=self.chrome_options)
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
    date_suffix: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d")
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
        self.image_converter = HtmlToImageConverter(self.articles_dir)
        self.logos = logos
        self.newsletter_filename: str | None = None
        self.article_filenames: list[str] = []
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.articles_dir.mkdir(parents=True, exist_ok=True)

    def build(self, config: "BuildConfiguration"):
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

    def _prepare_data(self, config: "BuildConfiguration") -> NewsletterData:
        header = Header(
            title=config.header_title,
            description=config.header_description,
            thumbnail=config.header_thumbnail,
            publish_date=config.date_suffix,
        )
        section = Section(introduction=config.first_section_intro)
        footer = Footer(title=config.footer_title)
        articles = self._load_articles(config.date_suffix)
        return NewsletterData(
            header=header, section=section, articles=articles, footer=footer
        )

    def _load_articles(self, date_suffix: str) -> list[Article]:
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
                articles.append(Article.model_validate(data))
            except Exception as e:
                logger.error("Failed to process article file '%s': %s", file_path, e)
        return sorted(articles, key=lambda a: a.published_date, reverse=True)

    def _save_individual_articles(
        self, articles: list[Article], config: "BuildConfiguration"
    ):
        for i, article in enumerate(articles):
            article_html = self.renderer.render_article(article)
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
        config: "BuildConfiguration",
        index: int | None = None,
    ) -> Path:
        filename = self._generate_filename(basename, config, index)
        save_dir = self.articles_dir if basename == "article" else self.outputs_dir
        output_path = save_dir / filename
        output_path.write_text(content, "utf-8")
        return output_path

    @staticmethod
    def _generate_filename(
        basename: str, config: "BuildConfiguration", index: int | None = None
    ) -> str:
        parts = [basename, config.stage, config.date_suffix, config.language.value]
        if index is not None:
            parts.append(f"a{str(index).zfill(2)}")
        return "-".join(parts) + ".html"

import sys
from pathlib import Path
from typing import Final

sys.path.append(str(Path(__file__).parent.parent))
from app.configs import Config
from app.src import (
    BuildConfiguration,
    Language,
    LocalPaths,
    NewsletterBuilder,
    get_date_range,
    logger,
)

END_DATE: Final[str | None] = None
LANGUAGE: Final[Language | None] = None

if __name__ == "__main__":
    config = Config.load()
    _, end_date = get_date_range(END_DATE, config.scraping.days_back)
    date_suffix = end_date.strftime("%Y-%m-%d")

    root_dir = Path(__file__).resolve().parent.parent
    inputs_dir = root_dir / LocalPaths.INPUTS_DIR.value
    outputs_dir = root_dir / LocalPaths.OUTPUTS_DIR.value
    templates_dir = root_dir / LocalPaths.TEMPLATES_DIR.value
    greeting_path = inputs_dir / date_suffix / LocalPaths.GREETING_FILE.value

    with open(greeting_path, "r", encoding="utf-8") as f:
        first_section_intro = f.read()

    builder = NewsletterBuilder(
        inputs_dir=inputs_dir,
        outputs_dir=outputs_dir,
        templates_dir=templates_dir,
        logos=config.newsletter.logos,
    )

    build_config = BuildConfiguration(
        stage=config.resources.stage,
        date_suffix=date_suffix,
        language=Language(LANGUAGE or Language.KO),
        header_title=config.newsletter.header_title or "",
        header_description=config.newsletter.header_description or "",
        header_thumbnail=config.newsletter.header_thumbnail or "",
        first_section_intro=first_section_intro,
        footer_title=config.newsletter.footer_title or "",
        save_individual_articles=config.newsletter.save_articles,
        convert_to_images=config.newsletter.convert_to_images,
    )

    newsletter_path, _ = builder.build(build_config)
    logger.info("Newsletter created: '%s'", newsletter_path)

import logging
import os
from datetime import datetime

from app.domain.constants import EnvVars, LocalPaths
from app.paths import REPO_ROOT

LOG_FORMAT: str = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
LOGGER_NAME: str = "app"


def is_running_in_aws() -> bool:
    aws_env_vars = [
        "AWS_EXECUTION_ENV",
        "AWS_BATCH_JOB_ID",
        "ECS_CONTAINER_METADATA_URI",
    ]
    return any(env_var in os.environ for env_var in aws_env_vars)


def _resolve_level(name: str) -> int:
    """A logging level from its name, falling back to INFO.

    ``getattr(logging, name)`` was used here, which resolves any attribute of the
    module — ``LOG_LEVEL=disable`` returned the ``logging.disable`` FUNCTION and
    ``setLevel`` raised on it, crashing at import. ``getLevelNamesMapping`` only
    knows actual levels.
    """
    return logging.getLevelNamesMapping().get(name.upper(), logging.INFO)


def _file_handler(formatter: logging.Formatter) -> logging.FileHandler:
    logs_dir = REPO_ROOT / LocalPaths.LOGS_DIR.value
    logs_dir.mkdir(parents=True, exist_ok=True)
    name, ext = LocalPaths.LOGS_FILE.value.rsplit(".", 1)
    timestamp = datetime.now().strftime("%Y-%m-%d")
    handler = logging.FileHandler(
        logs_dir / f"{name}_{timestamp}.{ext}", encoding="utf-8"
    )
    handler.setFormatter(formatter)
    return handler


def configure_logger(
    name: str = LOGGER_NAME, level: int = logging.INFO
) -> logging.Logger:
    """Console logging, plus a dated file outside AWS (CloudWatch covers it there)."""
    logger_obj = logging.getLogger(name)
    logger_obj.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT)
    # Close before dropping: reconfiguring (as tests do) would otherwise leak the
    # previous file handle, since clearing the list alone does not close it.
    for handler in logger_obj.handlers:
        handler.close()
    logger_obj.handlers.clear()
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger_obj.addHandler(console_handler)
    log_to_file = os.getenv(EnvVars.LOG_TO_FILE.value, "true").lower() != "false"
    if log_to_file and not is_running_in_aws():
        logger_obj.addHandler(_file_handler(formatter))
    logger_obj.propagate = False
    return logger_obj


logger = configure_logger(
    level=_resolve_level(os.getenv(EnvVars.LOG_LEVEL.value, "INFO"))
)

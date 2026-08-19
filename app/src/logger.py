import logging
import os
from datetime import datetime
from pathlib import Path

from .constants import EnvVars, LocalPaths

LOG_FORMAT: str = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
LOGGER_NAME: str = "app"


def is_running_in_aws() -> bool:
    aws_env_vars = [
        "AWS_EXECUTION_ENV",
        "AWS_LAMBDA_FUNCTION_NAME",
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
    logs_dir = Path(__file__).resolve().parent.parent.parent / LocalPaths.LOGS_DIR.value
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
    """Console logging, plus a dated file outside AWS (CloudWatch covers it there).

    The two handler builders used to guard against adding a duplicate handler,
    which could not happen: this function clears the handler list immediately
    before calling them. The level and name were also wrapped in a ``LoggerConfig``
    class whose ``log_format`` and ``file_logging_enabled`` parameters no caller
    ever set.
    """
    logger_obj = logging.getLogger(name)
    logger_obj.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT)
    logger_obj.handlers.clear()
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger_obj.addHandler(console_handler)
    if not is_running_in_aws():
        logger_obj.addHandler(_file_handler(formatter))
    logger_obj.propagate = False
    return logger_obj


logger = configure_logger(
    level=_resolve_level(os.getenv(EnvVars.LOG_LEVEL.value, "INFO"))
)

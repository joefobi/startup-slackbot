"""
log.py — shared structlog configuration.

Import and call configure() once at application entry points (app.py, runner.py,
langsmith_eval.py). All modules then use:

    import structlog
    log = structlog.get_logger()
"""

import logging
import structlog


def configure(level: int = logging.INFO) -> None:
    """
    Configure structlog with a human-readable console renderer.

    Args:
        level (int): Standard library logging level (default: logging.INFO).
    """
    logging.basicConfig(level=level, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="%H:%M:%S", utc=False),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

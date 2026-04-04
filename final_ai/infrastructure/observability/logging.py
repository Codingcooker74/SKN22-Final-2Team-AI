import logging

_BASE_LOGGER_NAME = "final_ai"


def get_logger(name: str | None = None) -> logging.Logger:
    if not name or name == _BASE_LOGGER_NAME:
        logger = logging.getLogger(_BASE_LOGGER_NAME)
    elif name.startswith(f"{_BASE_LOGGER_NAME}."):
        logger = logging.getLogger(name)
    else:
        logger = logging.getLogger(f"{_BASE_LOGGER_NAME}.{name}")

    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger

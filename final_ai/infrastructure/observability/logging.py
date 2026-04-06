import logging
import sys

_BASE_LOGGER_NAME = "final_ai"


def get_logger(name: str | None = None) -> logging.Logger:
    if not name or name == _BASE_LOGGER_NAME:
        logger = logging.getLogger(_BASE_LOGGER_NAME)
    elif name.startswith(f"{_BASE_LOGGER_NAME}."):
        logger = logging.getLogger(name)
    else:
        logger = logging.getLogger(f"{_BASE_LOGGER_NAME}.{name}")

    # 핸들러가 없는 경우 콘솔 출력을 위한 StreamHandler 추가
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # 포맷 설정
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # 콘솔 핸들러 생성 및 추가
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 로그 전파 방지 (중복 출력 방지)
        logger.propagate = False

    return logger

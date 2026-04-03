from typing import Any

from fastapi.responses import JSONResponse


def error_response(
    message: str,
    *,
    status_code: int,
    code: str,
    details: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    payload: dict[str, Any] = {
        "error": {
            "code": code,
            "message": message,
        }
    }
    if details:
        payload["error"]["details"] = details
    return JSONResponse(status_code=status_code, content=payload, headers=headers)


def bad_request_response(
    message: str,
    *,
    code: str = "bad_request",
    details: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return error_response(message, status_code=400, code=code, details=details, headers=headers)


def internal_error_response(
    message: str = "요청 처리 중 오류가 발생했습니다.",
    *,
    code: str = "internal_error",
    details: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return error_response(message, status_code=500, code=code, details=details, headers=headers)

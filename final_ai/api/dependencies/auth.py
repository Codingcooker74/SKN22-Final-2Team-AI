from dataclasses import dataclass
from typing import Annotated
from uuid import uuid4

from fastapi import Header, Request


@dataclass(frozen=True)
class RequestAuthContext:
    request_id: str
    user_id: str | None
    session_id: str | None
    authorization: str | None
    access_token: str | None
    client_ip: str | None
    user_agent: str | None

    def log_extra(self, **extra) -> dict[str, str]:
        metadata = {
            "request_id": self.request_id,
            "session_id": self.session_id or "",
            "user_id": self.user_id or "",
            "client_ip": self.client_ip or "",
        }
        for key, value in extra.items():
            metadata[key] = "" if value is None else str(value)
        return metadata


def _extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    prefix = "bearer "
    lowered = authorization.strip().lower()
    if not lowered.startswith(prefix):
        return None
    token = authorization.strip()[len(prefix):].strip()
    return token or None


def get_request_auth_context(
    request: Request,
    x_request_id: Annotated[str | None, Header(alias="X-Request-Id")] = None,
    x_user_id: Annotated[str | None, Header(alias="X-User-Id")] = None,
    x_session_id: Annotated[str | None, Header(alias="X-Session-Id")] = None,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> RequestAuthContext:
    context = RequestAuthContext(
        request_id=x_request_id or str(uuid4()),
        user_id=x_user_id,
        session_id=x_session_id,
        authorization=authorization,
        access_token=_extract_bearer_token(authorization),
        client_ip=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )
    request.state.auth_context = context
    return context

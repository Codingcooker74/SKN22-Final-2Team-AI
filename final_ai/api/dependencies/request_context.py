from contextlib import contextmanager
from contextvars import ContextVar

_request_cancel_event: ContextVar[object | None] = ContextVar("request_cancel_event", default=None)


class RequestCancelled(RuntimeError):
    pass


@contextmanager
def bind_request_cancel_event(cancel_event):
    token = _request_cancel_event.set(cancel_event)
    try:
        yield
    finally:
        _request_cancel_event.reset(token)


def ensure_request_active() -> None:
    cancel_event = _request_cancel_event.get()
    if cancel_event is not None and cancel_event.is_set():
        raise RequestCancelled("Client disconnected")

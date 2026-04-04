"""Graph layer packages."""

from .state import ChatState

__all__ = ["ChatState", "build_graph", "chat", "graph"]


def __getattr__(name: str):
    if name in {"build_graph", "chat", "graph"}:
        from .builder import build_graph, chat, graph

        exports = {
            "build_graph": build_graph,
            "chat": chat,
            "graph": graph,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

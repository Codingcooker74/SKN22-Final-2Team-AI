from final_ai.graph.builder import build_graph
from final_ai.infrastructure.observability import traceable

_graph = None


def get_chat_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


@traceable(name="tailtalk_fastapi_chat_graph", run_type="chain")
def invoke_chat_graph(initial_state: dict, config: dict) -> dict:
    return get_chat_graph().invoke(initial_state, config=config)

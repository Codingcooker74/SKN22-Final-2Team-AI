from __future__ import annotations


try:
    from langsmith import traceable as _traceable
    from langsmith.wrappers import wrap_openai as _wrap_openai
except ImportError:
    def traceable(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator

    def wrap_openai(client):
        return client
else:
    def traceable(*args, **kwargs):
        return _traceable(*args, **kwargs)

    def wrap_openai(client):
        return _wrap_openai(client)

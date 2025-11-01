"""Minimal FastAPI compatibility layer for offline test environments."""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Pattern, Tuple


def _load_real_fastapi():
    original = list(sys.path)
    this_dir = Path(__file__).resolve().parent
    try:
        sys.path = [p for p in original if Path(p or "").resolve() != this_dir]
        spec = importlib.util.find_spec("fastapi")
        if spec and spec.origin and Path(spec.origin).resolve() != Path(__file__).resolve():
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            if spec.loader:
                spec.loader.exec_module(module)
            return module
        return None
    except ModuleNotFoundError:
        return None
    finally:
        sys.path = original


_REAL_FASTAPI = _load_real_fastapi()
if _REAL_FASTAPI is not None:
    globals().update({k: getattr(_REAL_FASTAPI, k) for k in dir(_REAL_FASTAPI)})
else:
    __all__ = [
        "FastAPI",
        "HTTPException",
        "Body",
        "Query",
        "Response",
    ]

    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: Any = None) -> None:
            super().__init__(detail)
            self.status_code = int(status_code)
            self.detail = detail

    class Response:
        def __init__(self, content: Any = None, status_code: int = 200) -> None:
            self.content = content
            self.status_code = int(status_code)

        def json(self) -> Any:
            if isinstance(self.content, (bytes, bytearray)):
                return json.loads(self.content.decode("utf-8"))
            if isinstance(self.content, str):
                return json.loads(self.content)
            return self.content

    def Body(*_: Any, default: Any = None, **__: Any) -> Any:
        return default

    def Query(*_: Any, default: Any = None, **__: Any) -> Any:
        return default

    class _Route:
        def __init__(self, path: str, methods: Iterable[str], endpoint: Callable[..., Any]) -> None:
            self.path = path
            self.methods = {m.upper() for m in methods}
            self.endpoint = endpoint
            self.regex, self.param_names = self._compile_path(path)

        @staticmethod
        def _compile_path(path: str) -> Tuple[Pattern[str], List[str]]:
            param_pattern = re.compile(r"{([^}]+)}")
            params: List[str] = []

            def replacer(match: re.Match[str]) -> str:
                raw = match.group(1)
                if ":" in raw:
                    name, conv = raw.split(":", 1)
                else:
                    name, conv = raw, ""
                params.append(name)
                if conv == "path":
                    return rf"(?P<{name}>.+)"
                return rf"(?P<{name}>[^/]+)"

            regex_pattern = "^" + param_pattern.sub(replacer, path.rstrip("/")) + "/*$"
            return re.compile(regex_pattern), params

        def match(self, method: str, url_path: str) -> Optional[Dict[str, str]]:
            if method.upper() not in self.methods:
                return None
            match = self.regex.match(url_path.rstrip("/"))
            if not match:
                return None
            return match.groupdict()

    class FastAPI:
        def __init__(self, title: str = "FastAPI", version: str = "0.1.0") -> None:
            self.title = title
            self.version = version
            self.routes: List[_Route] = []
            self._event_handlers: Dict[str, List[Callable[[], Any]]] = {"startup": [], "shutdown": []}

        def add_middleware(self, *_: Any, **__: Any) -> None:  # pragma: no cover - trivial stub
            return None

        def on_event(self, event: str) -> Callable[[Callable[[], Any]], Callable[[], Any]]:
            def decorator(func: Callable[[], Any]) -> Callable[[], Any]:
                self._event_handlers.setdefault(event, []).append(func)
                return func

            return decorator

        def _register(self, path: str, methods: Iterable[str]) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
                self.routes.append(_Route(path, methods, func))
                return func

            return decorator

        def get(self, path: str, **_: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            return self._register(path, ["GET"])

        def post(self, path: str, **_: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            return self._register(path, ["POST"])

        def delete(self, path: str, **_: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            return self._register(path, ["DELETE"])

        def run_event(self, event: str) -> None:
            for handler in self._event_handlers.get(event, []):
                handler()

    def _prepare_content(result: Any) -> Response:
        if isinstance(result, Response):
            return result
        if hasattr(result, "model_dump"):
            return Response(result.model_dump())
        if dataclasses.is_dataclass(result):  # pragma: no cover - defensive
            return Response(dataclasses.asdict(result))
        if isinstance(result, (dict, list, str, int, float, type(None))):
            return Response(result)
        return Response(str(result))

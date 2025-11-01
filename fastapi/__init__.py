"""Minimal FastAPI compatibility layer for offline test environments."""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Pattern, Tuple, Union


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

    import asyncio
    import inspect
    from datetime import date, datetime
    from typing import Any, Callable, Dict, Iterable, List, Optional, Pattern, Tuple, Union
    from typing import get_args, get_origin
    from urllib.parse import parse_qs

    try:  # pragma: no cover - optional dependency when running tests without pydantic
        from pydantic import BaseModel
    except Exception:  # noqa: BLE001
        class BaseModel:  # type: ignore[override]
            def __init__(self, **data: Any) -> None:
                for key, value in data.items():
                    setattr(self, key, value)

            def model_dump(self) -> Dict[str, Any]:
                return dict(self.__dict__)

            @classmethod
            def parse_obj(cls, data: Any) -> "BaseModel":
                return cls(**data)

    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: Any = None) -> None:
            super().__init__(detail)
            self.status_code = int(status_code)
            self.detail = detail

    class Response:
        def __init__(
            self,
            content: Any = None,
            status_code: int = 200,
            media_type: str = "application/json",
            headers: Optional[Dict[str, str]] = None,
        ) -> None:
            self.content = content
            self.status_code = int(status_code)
            self.media_type = media_type
            self.headers = dict(headers or {})

        def json(self) -> Any:
            if isinstance(self.content, (bytes, bytearray)):
                return json.loads(self.content.decode("utf-8"))
            if isinstance(self.content, str):
                return json.loads(self.content)
            return self.content

    def Body(*_: Any, default: Any = None, **__: Any) -> Any:  # pragma: no cover - metadata ignored
        return default

    def Query(*_: Any, default: Any = None, **__: Any) -> Any:  # pragma: no cover - metadata ignored
        return default

    class _Route:
        def __init__(
            self,
            path: str,
            methods: Iterable[str],
            endpoint: Callable[..., Any],
            order: int,
        ) -> None:
            self.path = path
            self.methods = {m.upper() for m in methods}
            self.endpoint = endpoint
            self.order = order
            (
                self.regex,
                self.param_names,
                self.dynamic_segments,
                self.static_length,
            ) = self._compile_path(path)

        @staticmethod
        def _compile_path(path: str) -> Tuple[Pattern[str], List[str], int, int]:
            param_pattern = re.compile(r"{([^}]+)}")
            params: List[str] = []
            segments = [segment for segment in path.strip("/").split("/") if segment]
            dynamic_segments = 0
            static_length = 0

            def replacer(match: re.Match[str]) -> str:
                raw = match.group(1)
                if ":" in raw:
                    name, conv = raw.split(":", 1)
                else:
                    name, conv = raw, ""
                params.append(name)
                nonlocal dynamic_segments
                dynamic_segments += 1
                if conv == "path":
                    return rf"(?P<{name}>.+)"
                return rf"(?P<{name}>[^/]+)"

            regex_pattern = "^" + param_pattern.sub(replacer, path.rstrip("/")) + "/*$"
            if segments:
                static_length = sum(len(segment) for segment in segments if "{" not in segment)
            return re.compile(regex_pattern), params, dynamic_segments, static_length

        @property
        def priority(self) -> Tuple[int, int, int]:
            # Prefer static routes (fewer dynamic segments), then longer static matches.
            # Fall back to registration order for stability.
            return (self.dynamic_segments, -self.static_length, self.order)

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
            self._startup_complete = asyncio.Event()

        def add_middleware(self, *_: Any, **__: Any) -> None:  # pragma: no cover - middleware ignored in stub
            return None

        def on_event(self, event: str) -> Callable[[Callable[[], Any]], Callable[[], Any]]:
            def decorator(func: Callable[[], Any]) -> Callable[[], Any]:
                self._event_handlers.setdefault(event, []).append(func)
                return func

            return decorator

        def _register(self, path: str, methods: Iterable[str]) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
                self.routes.append(_Route(path, methods, func, len(self.routes)))
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

        def __call__(self, scope: Dict[str, Any]) -> Callable[[Callable[..., Any], Callable[..., Any]], Any]:
            async def app(receive: Callable[[], Any], send: Callable[[Dict[str, Any]], Any]) -> None:
                await self._ensure_startup()
                await self._handle_request(scope, receive, send)

            return app

        async def _ensure_startup(self) -> None:
            if not self._startup_complete.is_set():
                self.run_event("startup")
                self._startup_complete.set()

        async def _handle_request(
            self,
            scope: Dict[str, Any],
            receive: Callable[[], Any],
            send: Callable[[Dict[str, Any]], Any],
        ) -> None:
            if scope.get("type") != "http":
                await self._send_response(send, Response({"detail": "Unsupported scope"}, status_code=500))
                return

            method = scope.get("method", "GET").upper()
            path = scope.get("path", "")
            query_bytes = scope.get("query_string", b"")
            query_str = query_bytes.decode("utf-8", errors="ignore")
            raw_query = parse_qs(query_str, keep_blank_values=True)
            query_params: Dict[str, Any] = {
                key: values if len(values) > 1 else values[0]
                for key, values in raw_query.items()
            }

            body_bytes = b""
            while True:
                message = await receive()
                if message.get("type") != "http.request":
                    continue
                body_bytes += message.get("body", b"")
                if not message.get("more_body"):
                    break

            json_body: Any = None
            if body_bytes:
                try:
                    json_body = json.loads(body_bytes.decode("utf-8"))
                except json.JSONDecodeError:
                    json_body = body_bytes

            best_match: Optional[Tuple[_Route, Dict[str, str]]] = None
            best_priority: Optional[Tuple[int, int, int]] = None
            for route in self.routes:
                params = route.match(method, path)
                if params is None:
                    continue
                if best_match is None:
                    best_match = (route, params)
                    best_priority = route.priority
                    continue
                if best_priority is not None and route.priority < best_priority:
                    best_match = (route, params)
                    best_priority = route.priority

            if not best_match:
                await self._send_response(send, Response({"detail": "Not Found"}, status_code=404))
                return

            route, path_params = best_match

            try:
                kwargs = self._build_kwargs(route.endpoint, path_params, query_params, json_body)
                result = route.endpoint(**kwargs)
                if inspect.isawaitable(result):
                    result = await result
            except HTTPException as exc:
                await self._send_response(send, Response({"detail": exc.detail}, status_code=exc.status_code))
                return
            except Exception as exc:  # noqa: BLE001
                await self._send_response(send, Response({"detail": str(exc)}, status_code=500))
                return

            response = _prepare_content(result)
            await self._send_response(send, response)

        def _build_kwargs(
            self,
            endpoint: Callable[..., Any],
            path_params: Dict[str, str],
            query_params: Dict[str, Any],
            json_body: Any,
        ) -> Dict[str, Any]:
            signature = inspect.signature(endpoint)
            kwargs: Dict[str, Any] = {}
            body_candidates = [
                name
                for name in signature.parameters
                if name not in path_params and name not in query_params
            ]

            for name, param in signature.parameters.items():
                annotation = param.annotation
                raw: Any = None
                if name in path_params:
                    raw = path_params[name]
                elif name in query_params:
                    raw = query_params[name]
                else:
                    raw = self._extract_body_value(name, body_candidates, json_body)
                    if raw is None and param.default is not inspect._empty:
                        raw = param.default
                kwargs[name] = _convert_value(raw, annotation)
            return kwargs

        def _extract_body_value(self, name: str, candidates: List[str], json_body: Any) -> Any:
            if json_body is None:
                return None
            if isinstance(json_body, dict):
                if name in json_body:
                    return json_body[name]
                if len(candidates) == 1 and candidates[0] == name:
                    return json_body
                return None
            if len(candidates) == 1 and candidates[0] == name:
                return json_body
            return None

        async def _send_response(self, send: Callable[[Dict[str, Any]], Any], response: "Response") -> None:
            body, media_type = _encode_body(response.content, response.media_type)
            headers = [(b"content-type", media_type.encode("latin-1"))]
            for key, value in response.headers.items():
                headers.append((key.encode("latin-1"), str(value).encode("latin-1")))
            await send({"type": "http.response.start", "status": int(response.status_code), "headers": headers})
            await send({"type": "http.response.body", "body": body, "more_body": False})

    def _encode_body(content: Any, media_type: Optional[str]) -> Tuple[bytes, str]:
        if isinstance(content, Response):  # pragma: no cover - defensive
            return _encode_body(content.content, content.media_type)
        if isinstance(content, (bytes, bytearray)):
            return bytes(content), media_type or "application/octet-stream"
        if isinstance(content, str):
            return content.encode("utf-8"), media_type or "text/plain; charset=utf-8"
        if isinstance(content, (dict, list)):
            return json.dumps(content).encode("utf-8"), media_type or "application/json"
        if content is None:
            return b"null", media_type or "application/json"
        return str(content).encode("utf-8"), media_type or "text/plain; charset=utf-8"

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

    def _convert_value(value: Any, annotation: Any) -> Any:
        if value is None:
            return None
        if annotation is inspect._empty or annotation is Any:
            return value

        origin = get_origin(annotation)
        args = get_args(annotation)

        if origin is Union:  # type: ignore[name-defined]
            non_none = [arg for arg in args if arg is not type(None)]  # noqa: E721
            if value is None:
                return None
            for candidate in non_none:
                try:
                    return _convert_value(value, candidate)
                except Exception:  # noqa: BLE001
                    continue
            return value

        if inspect.isclass(annotation) and issubclass(annotation, BaseModel):  # type: ignore[arg-type]
            if isinstance(value, annotation):
                return value
            if hasattr(annotation, "model_validate"):
                return annotation.model_validate(value)  # type: ignore[return-value]
            if hasattr(annotation, "parse_obj"):
                return annotation.parse_obj(value)  # type: ignore[return-value]
            return annotation(**value)

        if origin in {list, List, Iterable, tuple, set}:  # type: ignore[arg-type]
            item_type = args[0] if args else Any
            if not isinstance(value, (list, tuple, set)):
                value = [value]
            converted = [_convert_value(item, item_type) for item in value]
            if origin is tuple:
                return tuple(converted)
            if origin is set:
                return set(converted)
            return list(converted)

        if annotation in {str, int, float}:
            return annotation(value)

        if annotation is bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() not in {"", "0", "false", "no"}
            return bool(value)

        if annotation in {datetime, date}:
            if isinstance(value, (datetime, date)):
                return value
            if isinstance(value, str):
                text = value.strip()
                if not text:
                    return None
                if annotation is datetime and text.endswith("Z"):
                    text = text[:-1] + "+00:00"
                return annotation.fromisoformat(text)  # type: ignore[return-value]
        return value

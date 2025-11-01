"""Small TestClient compatible with the FastAPI subset used in tests."""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def _load_real_submodule():
    original = list(sys.path)
    this_dir = Path(__file__).resolve().parent.parent
    try:
        sys.path = [p for p in original if Path(p or "").resolve() != this_dir]
        spec = importlib.util.find_spec("fastapi.testclient")
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


_REAL = _load_real_submodule()
if _REAL is not None:
    globals().update({k: getattr(_REAL, k) for k in dir(_REAL)})
else:
    from . import HTTPException, Response, _prepare_content

    class _SimpleResponse:
        def __init__(self, response: Response) -> None:
            self._response = response
            self.status_code = response.status_code

        def json(self) -> Any:
            return self._response.json()

        @property
        def content(self) -> Any:
            return self._response.content

    class TestClient:
        def __init__(self, app: Any) -> None:
            self.app = app

        def _call(self, method: str, path: str, **kwargs: Any) -> _SimpleResponse:
            for route in getattr(self.app, "routes", []):
                params = route.match(method, path)
                if params is None:
                    continue
                try:
                    result = route.endpoint(**params)
                except HTTPException as exc:
                    return _SimpleResponse(Response({"detail": exc.detail}, status_code=exc.status_code))
                if isinstance(result, Response):
                    response = result
                elif is_dataclass(result):
                    response = Response(asdict(result))
                else:
                    response = _prepare_content(result)
                return _SimpleResponse(response)
            return _SimpleResponse(Response({"detail": "not found"}, status_code=404))

        def get(self, path: str, **kwargs: Any) -> _SimpleResponse:
            return self._call("GET", path, **kwargs)

        def post(self, path: str, **kwargs: Any) -> _SimpleResponse:
            return self._call("POST", path, **kwargs)

        def delete(self, path: str, **kwargs: Any) -> _SimpleResponse:
            return self._call("DELETE", path, **kwargs)

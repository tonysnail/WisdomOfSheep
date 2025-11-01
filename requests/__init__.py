"""Minimal stand-in for the requests package used in tests."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Optional


def _load_real():
    original = list(sys.path)
    this_dir = Path(__file__).resolve().parent
    try:
        sys.path = [p for p in original if Path(p or "").resolve() != this_dir]
        spec = importlib.util.find_spec("requests")
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


_REAL = _load_real()
if _REAL is not None:
    globals().update({k: getattr(_REAL, k) for k in dir(_REAL)})
else:
    class Response:
        def __init__(self, status_code: int = 200, text: str = "", json_data: Optional[Any] = None) -> None:
            self.status_code = status_code
            self.text = text
            self._json_data = json_data

        def json(self) -> Any:
            if self._json_data is not None:
                return self._json_data
            raise ValueError("No JSON data available")

    class RequestException(Exception):
        pass

    def _not_available(*args: Any, **kwargs: Any) -> Response:
        raise RequestException("network operations are unavailable in test shim")

    get = _not_available
    post = _not_available
    put = _not_available
    patch = _not_available
    head = _not_available
    Session = object

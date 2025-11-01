"""Lightweight subset of pydantic used for local testing."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict


def _load_real():
    original = list(sys.path)
    this_dir = Path(__file__).resolve().parent
    try:
        sys.path = [p for p in original if Path(p or "").resolve() != this_dir]
        spec = importlib.util.find_spec("pydantic")
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
    class BaseModel:
        def __init__(self, **data: Any) -> None:
            annotations = getattr(self, "__annotations__", {})
            for name in annotations:
                if name in data:
                    value = data[name]
                else:
                    value = getattr(self.__class__, name, None)
                setattr(self, name, value)
            for key, value in data.items():
                if key not in annotations:
                    setattr(self, key, value)

        def model_dump(self) -> Dict[str, Any]:
            def _convert(value: Any) -> Any:
                if hasattr(value, "model_dump"):
                    return value.model_dump()
                if isinstance(value, list):
                    return [_convert(item) for item in value]
                if isinstance(value, dict):
                    return {k: _convert(v) for k, v in value.items()}
                return value

            annotations = getattr(self, "__annotations__", {})
            return {name: _convert(getattr(self, name, None)) for name in annotations}

    def Field(default: Any = None, **kwargs: Any) -> Any:  # pragma: no cover - no validation
        return default

    def field_validator(*fields: str, **kwargs: Any):  # pragma: no cover - passthrough decorator
        def decorator(func):
            return func
        return decorator

    def conint(*args: Any, **kwargs: Any):  # pragma: no cover - type alias
        return int

    def confloat(*args: Any, **kwargs: Any):  # pragma: no cover - type alias
        return float

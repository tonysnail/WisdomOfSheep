"""Tiny subset of NumPy required for unit tests."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def _load_real():
    original = list(sys.path)
    this_dir = Path(__file__).resolve().parent
    try:
        sys.path = [p for p in original if Path(p or "").resolve() != this_dir]
        spec = importlib.util.find_spec("numpy")
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
    import math

    def isnan(value: Any) -> bool:
        try:
            return math.isnan(float(value))
        except Exception:
            return False

    nan = float("nan")

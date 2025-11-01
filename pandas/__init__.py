"""Minimal pandas placeholder for offline tests."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_real():
    original = list(sys.path)
    this_dir = Path(__file__).resolve().parent
    try:
        sys.path = [p for p in original if Path(p or "").resolve() != this_dir]
        spec = importlib.util.find_spec("pandas")
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
    class DataFrame:  # pragma: no cover - placeholder
        pass

    def Series(*args, **kwargs):  # pragma: no cover - placeholder
        return []

"""Utilities for estimating council analysis durations."""
from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

_DEFAULT_MATRIX = {
    "xx00": 0.0,
    "xx01": 0.0,
    "xx02": 0.0,
    "xx11": 0.0,
    "xx12": 0.0,
    "xx22": 0.0,
}

_DEFAULT_VECTOR = {"xy0": 0.0, "xy1": 0.0, "xy2": 0.0}


@dataclass
class _ModelState:
    count: int = 0
    sum_duration: float = 0.0
    sum_article_tokens: float = 0.0
    sum_summary_tokens: float = 0.0
    sum_total_tokens: float = 0.0
    matrix: Dict[str, float] = None  # type: ignore[assignment]
    vector: Dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.matrix is None:
            self.matrix = dict(_DEFAULT_MATRIX)
        if self.vector is None:
            self.vector = dict(_DEFAULT_VECTOR)


class CouncilTimeModel:
    """Incremental linear regression for council analysis durations."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._state = self._load()

    # ------------------------- Persistence helpers -------------------------
    def _load(self) -> _ModelState:
        if not self._path.exists():
            return _ModelState()
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return _ModelState()

        state = _ModelState()
        state.count = int(payload.get("count", 0) or 0)
        state.sum_duration = float(payload.get("sum_duration", 0.0) or 0.0)
        state.sum_article_tokens = float(payload.get("sum_article_tokens", 0.0) or 0.0)
        state.sum_summary_tokens = float(payload.get("sum_summary_tokens", 0.0) or 0.0)
        state.sum_total_tokens = float(payload.get("sum_total_tokens", 0.0) or 0.0)

        matrix = payload.get("matrix")
        if isinstance(matrix, dict):
            state.matrix.update({key: float(matrix.get(key, 0.0) or 0.0) for key in _DEFAULT_MATRIX})
        vector = payload.get("vector")
        if isinstance(vector, dict):
            state.vector.update({key: float(vector.get(key, 0.0) or 0.0) for key in _DEFAULT_VECTOR})
        return state

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(".tmp")
        payload = {
            "count": self._state.count,
            "sum_duration": self._state.sum_duration,
            "sum_article_tokens": self._state.sum_article_tokens,
            "sum_summary_tokens": self._state.sum_summary_tokens,
            "sum_total_tokens": self._state.sum_total_tokens,
            "matrix": self._state.matrix,
            "vector": self._state.vector,
        }
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, self._path)

    # --------------------------- Public interface ---------------------------
    def predict(self, article_tokens: float, summary_tokens: float) -> Optional[float]:
        with self._lock:
            coeffs = self._coefficients_locked()
            if coeffs is not None:
                intercept, article_coeff, summary_coeff = coeffs
                estimate = intercept + article_coeff * article_tokens + summary_coeff * summary_tokens
                if math.isfinite(estimate) and estimate > 0:
                    return estimate
            # fallback to average rate per token if regression not available
            tokens = article_tokens + summary_tokens
            if tokens > 0 and self._state.sum_total_tokens > 0:
                rate = self._state.sum_duration / max(self._state.sum_total_tokens, 1e-9)
                estimate = rate * tokens
                if math.isfinite(estimate) and estimate > 0:
                    return estimate
            if self._state.count > 0:
                avg = self._state.sum_duration / max(self._state.count, 1)
                if math.isfinite(avg) and avg > 0:
                    return avg
        return None

    def observe(self, article_tokens: float, summary_tokens: float, duration_seconds: float) -> None:
        if not math.isfinite(duration_seconds) or duration_seconds <= 0:
            return
        article = float(max(article_tokens, 0.0))
        summary = float(max(summary_tokens, 0.0))
        with self._lock:
            self._state.count += 1
            self._state.sum_duration += duration_seconds
            self._state.sum_article_tokens += article
            self._state.sum_summary_tokens += summary
            self._state.sum_total_tokens += article + summary

            x0 = 1.0
            x1 = article
            x2 = summary
            self._state.matrix["xx00"] += x0 * x0
            self._state.matrix["xx01"] += x0 * x1
            self._state.matrix["xx02"] += x0 * x2
            self._state.matrix["xx11"] += x1 * x1
            self._state.matrix["xx12"] += x1 * x2
            self._state.matrix["xx22"] += x2 * x2

            self._state.vector["xy0"] += x0 * duration_seconds
            self._state.vector["xy1"] += x1 * duration_seconds
            self._state.vector["xy2"] += x2 * duration_seconds

            self._save()

    # ------------------------------ Internals ------------------------------
    def _coefficients_locked(self) -> Optional[Tuple[float, float, float]]:
        mat = self._state.matrix
        vec = self._state.vector
        a = mat["xx00"]
        b = mat["xx01"]
        c = mat["xx02"]
        d = mat["xx11"]
        e = mat["xx12"]
        f = mat["xx22"]

        matrix = (
            (a, b, c),
            (b, d, e),
            (c, e, f),
        )
        det = _determinant_3x3(matrix)
        if abs(det) < 1e-9:
            return None
        inv = _invert_3x3(matrix, det)
        if inv is None:
            return None
        b0 = vec["xy0"]
        b1 = vec["xy1"]
        b2 = vec["xy2"]
        coeffs = (
            inv[0][0] * b0 + inv[0][1] * b1 + inv[0][2] * b2,
            inv[1][0] * b0 + inv[1][1] * b1 + inv[1][2] * b2,
            inv[2][0] * b0 + inv[2][1] * b1 + inv[2][2] * b2,
        )
        if any(not math.isfinite(val) for val in coeffs):
            return None
        return coeffs


def _determinant_3x3(matrix: Tuple[Tuple[float, float, float], ...]) -> float:
    (a, b, c), (d, e, f), (g, h, i) = matrix
    return (
        a * (e * i - f * h)
        - b * (d * i - f * g)
        + c * (d * h - e * g)
    )


def _invert_3x3(
    matrix: Tuple[Tuple[float, float, float], ...],
    det: Optional[float] = None,
) -> Optional[Tuple[Tuple[float, float, float], ...]]:
    if det is None:
        det = _determinant_3x3(matrix)
    if abs(det) < 1e-9:
        return None
    (a, b, c), (d, e, f), (g, h, i) = matrix
    adj = (
        (e * i - f * h, -(b * i - c * h), b * f - c * e),
        (-(d * i - f * g), a * i - c * g, -(a * f - c * d)),
        (d * h - e * g, -(a * h - b * g), a * e - b * d),
    )
    inv_det = 1.0 / det
    return tuple(tuple(val * inv_det for val in row) for row in adj)


def approximate_token_count(text: Optional[str]) -> int:
    if not text:
        return 0
    stripped = text.strip()
    if not stripped:
        return 0
    # Count non-whitespace segments and fall back to char-based estimate.
    # This keeps behaviour deterministic without external tokenizers.
    segments = stripped.split()
    seg_count = len(segments)
    approx = max(int(math.ceil(len(stripped) / 4)), 1)
    return max(seg_count, approx)


__all__ = ["CouncilTimeModel", "approximate_token_count"]

"""Finite-difference helpers for gradient checks."""

from __future__ import annotations

from collections.abc import Callable


def central_difference(fn: Callable[[float], float], x: float, h: float = 1e-6) -> float:
    """Compute a scalar central finite-difference derivative."""

    return (fn(x + h) - fn(x - h)) / (2.0 * h)


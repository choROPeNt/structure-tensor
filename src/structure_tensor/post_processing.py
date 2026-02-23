
from __future__ import annotations

import logging
from typing import Any, Literal, Tuple

# ----------------------------
# Backend selection
# ----------------------------
try:
    import cupy as lib  # pyright: ignore[reportMissingImports]
    xp: Literal["cupy", "numpy"] = "cupy"
except ImportError:
    import numpy as lib  # type: ignore
    xp = "numpy"

Array = Any


def _as_float(x: Array, dtype=None) -> Array:
    """Ensure floating dtype for stable metrics."""
    if dtype is None:
        # sensible defaults: float32 on GPU, float64 on CPU
        dtype = lib.float32 if xp == "cupy" else lib.float64
    x = lib.asarray(x)
    if not lib.issubdtype(x.dtype, lib.floating):
        x = x.astype(dtype, copy=False)
    return x


def align_direction(
    v: Array,
    axes: Tuple[str, ...] = ("x", "y"),
    inplace: bool = False,
) -> Array:
    """
    Flip vectors so their projection onto the given axes is positive.

    Expected layout:
      - (3, X, Y, Z) components first
    """
    v = _as_float(v)

    if not inplace:
        v = v.copy()

    if not (v.ndim >= 2 and v.shape[0] == 3):
        raise ValueError(
            f"Expected v with shape (3, ...). Got {getattr(v, 'shape', None)}."
        )

    axis_map = {"x": 0, "y": 1, "z": 2}
    if not all(a in axis_map for a in axes):
        raise ValueError("axes must be any of 'x', 'y', 'z'")

    # reference direction in the selected subspace
    ref = lib.zeros((3, 1, 1, 1), dtype=v.dtype)  # broadcast over X,Y,Z
    for a in axes:
        ref[axis_map[a], 0, 0, 0] = 1.0
    ref /= lib.linalg.norm(ref)

    dot = lib.sum(v * ref, axis=0)          # (X,Y,Z)
    sign = lib.where(dot < 0, -1.0, 1.0)    # (X,Y,Z)
    v *= sign[None, ...]                    # broadcast to (3,X,Y,Z)

    return v
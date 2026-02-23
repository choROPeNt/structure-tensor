from __future__ import annotations

import logging
from typing import Any, Literal

# ----------------------------
# Backend selection
# ----------------------------
try:
    import cupy as lib  # pyright: ignore[reportMissingImports]
    xp: Literal["cupy", "numpy"] = "cupy"
    print("Using CuPy backend (GPU)")
except ImportError:
    import numpy as lib  # type: ignore
    xp = "numpy"
    print("Using NumPy backend (CPU)")

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


def metric_mse(a: Array, b: Array) -> float:
    a = _as_float(a)
    b = _as_float(b, dtype=a.dtype)
    diff = a - b
    return float(lib.mean(diff * diff))


def metric_rmse(a: Array, b: Array) -> float:
    a = _as_float(a)
    b = _as_float(b, dtype=a.dtype)
    diff = a - b
    return float(lib.sqrt(lib.mean(diff * diff)))


def cosine_similarity(
    vec1: Array,
    vec2: Array,
    axis: int = 0,
    sign_invariant: bool = False,
    eps: float = 1e-12,
    return_map: bool = False,
    mask: Array | None = None,
):
    """
    Cosine similarity between two vector fields.

    Parameters
    ----------
    vec1, vec2 : array
        Vector fields, typically shape (3, X, Y, Z) with vector-components on `axis`.
    axis : int
        Axis that holds the vector components.
    sign_invariant : bool
        If True, returns |cos| (useful for eigenvectors with ± ambiguity).
    eps : float
        Small value to avoid division by zero.
    return_map : bool
        If True, also return the cosine map.
    mask : array | None
        Optional boolean mask broadcastable to cosine map, to ignore invalid voxels.

    Returns
    -------
    mean_cos, min_cos, max_ang_deg [, cos_map]
    """

    v1 = _as_float(vec1)
    v2 = _as_float(vec2, dtype=v1.dtype)

    # Normalize to unit vectors (robust even if not already normalized)
    n1 = lib.linalg.norm(v1, axis=axis, keepdims=True)
    n2 = lib.linalg.norm(v2, axis=axis, keepdims=True)
    n1 = lib.maximum(n1, eps)
    n2 = lib.maximum(n2, eps)

    u1 = v1 / n1
    u2 = v2 / n2

    cos = lib.sum(u1 * u2, axis=axis)

    if sign_invariant:
        cos = lib.abs(cos)

    cos = lib.clip(cos, -1.0, 1.0)

    if mask is not None:
        # ensure boolean and broadcast
        m = lib.asarray(mask).astype(bool)
        cos_use = cos[m]
        if cos_use.size == 0:
            raise ValueError("Mask removed all elements; cannot compute stats.")
    else:
        cos_use = cos

    mean_cos = float(lib.mean(cos_use))
    min_cos = float(lib.min(cos_use))

    # max angle (deg) corresponding to the *smallest* cosine
    # arccos is monotonic decreasing on [-1,1]
    max_ang = float(lib.degrees(lib.arccos(lib.min(cos_use))))

    if return_map:
        return mean_cos, min_cos, max_ang, cos
    return mean_cos, min_cos, max_ang
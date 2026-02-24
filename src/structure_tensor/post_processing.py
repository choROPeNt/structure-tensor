
from __future__ import annotations

from typing import Any, Literal, Tuple, Optional

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



def align_direction(
    v: Array,
    axes: Optional[Tuple[str, ...]] = ("x", "y"),
    inplace: bool = False,
    dtype: lib.dtype | type = None,
) -> Array:
    """
    Flip vectors so their projection onto the given axes is positive.

    Expected layout:
      - (3, X, Y, Z) components first

    If axes is None or empty, returns v unchanged (or a copy if inplace=False).
    """
    v = lib.asarray(v)
    if dtype is not None:
        v = v.astype(dtype, copy=False)
    

    # no alignment requested
    if axes is None or len(axes) == 0:
        return v if inplace else v.copy()

    if not inplace:
        v = v.copy()


    if not (v.ndim >= 2 and v.shape[0] == 3):
        raise ValueError(f"Expected v with shape (3, ...). Got {getattr(v, 'shape', None)}.")

    axis_map = {"x": 0, "y": 1, "z": 2}
    if not all(a in axis_map for a in axes):
        raise ValueError("axes must be any of 'x', 'y', 'z'")

    # ref shape: (3, 1, 1, ..., 1) matching v.ndim
    ref_shape = (3,) + (1,) * (v.ndim - 1)
    ref = lib.zeros(ref_shape, dtype=v.dtype)
    for a in axes:
        ref[axis_map[a]] = 1.0

    # normalize ref safely (axes non-empty => norm > 0)
    # avoid float() to not sync on GPU
    n = lib.sqrt(lib.sum(ref * ref))
    ref = ref / n

    dot = lib.sum(v * ref, axis=0)
    sign = lib.where(dot < 0, -1.0, 1.0).astype(v.dtype, copy=False)

    # v *= sign[None, ...]
    lib.multiply(v, sign[None, ...], out=v)

    return  v 




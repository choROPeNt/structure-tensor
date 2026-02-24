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


def normalize(
    vol: Array,
    method: Literal["robust", "minmax", "zscore"] = "robust",
    percentiles: Tuple[float, float] = (1.0, 99.0),
    dtype=None,
) -> Array:
    """
    Normalize volume for structure tensor analysis.

    Parameters
    ----------
    vol : array
        Input volume (uint16 or float).
    method : {"robust", "minmax", "zscore"}
        Normalization strategy.
    percentiles : tuple
        Used for robust scaling.
    lib : numpy or cupy module (optional)
    dtype : target float dtype (default: float32)

    Returns
    -------
    vol_norm : array (float)
    """


    if dtype is None:
        dtype = lib.float32

    vol = lib.asarray(vol, dtype=dtype)

    if method == "robust":
        p_low, p_high = percentiles
        lo = lib.percentile(vol, p_low)
        hi = lib.percentile(vol, p_high)

        vol = lib.clip(vol, lo, hi)
        denom = hi - lo if (hi - lo) != 0 else 1.0
        vol = (vol - lo) / denom

    elif method == "minmax":
        lo = lib.min(vol)
        hi = lib.max(vol)
        denom = hi - lo if (hi - lo) != 0 else 1.0
        vol = (vol - lo) / denom

    elif method == "zscore":
        mean = lib.mean(vol)
        std = lib.std(vol)
        std = std if std != 0 else 1.0
        vol = (vol - mean) / std

    else:
        raise ValueError("method must be 'robust', 'minmax', or 'zscore'")

    return vol
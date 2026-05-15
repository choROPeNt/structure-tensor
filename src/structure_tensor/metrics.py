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


def aniso_orientation_plot(
    val: Array,
    vec: Array,
    aniso_kind: str = "fa",
    orientation: str = "polar",
    bins: int = 200,
    log_scale: bool = True,
    ax=None,
    cmap: str = "inferno",
    **hexbin_kwargs,
):
    """
    2-D density plot (hexbin) of anisotropy vs. eigenvector orientation.

    Parameters
    ----------
    val : array (3, ...)
        Eigenvalues in descending order, as returned by ``eig_special_3d``.
    vec : array (3, ...)
        Primary eigenvector (shape ``(3, ...)``), same spatial layout as ``val``.
    aniso_kind : str
        Anisotropy measure — passed directly to :func:`anisotropy`.
    orientation : str
        ``"polar"``   – angle from Z-axis in degrees, range [0°, 90°] (sign-invariant).
        ``"azimuth"`` – in-plane angle in XY, range [0°, 180°] (sign-invariant).
    bins : int
        ``gridsize`` for ``hexbin``.
    log_scale : bool
        Colour by ``log10(count + 1)`` rather than raw count.
    ax : matplotlib.axes.Axes | None
        Axes to draw into; creates a new figure if *None*.
    cmap : str
        Matplotlib colormap name.
    **hexbin_kwargs
        Forwarded to ``ax.hexbin``.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt  # lazy — keeps metrics importable without matplotlib

    vec = lib.asarray(vec)
    if vec.shape[0] != 3:
        raise ValueError(f"vec must have shape (3, ...). Got {vec.shape}.")

    vx, vy, vz = vec[0].ravel(), vec[1].ravel(), vec[2].ravel()

    orientation = orientation.lower()
    if orientation == "polar":
        # angle from Z, sign-invariant: arccos(|vz|)  ∈ [0°, 90°]
        vz_clip = lib.clip(lib.abs(vz), 0.0, 1.0)
        angle_rad = lib.arccos(vz_clip)
        xlabel = "Polar angle from Z (°)"
        xlim = (0, 90)
    elif orientation == "azimuth":
        # in-plane angle, sign-invariant: arctan2(|vy|, vx) mapped to [0°,180°]
        phi = lib.arctan2(vy, vx)         # [-π, π]
        phi = phi % lib.pi                # [0, π]  (fold negatives)
        angle_rad = phi
        xlabel = "Azimuth in XY (°)"
        xlim = (0, 180)
    else:
        raise ValueError(f"Unknown orientation {orientation!r}. Use 'polar' or 'azimuth'.")

    angle_deg = lib.degrees(angle_rad)
    fa = anisotropy(val, kind=aniso_kind).ravel()

    # move to numpy if on GPU
    try:
        angle_deg = angle_deg.get()
        fa = fa.get()
    except AttributeError:
        pass

    import numpy as np
    angle_deg = np.asarray(angle_deg, dtype=np.float64)
    fa = np.asarray(fa, dtype=np.float64)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    bins_kwarg = hexbin_kwargs.pop("gridsize", bins)
    reduce_fn = (lambda x: lib.log10(x + 1)) if log_scale else None

    hb = ax.hexbin(
        angle_deg,
        fa,
        gridsize=bins_kwarg,
        cmap=cmap,
        bins="log" if log_scale else None,
        mincnt=1,
        **hexbin_kwargs,
    )
    cb = ax.get_figure().colorbar(hb, ax=ax)
    cb.set_label("log₁₀(count + 1)" if log_scale else "count")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"Anisotropy ({aniso_kind.upper()})")
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1)

    return ax


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


def anisotropy(
    val: Array,
    kind: str = "fa",
    eps: float = 1e-12,
) -> Array:
    """
    Compute a scalar anisotropy measure from structure-tensor eigenvalues.

    Parameters
    ----------
    val : array (3, ...)
        Eigenvalues in descending order (λ1 ≥ λ2 ≥ λ3), as returned by
        ``eig_special_3d`` with ``eigenvalue_order="desc"`` (the default).
    kind : str
        Measure to compute:

        - ``"fa"``                – Fractional Anisotropy, range [0, 1]  *(default)*
        - ``"linear"`` / ``"cl"`` – (λ1 − λ2) / λ1  (fiber-like)
        - ``"planar"`` / ``"cp"`` – (λ2 − λ3) / λ1  (plate-like)
        - ``"spherical"`` / ``"cs"`` – λ3 / λ1                    (isotropic)
        - ``"ratio"``                – max(1 − λ_min/λ_max, 0)   range [0, 1)

        The three Westin shape measures sum to 1: cl + cp + cs = 1.
    eps : float
        Guard against division by zero when λ1 ≈ 0.

    Returns
    -------
    Array with shape ``val.shape[1:]``.
    """
    val = _as_float(lib.asarray(val))
    if val.shape[0] != 3:
        raise ValueError(f"val must have shape (3, ...). Got {val.shape}.")

    l1, l2, l3 = val[0], val[1], val[2]
    kind = kind.lower()

    if kind == "fa":
        mu = (l1 + l2 + l3) / 3
        num = lib.sqrt((l1 - mu) ** 2 + (l2 - mu) ** 2 + (l3 - mu) ** 2)
        den = lib.sqrt(l1 ** 2 + l2 ** 2 + l3 ** 2)
        return lib.sqrt(lib.asarray(1.5, dtype=val.dtype)) * num / lib.maximum(den, eps)

    if kind in ("linear", "cl"):
        return (l1 - l2) / lib.maximum(l1, eps)

    if kind in ("planar", "cp"):
        return (l2 - l3) / lib.maximum(l1, eps)

    if kind in ("spherical", "cs"):
        return l3 / lib.maximum(l1, eps)

    if kind == "ratio":
        # max(1 - λ_min/λ_max, 0)  — 0 when isotropic, →1 when λ_min≪λ_max
        # val is descending: l1=λ_max, l3=λ_min
        return lib.maximum(1.0 - l3 / lib.maximum(l1, eps), 0.0)

    raise ValueError(
        f"Unknown anisotropy kind {kind!r}. "
        "Choose from: 'fa', 'linear'/'cl', 'planar'/'cp', 'spherical'/'cs', 'ratio'."
    )


def cosine_similarity(
    vec1: Array,
    vec2: Array,
    axis: int = 0,
    sign_invariant: bool = False,
    eps: float = 1e-12,
    return_map: bool = False,
    mask: Array | None = None,
) -> Tuple[float,...]:
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
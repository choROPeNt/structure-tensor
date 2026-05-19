import argparse
import logging
import queue
import threading
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

# HDF5 + numeric
import h5py as h5
from structure_tensor.h5_io import H5BlockReader, H5BlockWriter

from structure_tensor.pre_processing import normalize
## need something with padding
from structure_tensor.post_processing import align_direction


# --- backend selection (NumPy CPU vs CuPy GPU) -------------------------------
try:
    import cupy as lib  # pyright: ignore[reportMissingImports]
    xp = "cupy"
    from structure_tensor.cp import structure_tensor_3d, eig_special_3d
    from cupyx.scipy.ndimage import gaussian_filter as _gaussian_filter  # pyright: ignore[reportMissingImports]
except ImportError:
    import numpy as lib
    xp = "numpy"
    from structure_tensor import structure_tensor_3d, eig_special_3d
    from scipy.ndimage import gaussian_filter as _gaussian_filter



def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger()


def get_device_string() -> str:
    """Return a human-readable device/backend string."""
    if xp == "cupy":
        try:
            dev = lib.cuda.runtime.getDevice()  # type: ignore
            props = lib.cuda.runtime.getDeviceProperties(dev) # type: ignore
            name = props.get("name", b"").decode("utf-8", errors="replace")
            mem_gb = props.get("totalGlobalMem", 0) / (1024**3)
            return f"GPU (CuPy) | id={dev} | {name} | {mem_gb:.1f} GB"
        except Exception:
            return "GPU (CuPy)"
    return "CPU (NumPy)"


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    if not path.is_file():
        raise ValueError(f"Config path is not a file: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("YAML config must be a mapping (top-level dict).")
    return cfg


def require(cfg: dict, key: str):
    if key not in cfg:
        raise KeyError(f"Missing required config key: '{key}'")
    return cfg[key]


def build_output_specs(
    in_path: Path,
    keys_in: tuple[str, ...] = ("volume",),
    block_size: tuple[int, int, int] = (128, 128, 128),
    chunk_size: tuple[int, int, int] = (128, 128, 128),  # storage chunks
) -> dict:
    if len(block_size) != 3 or len(chunk_size) != 3:
        raise ValueError("block_size and chunk_size must be (Z,Y,X)")

    bz, by, bx = map(int, block_size)
    cz, cy, cx = map(int, chunk_size)

    in_key = keys_in[0] if keys_in else "volume"

    with h5.File(in_path, "r") as F:
        obj = F[in_key]
        if not isinstance(obj, h5.Dataset):
            raise TypeError(f"'{in_key}' is not a Dataset in {in_path}")
        vol_shape = obj.shape
        vol_dtype = obj.dtype




    out_vec_shape = (3,) + vol_shape

    specs = {
        "vec": dict(
            shape=out_vec_shape,
            dtype=lib.float32,                
            chunks=(3, cz, cy, cx),
            # compression="gzip",
            # compression_opts=4,
        ),
        # "vol": dict(
        #     shape=vol_shape,
        #     dtype=vol_dtype,                  # keep input dtype
        #     chunks=(cz, cy, cx),
        #     compression="gzip",
        #     compression_opts=4,
        # ),
        "eig": dict(
            shape=(3,) + vol_shape,   # (λ1, λ2, λ3) per voxel
            dtype=lib.float32,
            chunks=(3, cz, cy, cx),
            # compression="gzip",
            # compression_opts=4,
        ),
    }
    return specs


def edge_aware_smooth_vec(v, iters=2, sigma_theta_deg=24.0, eps=1e-12):
    """
    Edge-aware smoothing for a line field (v ≡ -v). Preserves 90° jumps.
    v: (3,X,Y,Z)
    """
    v = v.copy()
    sigma_theta = lib.deg2rad(sigma_theta_deg)

    def normed(x):
        n = lib.linalg.norm(x, axis=0, keepdims=True)
        return x / (lib.maximum(n, eps))

    v = normed(v)

    # 6-neighborhood shifts
    shifts = [(+1,0,0),(-1,0,0),(0,+1,0),(0,-1,0),(0,0,+1),(0,0,-1)]

    for _ in range(iters):
        acc = lib.zeros_like(v)
        wsum = lib.zeros(v.shape[1:], dtype=v.dtype)

        for dx,dy,dz in shifts:
            vn = lib.roll(v, shift=(dx,dy,dz), axis=(1,2,3))

            # sign-invariant angle via abs(dot)
            c = lib.abs(lib.sum(v * vn, axis=0))
            c = lib.clip(c, 0.0, 1.0)
            theta = lib.arccos(c)

            w = lib.exp(-(theta*theta) / (2*sigma_theta*sigma_theta)).astype(v.dtype)

            acc += vn * w[None, ...]
            wsum += w

        # include self weight
        acc += v
        wsum += 1.0

        v = acc / wsum[None, ...]
        v = normed(v)

    return v



def _to_np(arr):
    """Move array to CPU numpy (no-op if already numpy)."""
    return arr.get() if xp == "cupy" else arr


def compute_global_percentiles(
    in_path: Path,
    key: str,
    p_low: float = 1.0,
    p_high: float = 99.0,
    stride: int = 4,
) -> tuple[float, float]:
    """Sample the volume at `stride` to estimate a robust global intensity range.

    Excludes zero-valued voxels so background zeros don't pull the low percentile down.
    """
    import numpy as np_cpu
    with h5.File(in_path, "r") as F:
        data = F[key][::stride, ::stride, ::stride]
    data = np_cpu.asarray(data, dtype=np_cpu.float32).ravel()
    data = data[data > 0]
    lo = float(np_cpu.percentile(data, p_low))
    hi = float(np_cpu.percentile(data, p_high))
    return lo, hi


def save_block_figure(vol_norm, vec, val, zsl, ysl, xsl, fig_dir: Path,
                      vol_raw=None, bg=None) -> Path:
    """Save start / center / end Z-slices to a PNG.

    vol_raw and bg are optional; when provided they appear as extra rows above
    vol_norm so the effect of background correction is immediately visible.
    """
    import numpy as np_cpu

    def to_np(a):
        return np_cpu.asarray(_to_np(a))

    z_idx    = [0, vol_norm.shape[0] // 2, vol_norm.shape[0] - 1]
    z_labels = ["start", "center", "end"]

    rows = []
    if vol_raw is not None:
        rows.append(("vol_raw",  to_np(vol_raw), "gray"))
    if bg is not None:
        rows.append(("bg",       to_np(bg),      "gray"))
    rows += [
        ("vol_norm", to_np(vol_norm), "gray"),
        ("vec_x",    to_np(vec[0]),   "RdBu"),
        ("vec_y",    to_np(vec[1]),   "RdBu"),
        ("vec_z",    to_np(vec[2]),   "RdBu"),
        ("eig_0",    to_np(val[0]),   "plasma"),
    ]

    fig, axes = plt.subplots(len(rows), 3, figsize=(11, 4 * len(rows)), constrained_layout=True)
    fig.suptitle(
        f"Block  z={zsl.start}:{zsl.stop}  y={ysl.start}:{ysl.stop}  x={xsl.start}:{xsl.stop}",
        fontsize=10,
    )

    for r, (row_label, data, cmap) in enumerate(rows):
        vmin, vmax = float(data.min()), float(data.max())
        for c, (zi, zlabel) in enumerate(zip(z_idx, z_labels)):
            ax = axes[r, c]
            im = ax.imshow(data[zi], cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", interpolation="nearest")
            if r == 0:
                ax.set_title(f"{zlabel}  (z={zi})", fontsize=8)
            if c == 0:
                ax.set_ylabel(row_label, fontsize=8)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / f"block_z{zsl.start:05d}_y{ysl.start:05d}_x{xsl.start:05d}.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fig_path


# =============================================================================
# Main
# =============================================================================

def main(config_path: Path, plot: bool = False) -> None:
    logger = setup_logging()
    cfg = load_yaml(config_path)

    logger.info("Loaded config: %s", config_path.resolve())
    logger.info("Backend/device: %s", get_device_string())

    in_path = Path(require(cfg, "file_path"))
    raw_internal_path = require(cfg, "raw_internal_path")
    out_path = Path(cfg.get("out_path", in_path.with_suffix(".test.vec.h5")))

    logger.info("Input file: %s", in_path)
    logger.info("Internal dataset path: %s", raw_internal_path)
    logger.info("Output file: %s", out_path)

    # block size (Z,Y,X) from YAML
    block_size = tuple(cfg.get("block_size", [128, 128, 128]))
    if len(block_size) != 3:
        raise ValueError("block_size must be length 3 (Z,Y,X)")
    block_size = (int(block_size[0]), int(block_size[1]), int(block_size[2]))

    keys_in = (raw_internal_path,)
    specs = build_output_specs(in_path=in_path, keys_in=keys_in, block_size=block_size)

    logger.info("Block size (Z,Y,X): %s", block_size)
    # logger.info("Detected volume shape (Z,Y,X): %s", specs["vol"]["shape"])
    logger.info("Planned vec output shape (3,Z,Y,X): %s", specs["vec"]["shape"])
    # logger.info("Planned chunks vec: %s | vol: %s", specs["vec"]["chunks"], specs["vol"]["chunks"])

    # --- Parameters from YAML -------------------------------------------------
    voxel_size = float(require(cfg, "voxel_size"))         # mm/px
    fiber_diameter = float(require(cfg, "fiber_diameter")) # mm

    # Gaussian params
    r = fiber_diameter / 2 / voxel_size
    sigma = round(float(lib.sqrt(r**2 / 2)), 2)
    rho = round(3 * sigma, 2)

    axes           = tuple(cfg.get("axes", ["x", "z"]))
    mask_threshold = float(cfg.get("mask_threshold", 0.0))
    bg_sigma       = float(cfg.get("bg_sigma", 0.0))   # 0 = disabled

    logger.info("Params: voxel_size=%g mm/px | fiber_diameter=%g mm", voxel_size, fiber_diameter)
    logger.info("Mask threshold: %g (voxels <= this treated as background)", mask_threshold)
    logger.info("Background correction: bg_sigma=%g %s", bg_sigma, "(disabled)" if bg_sigma == 0 else "")
    logger.info("Gaussian: r=%g px | sigma=%g | rho=%g | axes=%s", r, sigma, rho, axes)

    fig_dir = out_path.parent / f"{out_path.stem}_figs" if plot else None
    if plot:
        logger.info("Plot mode: figures -> %s", fig_dir)

    # --- Global intensity normalization ---------------------------------------
    norm_p_low  = float(cfg.get("norm_p_low",  1.0))
    norm_p_high = float(cfg.get("norm_p_high", 99.0))
    logger.info("Sampling global percentiles (p%.1f / p%.1f) ...", norm_p_low, norm_p_high)
    global_lo, global_hi = compute_global_percentiles(
        in_path, raw_internal_path, p_low=norm_p_low, p_high=norm_p_high
    )
    global_scale = max(global_hi - global_lo, 1e-8)
    logger.info("Global intensity range: lo=%.4g  hi=%.4g", global_lo, global_hi)

    # --- Processing loop ------------------------------------------------------
    _SENTINEL = object()

    def _fetch_blocks(reader, q: queue.Queue) -> None:
        try:
            for slices, batch in iter(reader):  # type: ignore
                q.put((slices, batch))
        finally:
            q.put(_SENTINEL)

    with H5BlockReader(in_path, keys_in, block_size=block_size, dtype=lib.float32, strict=False) as reader, \
         H5BlockWriter(out_path, specs, mode="w") as writer:

        prefetch_q: queue.Queue = queue.Queue(maxsize=2)
        fetch_thread = threading.Thread(target=_fetch_blocks, args=(reader, prefetch_q), daemon=True)
        fetch_thread.start()

        while True:
            item = prefetch_q.get()
            if item is _SENTINEL:
                break
            (zsl, ysl, xsl), batch = item

            vol = batch.get(raw_internal_path)
            if vol is None:
                vol = batch.get("volume")
            if vol is None:
                continue
            vol = lib.asarray(vol)
            logger.info(
                "Analysing | block=%s,%s,%s | shape=%s | dtype=%s | size=%.2f MB",
                zsl, ysl, xsl,
                vol.shape,
                vol.dtype,
                vol.nbytes / 1024**2
            )

            mask    = vol > mask_threshold
            vol     = vol * mask
            vol_raw = vol  # keep for plotting before any correction

            bg = None
            if bg_sigma > 0:
                # subtract slowly-varying background (beam hardening / cupping)
                bg   = _gaussian_filter(vol.astype(lib.float64), sigma=bg_sigma)
                vol  = (vol.astype(lib.float64) - bg).astype(lib.float32)
                vol *= mask   # re-apply mask after subtraction
                vol_norm = lib.clip(vol, 0.0, vol.max()) / lib.maximum(vol.max(), 1e-8)
            else:
                vol_norm = lib.clip(vol, global_lo, global_hi)
                vol_norm = (vol_norm - global_lo) / global_scale

            S = structure_tensor_3d(vol_norm, sigma, rho)
            val, vec = eig_special_3d(S, full=False)  # expect vec: (3, bz, by, bx)

            # zero background in output (ST Gaussian may bleed across the boundary)
            mask_bc  = mask[lib.newaxis]                 # (1, Z, Y, X)
            vec     *= mask_bc
            val     *= mask_bc

            # Debug only if suspicious
            maxabs_pre = float(lib.max(lib.abs(vec)))
            finite_ok = bool(lib.isfinite(vec).all())
            if (not finite_ok) or (maxabs_pre > 1.5):
                logger.warning(
                                "pre-align: z=%s, y=%s, x=%s finite=%s maxabs=%g",
                                zsl, ysl, xsl, finite_ok, maxabs_pre
                            )

            vec = align_direction(vec, axes=axes)

            # Safe renormalize + clamp
            l = lib.linalg.norm(vec, axis=0, keepdims=True)
            vec = lib.where(l > 1e-12, vec / l, 0.0)
            lib.clip(vec, -1.0, 1.0, out=vec)

            maxabs_post = float(lib.max(lib.abs(vec)))
            if maxabs_post > 1.01:
                print(f"[WARN] post-norm: z={zsl}, y={ysl}, x={xsl} maxabs={maxabs_post}")

            # vec = edge_aware_smooth_vec(vec, iters=20, sigma_theta_deg=24)

            if plot:
                fig_path = save_block_figure(
                    vol_norm, vec, val, zsl, ysl, xsl, fig_dir,
                    vol_raw=vol_raw, bg=bg,
                )
                logger.info("Saved figure: %s", fig_path)

            writer.write_block("vec", zsl,ysl,xsl, vec.astype(lib.float32, copy=False))
            # writer.write_block("vol", zsl,ysl,xsl, vol.astype(lib.uint16, copy=False))
            writer.write_block("eig", zsl,ysl,xsl, val.astype(lib.float32, copy=False))

        fetch_thread.join()

    logger.info("Finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse a Volume via Batches.")
    parser.add_argument(
        "-c","--config",
        type=Path,
        required=True,
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "-p", "--plot",
        action="store_true",
        help="Save per-block slice figures (start/center/end Z) alongside the output.",
    )
    args = parser.parse_args()
    main(args.config, plot=args.plot)
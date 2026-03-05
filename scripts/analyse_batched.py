import argparse
import logging
from pathlib import Path

import yaml

# HDF5 + numeric
import h5py as h5
from structure_tensor.h5_io import H5BlockReader, H5BlockWriter

from structure_tensor.st3d import eigh_baseline_3d, S6_to_mat33

from structure_tensor.pre_processing import normalize
## need something with padding
from structure_tensor.post_processing import align_direction


# --- backend selection (NumPy CPU vs CuPy GPU) -------------------------------
try:
    import cupy as lib  # pyright: ignore[reportMissingImports]
    xp = "cupy"
    from structure_tensor.cp import structure_tensor_3d, eig_special_3d
except ImportError:
    import numpy as lib
    xp = "numpy"
    from structure_tensor import structure_tensor_3d, eig_special_3d



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
) -> dict:
    """
    Inspect the input HDF5 once and build dataset specs for outputs.

    Returns:
        specs: dict describing output datasets, e.g. specs["vec"], specs["vol"]
    """
    if len(block_size) != 3:
        raise ValueError("block_size must be length 3 (Z,Y,X)")
    bz, by, bx = (int(block_size[0]), int(block_size[1]), int(block_size[2]))

    # Use the first key as the primary input dataset
    in_key = keys_in[0] if len(keys_in) > 0 else "volume"

    with h5.File(in_path, "r") as F:
        if in_key not in F:
            raise KeyError(f"Missing dataset '{in_key}' in: {in_path}")

        obj = F[in_key]
        if not isinstance(obj, h5.Dataset):
            raise TypeError(f"'{in_key}' is not an HDF5 dataset in: {in_path}")

        vol_shape = obj.shape  # (Z, Y, X)
        if len(vol_shape) != 3:
            raise ValueError(f"Expected '{in_key}' shape (Z,Y,X). Got: {vol_shape}")

        out_vec_shape = (3,) + vol_shape  # (3, Z, Y, X)

    specs = {
        "vec": dict(
            shape=out_vec_shape,
            dtype=lib.float32,           # works for numpy/cupy
            chunks=(3, bz, by, bx),      # (C, Z, Y, X)
            compression="gzip",
            compression_opts=4,
        ),
        "vol": dict(
            shape=vol_shape,
            dtype=lib.uint16,            # works for numpy/cupy
            chunks=(bz, by, bx),         # (Z, Y, X)
            compression="gzip",
            compression_opts=4,
        ),
        "val": dict(
            shape=out_vec_shape,
            dtype=lib.float32,            # works for numpy/cupy
            chunks=(3, bz, by, bx),         # (Z, Y, X)
            compression="gzip",
            compression_opts=4,
        ),
    }
    return specs


# =============================================================================
# Main
# =============================================================================

def main(config_path: Path) -> None:
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
    logger.info("Detected volume shape (Z,Y,X): %s", specs["vol"]["shape"])
    logger.info("Planned vec output shape (3,Z,Y,X): %s", specs["vec"]["shape"])
    logger.info("Planned chunks vec: %s | vol: %s", specs["vec"]["chunks"], specs["vol"]["chunks"])

    # --- Parameters from YAML -------------------------------------------------
    voxel_size = float(require(cfg, "voxel_size"))         # µm/px
    fiber_diameter = float(require(cfg, "fiber_diameter")) # µm

    # Gaussian params
    r = fiber_diameter / 2 / voxel_size
    sigma = round(float(lib.sqrt(r**2 / 2)), 2)
    rho = 4 * sigma

    axes = tuple(cfg.get("axes", ["y", "z"]))

    logger.info("Params: voxel_size=%g µm/px | fiber_diameter=%g µm", voxel_size, fiber_diameter)
    logger.info("Gaussian: r=%g px | sigma=%g | rho=%g | axes=%s", r, sigma, rho, axes)

    # --- Processing loop ------------------------------------------------------
    # Assumes these exist in your project namespace:
    #   normalize, structure_tensor_3d, S6_to_mat33, eigh_baseline_3d,
    #   align_direction, edge_aware_smooth_vec
    with H5BlockReader(in_path, keys_in, block_size=block_size, dtype=lib.float32, strict=False) as reader, \
         H5BlockWriter(out_path, specs, mode="w") as writer:

        for (zsl, ysl, xsl), batch in iter(reader): # type: ignore

            vol = batch.get(raw_internal_path)
            if vol is None:
                vol = batch.get("volume")
            if vol is None:
                continue
            logger.info(
                "Analysing | block=%s,%s,%s | shape=%s | dtype=%s | size=%.2f MB",
                zsl, ysl, xsl,
                vol.shape,
                vol.dtype,
                vol.nbytes / 1024**2
            )

            vol_norm = normalize(vol, method="robust")

            S = structure_tensor_3d(vol_norm, sigma, rho)
            # S = S6_to_mat33_for_eigh(S)
            # print(S.shape,S.dtype)
            val, vec = eig_special_3d(S, full=False)  # expect vec: (3, bz, by, bx)

            print(val.shape)
            print(val[:3,0,0,0])
            print(val[:3,64,64,64])
            # Debug only if suspicious
            maxabs_pre = float(lib.max(lib.abs(vec)))
            finite_ok = bool(lib.isfinite(vec).all())
            if (not finite_ok) or (maxabs_pre > 1.5):
                print(f"[WARN] pre-align: z={zsl}, y={ysl}, x={xsl} finite={finite_ok} maxabs={maxabs_pre}")

            vec = align_direction(vec, axes=axes)

            # Safe renormalize + clamp
            l = lib.linalg.norm(vec, axis=0, keepdims=True)
            vec = lib.where(l > 1e-12, vec / l, 0.0)
            lib.clip(vec, -1.0, 1.0, out=vec)

            maxabs_post = float(lib.max(lib.abs(vec)))
            if maxabs_post > 1.01:
                print(f"[WARN] post-norm: z={zsl}, y={ysl}, x={xsl} maxabs={maxabs_post}")

            # vec = edge_aware_smooth_vec(vec, iters=100, sigma_theta_deg=24)


            writer.write_block("vec", zsl,ysl,xsl, vec.astype(lib.float32, copy=False))
            writer.write_block("vol", zsl,ysl,xsl, vol.astype(lib.uint16, copy=False))
            writer.write_block("val", zsl,ysl,xsl, val.astype(lib.float32, copy=False))

    logger.info("Finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse a Volume via Batches.")
    parser.add_argument(
        "--config",
        type=Path,  
        required=True,
        help="Path to the YAML config file.",
    )
    args = parser.parse_args()
    main(args.config)
from pathlib import Path

import h5py
import numpy as np
import yaml

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.ndimage import (
    binary_opening,
    generate_binary_structure,
    grey_dilation,
    gaussian_filter,
)

from structure_tensor.h5_io import H5BlockReader, H5BlockWriter
from structure_tensor.metrics import anisotropy
from structure_tensor.post_processing import align_direction
from structure_tensor.xdmf_io import write_xdmf_for_h5


def _vol_shape(path: str, key: str):
    """Return (Z, Y, X) of a 3-D or component-first 4-D dataset."""
    with h5py.File(path, "r") as f:
        obj = f[key]
        if not isinstance(obj, h5py.Dataset):
            raise TypeError(f"'{key}' is not a Dataset in {path}")
        if obj.ndim == 3:
            return tuple(int(d) for d in obj.shape)
        elif obj.ndim == 4:
            return tuple(int(d) for d in obj.shape[1:])
        raise ValueError(f"Unexpected ndim={obj.ndim} for '{key}'")


def iter_blocks(config: dict):
    """
    Yield (slices, batch) for every spatial block, reading vol and analysis
    data from their respective H5 files in lock-step.

    batch keys: vol_key + all analysis_data keys
    """
    vol_path   = config["vol_data"]["path"]
    vol_key    = config["vol_data"]["key"]
    ana_path   = config["analysis_data"]["path"]
    ana_keys   = tuple(config["analysis_data"]["keys"])
    block_size = tuple(config.get("block_size", [64, 256, 256]))

    with (
        H5BlockReader(vol_path, keys=(vol_key,), block_size=block_size) as vol_r,
        H5BlockReader(ana_path, keys=ana_keys,   block_size=block_size) as ana_r,
    ):
        for (slices, vol_batch), (_, ana_batch) in zip(vol_r, ana_r):
            yield slices, {**vol_batch, **ana_batch}


def build_output_specs(config: dict, vol_shape: tuple) -> dict:
    """
    Build the H5BlockWriter specs from the output section of the config.

    Config example:
        output:
          path: results.h5
          datasets:
            seg:
              dtype: uint8
              components: 1       # scalar → (Z,Y,X); >1 → (C,Z,Y,X)
            vec_masked:
              dtype: float32
              components: 3
    """
    block_size = tuple(config.get("block_size", [64, 256, 256]))
    specs      = {}

    for key, ds_cfg in config["output"]["datasets"].items():
        dtype      = np.dtype(ds_cfg.get("dtype", "float32"))
        components = int(ds_cfg.get("components", 1))

        if components == 1:
            shape  = vol_shape
            chunks = block_size
        else:
            shape  = (components,) + vol_shape
            chunks = (components,) + block_size

        specs[key] = dict(
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            compression="gzip",
        )

    return specs


def extract_features(eig_s: np.ndarray, vec_s: np.ndarray, raw_s: np.ndarray) -> np.ndarray:
    """
    Build per-voxel feature matrix from structure-tensor outputs.

    Parameters
    ----------
    eig_s : (3, N) or (3, Z, Y, X)  eigenvalues descending
    vec_s : (3, N) or (3, Z, Y, X)  primary eigenvector
    raw_s : (N,)  or (Z, Y, X)      smoothed raw intensity

    Returns
    -------
    X : (N, 6)  float32  [fa, cs, vx², vy², vz², raw]
    """
    fa = anisotropy(eig_s, kind="fa").ravel().astype(np.float32)
    cs = anisotropy(eig_s, kind="spherical").ravel().astype(np.float32)
    vx, vy, vz = vec_s[0].ravel(), vec_s[1].ravel(), vec_s[2].ravel()
    return np.column_stack([
        fa, cs,
        (vx * vx).astype(np.float32),
        (vy * vy).astype(np.float32),
        (vz * vz).astype(np.float32),
        raw_s.ravel().astype(np.float32),
    ])


def _normed(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=0, keepdims=True)
    return v / np.maximum(n, eps)


def fit_pipeline(config: dict, vol_shape: tuple):
    """
    Pass 1 — stream all blocks, proportionally subsample voxels, fit
    StandardScaler → PCA → KMeans.

    Returns
    -------
    scaler, pca, km
    """
    vol_key    = config["vol_data"]["key"]
    sigma      = float(config.get("clustering", {}).get("sigma", 1.6))
    truncate   = float(config.get("clustering", {}).get("truncate", 4.0))
    n_sample   = int(config.get("clustering", {}).get("n_sample", 200_000))
    n_clusters = int(config.get("clustering", {}).get("n_clusters", 3))
    pca_var    = float(config.get("clustering", {}).get("pca_variance", 0.95))
    rng        = np.random.default_rng(config.get("clustering", {}).get("seed", 0))

    total_voxels = int(np.prod(vol_shape))
    samples = []
    collected = 0

    print(f"Pass 1 — fitting pipeline (n_sample={n_sample}, n_clusters={n_clusters})")

    for (zsl, ysl, xsl), batch in iter_blocks(config):
        vol = batch[vol_key]
        eig = batch["eig"]
        vec = batch["vec"]

        vol_smooth = gaussian_filter(vol.astype(np.float32), sigma=sigma, truncate=truncate)

        X_block = extract_features(eig, vec, vol_smooth)  # (N_block, 6)
        n_block = X_block.shape[0]

        # proportional allocation: how many voxels to keep from this block
        n_want = max(1, round(n_sample * n_block / total_voxels))
        n_pick = min(n_want, n_block)
        idx = rng.choice(n_block, size=n_pick, replace=False)
        samples.append(X_block[idx])
        collected += n_pick
        print(f"  block z={zsl} y={ysl} x={xsl}  sampled {n_pick}/{n_block}", end="\r")

    print(f"\nTotal sampled voxels: {collected}")

    X_sample = np.concatenate(samples, axis=0)

    # if we over-sampled, draw down to exactly n_sample
    if X_sample.shape[0] > n_sample:
        idx = rng.choice(X_sample.shape[0], size=n_sample, replace=False)
        X_sample = X_sample[idx]

    print(f"Fitting StandardScaler + PCA({pca_var}) + KMeans(k={n_clusters}) ...")
    scaler = StandardScaler().fit(X_sample)
    X_scaled = scaler.transform(X_sample)

    pca = PCA(n_components=pca_var, random_state=0).fit(X_scaled)
    X_pca = pca.transform(X_scaled)
    print(f"  PCA kept {pca.n_components_} components  ({pca_var*100:.0f}% variance)")

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(X_pca)
    print(f"  KMeans inertia: {km.inertia_:.3e}")

    return scaler, pca, km


def cleanup_segmentation(config: dict, out_path: str, vol_shape: tuple, n_clusters: int):
    """
    Pass 3 — morphological cleanup + Gaussian soft-voting on the written seg
    dataset.  Reads and overwrites the 'seg' dataset in-place using Z-batches
    with halo overlap so border voxels are handled correctly.
    """
    morpho_cfg  = config.get("morpho", {})
    batch_size  = int(morpho_cfg.get("batch_size", 32))
    gauss_sigma = float(morpho_cfg.get("gauss_sigma", 2.0))

    # per-cluster morph iterations (yaml may give int keys already)
    raw_iters   = morpho_cfg.get("morph_iters", {i: 4 for i in range(n_clusters)})
    morph_iters = {int(k): int(v) for k, v in raw_iters.items()}

    struct  = generate_binary_structure(3, 1)
    morph_o = (max(morph_iters.values()) + 1) if morph_iters else 0
    gauss_o = int(np.ceil(4 * gauss_sigma)) if gauss_sigma > 0 else 0
    overlap = morph_o + gauss_o
    D       = vol_shape[0]

    print(f"\nPass 3 — morpho cleanup (overlap={overlap}, gauss_sigma={gauss_sigma})")

    with h5py.File(out_path, "r+") as f:
        seg_ds = f["seg"]
        if not isinstance(seg_ds, h5py.Dataset):
            raise TypeError(f"'seg' is not a Dataset in {out_path}")

        for z0 in range(0, D, batch_size):
            z1  = min(z0 + batch_size, D)
            z0h = max(0, z0 - overlap)
            z1h = min(D,  z1 + overlap)

            batch_s = np.asarray(seg_ds[z0h:z1h], dtype=np.int16)
            clean_b = batch_s.copy()

            # step 1: binary opening per cluster → mark removed voxels as -1
            for k in range(n_clusters):
                iters = morph_iters.get(k, 0)
                if iters == 0:
                    continue
                mask   = batch_s == k
                opened = np.asarray(binary_opening(mask, structure=struct, iterations=iters), dtype=bool)
                clean_b[mask & ~opened] = -1

            # fill gaps iteratively with grey_dilation
            for _ in range(overlap + 4):
                gap = clean_b == -1
                if not gap.any():
                    break
                grown = grey_dilation(clean_b, size=3)
                clean_b[gap] = grown[gap]

            # step 2: Gaussian soft-voting for smooth boundaries
            if gauss_sigma > 0:
                scores = np.stack([
                    gaussian_filter((clean_b == k).astype(np.float32), sigma=gauss_sigma)
                    for k in range(n_clusters)
                ], axis=0)                          # (n_clusters, bZ, Y, X)
                clean_b = scores.argmax(axis=0).astype(np.int16)

            # crop halo and write back
            lo = z0 - z0h
            hi = z1h - z1
            seg_ds[z0:z1] = clean_b[lo : len(clean_b) - hi if hi else None].astype(np.uint8)
            print(f"  morpho   z={z0:4d}–{z1:4d}", end="\r")

    print("morpho done.          ")


# def process_vec(config: dict, out_path: str, vol_shape: tuple):
#     """
#     Pass 4 — align sign per label, edge-aware smooth, then zero-out label 0.

#     Sub-phases:
#       4a  align_direction per cluster label, write to vec_masked
#       4b  edge-aware smooth (Z-halo to avoid wrap artifacts)
#       4c  zero-out label-0 voxels (smooth can bleed into boundaries)
#     """
#     vec_cfg    = config.get("vec_processing", {})
#     batch_size = int(vec_cfg.get("batch_size", 32))

#     smooth_cfg      = vec_cfg.get("smooth", {})
#     smooth_iters    = int(smooth_cfg.get("iters", 2))
#     sigma_theta_deg = float(smooth_cfg.get("sigma_theta_deg", 24.0))
#     sigma_theta     = np.deg2rad(sigma_theta_deg)

#     # align config: {label: [axes]} — labels absent here are zeroed out
#     raw_align  = vec_cfg.get("align", {1: ["x"], 2: ["z"]})
#     align_cfg  = {int(k): list(v) for k, v in raw_align.items()}

#     ana_path = config["analysis_data"]["path"]
#     D        = vol_shape[0]
#     halo     = smooth_iters  # each iter propagates 1 Z-slice of information
#     shifts   = [(+1,0,0),(-1,0,0),(0,+1,0),(0,-1,0),(0,0,+1),(0,0,-1)]

#     # ── 4a: align ────────────────────────────────────────────────────────────
#     print(f"\nPass 4a — aligning vec per label ...")
#     with h5py.File(ana_path, "r") as af, h5py.File(out_path, "r+") as of:
#         vec_src = af["vec"]
#         seg_ds  = of["seg"]
#         vm_ds   = of["vec_masked"]
#         if not isinstance(vec_src, h5py.Dataset):
#             raise TypeError("'vec' is not a Dataset in analysis file")
#         if not isinstance(seg_ds, h5py.Dataset):
#             raise TypeError("'seg' is not a Dataset in output file")
#         if not isinstance(vm_ds, h5py.Dataset):
#             raise TypeError("'vec_masked' is not a Dataset in output file")

#         for z0 in range(0, D, batch_size):
#             z1    = min(z0 + batch_size, D)
#             vec_b = np.asarray(vec_src[:, z0:z1], dtype=np.float32)  # (3, bZ, Y, X)
#             seg_b = np.asarray(seg_ds[z0:z1], dtype=np.uint8)         # (bZ, Y, X)

#             vm_b = np.zeros_like(vec_b)
#             for label, axes in align_cfg.items():
#                 mask = seg_b == label
#                 if not mask.any():
#                     continue
#                 vm_b[:, mask] = align_direction(vec_b[:, mask], axes=axes)

#             vm_ds[:, z0:z1] = vm_b
#             print(f"  align z={z0:4d}–{z1:4d}", end="\r")
#     print("align done.          ")

#     # ── 4b: edge-aware smooth with Z halo ────────────────────────────────────
#     print(f"Pass 4b — edge-aware smooth (iters={smooth_iters}, σ={sigma_theta_deg}°) ...")
#     with h5py.File(out_path, "r+") as of:
#         vm_ds = of["vec_masked"]
#         if not isinstance(vm_ds, h5py.Dataset):
#             raise TypeError("'vec_masked' is not a Dataset in output file")

#         for z0 in range(0, D, batch_size):
#             z1  = min(z0 + batch_size, D)
#             z0h = max(0, z0 - halo)
#             z1h = min(D,  z1 + halo)

#             v = np.asarray(vm_ds[:, z0h:z1h], dtype=np.float32)  # (3, bZ+halo, Y, X)
#             v = _normed(v)

#             for _ in range(smooth_iters):
#                 acc  = np.zeros_like(v)
#                 wsum = np.zeros(v.shape[1:], dtype=v.dtype)

#                 for sdz, sdy, sdx in shifts:
#                     if sdz != 0:
#                         # Z-shift: zero-pad instead of roll to avoid boundary wrap
#                         vn = np.zeros_like(v)
#                         if sdz > 0:
#                             vn[:, sdz:] = v[:, :-sdz]
#                         else:
#                             vn[:, :sdz] = v[:, -sdz:]
#                     else:
#                         vn = np.roll(v, shift=(sdy, sdx), axis=(2, 3))

#                     c     = np.clip(np.abs(np.sum(v * vn, axis=0)), 0.0, 1.0)
#                     theta = np.arccos(c)
#                     w     = np.exp(-(theta ** 2) / (2 * sigma_theta ** 2)).astype(v.dtype)
#                     acc  += vn * w[None, ...]
#                     wsum += w

#                 acc += v; wsum += 1.0
#                 v    = _normed(acc / wsum[None, ...])

#             lo = z0 - z0h
#             hi = z1h - z1
#             vm_ds[:, z0:z1] = v[:, lo : v.shape[1] - hi if hi else None]
#             print(f"  smooth z={z0:4d}–{z1:4d}", end="\r")
#     print("edge-aware smooth done.")

#     # ── 4c: zero-out label-0 voxels (re-apply after smooth boundary bleed) ───
#     print("Pass 4c — masking label 0 ...")
#     with h5py.File(out_path, "r+") as of:
#         seg_ds = of["seg"]
#         vm_ds  = of["vec_masked"]
#         if not isinstance(seg_ds, h5py.Dataset) or not isinstance(vm_ds, h5py.Dataset):
#             raise TypeError("Dataset not found in output file")

#         for z0 in range(0, D, batch_size):
#             z1    = min(z0 + batch_size, D)
#             seg_b = np.asarray(seg_ds[z0:z1], dtype=np.uint8)
#             vm_b  = np.asarray(vm_ds[:, z0:z1], dtype=np.float32)
#             vm_b[:, seg_b == 0] = 0.0
#             vm_ds[:, z0:z1] = vm_b
#             print(f"  mask z={z0:4d}–{z1:4d}", end="\r")
#     print("mask done.          ")


def main(config: dict):
    required = ("vol_data", "analysis_data", "output")
    missing  = [k for k in required if k not in config]
    if missing:
        print(f"Config missing sections: {missing}")
        exit(1)

    vol_key    = config["vol_data"]["key"]
    sigma      = float(config.get("clustering", {}).get("sigma", 1.6))
    truncate   = float(config.get("clustering", {}).get("truncate", 4.0))
    n_clusters = int(config.get("clustering", {}).get("n_clusters", 3))

    vol_shape = _vol_shape(config["vol_data"]["path"], config["vol_data"]["key"])
    out_path  = config["output"]["path"]
    specs     = build_output_specs(config, vol_shape)

    print(f"Volume shape : {vol_shape}")
    print(f"Output       : {out_path}")
    for k, sp in specs.items():
        print(f"  {k:20s}  shape={sp['shape']}  dtype={sp['dtype']}")

    # ── Pass 1: fit ──────────────────────────────────────────────────────────
    scaler, pca, km = fit_pipeline(config, vol_shape)

    # ── Pass 2: predict raw labels + write ───────────────────────────────────
    print("\nPass 2 — predicting and writing raw labels ...")
    with H5BlockWriter(out_path, specs=specs) as writer:
        for (zsl, ysl, xsl), batch in iter_blocks(config):
            vol = batch[vol_key]
            eig = batch["eig"]
            vec = batch["vec"]

            vol_smooth = gaussian_filter(vol.astype(np.float32), sigma=sigma, truncate=truncate)

            X_block  = extract_features(eig, vec, vol_smooth)
            X_scaled = scaler.transform(X_block)
            X_pca    = pca.transform(X_scaled)
            labels   = km.predict(X_pca).reshape(vol.shape).astype(np.uint8)

            print(f"  block z={zsl} y={ysl} x={xsl}  labels={np.unique(labels)}", end="\r")
            writer.write_block("seg", zsl, ysl, xsl, labels)

    # ── Pass 3: morpho cleanup in-place ──────────────────────────────────────
    cleanup_segmentation(config, out_path, vol_shape, n_clusters)



    # # ── Pass 4: align + smooth + mask vec field ───────────────────────────────
    # if "vec_masked" in specs:
    #     process_vec(config, out_path, vol_shape)

    # ── XDMF ─────────────────────────────────────────────────────────────────
    xdmf_cfg     = config.get("xdmf", {})
    spacing_xyz  = tuple(xdmf_cfg.get("spacing_xyz", [1.0, 1.0, 1.0]))
    origin_xyz   = tuple(xdmf_cfg.get("origin_xyz",  [0.0, 0.0, 0.0]))

    xmf_path = write_xdmf_for_h5(
        out_path,
        keys=list(specs.keys()),
        grid_key="seg",
        spacing_xyz=spacing_xyz,  # type: ignore[arg-type]
        origin_xyz=origin_xyz,    # type: ignore[arg-type]
        make_vec_vds_if_needed=True,
    )
    print(f"XDMF  → {xmf_path}")
    print(f"\nDone → {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unsupervised Segmentation")
    parser.add_argument("-c", "--config",
                        type=str, required=True, help="Path to the config file.")

    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"Config file {config_path} does not exist.")
        exit(1)

    with open(config_path) as fh:
        config = yaml.safe_load(fh)

    main(config)

from __future__ import annotations

from typing import Dict, Iterable, Tuple, Optional, cast
from pathlib import Path

import h5py as h5

try:
    import cupy as lib  # pyright: ignore[reportMissingImports]
    xp = "cupy"
except ImportError:
    import numpy as lib
    xp = "numpy"


def load_h5_datasets(
    file_path: Path | str,
    keys: Tuple[str, ...],
    slices: Optional[Dict[str, tuple]] = None,
    dtype: lib.dtype | type = lib.float32,
    strict: bool = True,
) -> Dict[str, lib.ndarray]:
    """
    Load multiple datasets from HDF5 into a dictionary.

    Parameters
    ----------
    file_path : str or Path
    keys : list[str]
        List of dataset keys to load.
    slices : dict or None
        Optional dict {key: slice_tuple}
        Example:
            {
                "volume": (slice(0,10), slice(0,512), slice(0,512)),
                "vec": (slice(None), slice(0,10), slice(0,512), slice(0,512)),
            }
    dtype : numpy dtype
        Cast output to dtype.
    strict : bool
        If True → raise error if key missing or not dataset.
        If False → skip missing/non-dataset keys.

    Returns
    -------
    data_dict : dict
        {key: ndarray}
    """

    data = {}

    with h5.File(file_path, "r") as F:
        print("Available keys:", list(F.keys()))

        for key in keys:
            if key not in F:
                msg = f"Key '{key}' not found."
                if strict:
                    raise KeyError(msg)
                else:
                    print(msg)
                    continue

            obj = F[key]

            if not isinstance(obj, h5.Dataset):
                msg = f"Key '{key}' is not a Dataset but {type(obj)}"
                if strict:
                    raise TypeError(msg)
                else:
                    print(msg)
                    continue

            if slices and key in slices:
                arr = obj[slices[key]]
            else:
                arr = obj[:]

            data[key] = arr.astype(dtype, copy=False)

    return data



class H5BlockReader(Iterable):
    """
    Stream 3D blocks from HDF5 datasets.

    Supports:
      - (Z, Y, X) volumes
      - (C, Z, Y, X) fields (components-first)

    Iteration yields:
      (zsl, ysl, xsl), batch_dict

    batch[key] shapes:
      - for (Z,Y,X):     (bz, by, bx)
      - for (C,Z,Y,X):   (C, bz, by, bx)
    """

    def __init__(
        self,
        file_path: Path | str,
        keys: Tuple[str, ...],
        block_size: Tuple[int, int, int] = (64, 128, 128),
        dtype=lib.float32,
        strict: bool = True,
    ):
        self.file_path = str(file_path)
        self.keys = keys
        self.block_size = (int(block_size[0]), int(block_size[1]), int(block_size[2]))
        self.dtype = dtype
        self.strict = strict

        self.F: Optional[h5.File] = None
        self.ds: Dict[str, h5.Dataset] = {}

    def __enter__(self):
        self.F = h5.File(self.file_path, "r")

        for k in self.keys:
            if k not in self.F:
                if self.strict:
                    raise KeyError(f"Key '{k}' not found in {self.file_path}")
                continue

            obj = self.F[k]  # obj: Group | Dataset | Datatype (per stubs)

            if not isinstance(obj, h5.Dataset):
                if self.strict:
                    raise TypeError(f"Key '{k}' is not a Dataset (got {type(obj)})")
                continue

            self.ds[k] = cast(h5.Dataset, obj)  # ✅ tells Pylance it's a Dataset

        if not self.ds:
            raise RuntimeError("No datasets opened. Check keys/strict.")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.F is not None:
            self.F.close()
        self.F = None
        self.ds = {}

    def _zyx_shape(self) -> Tuple[int, int, int]:
        """Infer (Z,Y,X) from the first dataset."""
        d0 = next(iter(self.ds.values()))
        if d0.ndim == 3:
            Z, Y, X = d0.shape
        elif d0.ndim == 4:
            _, Z, Y, X = d0.shape
        else:
            raise ValueError(f"Unsupported dataset ndim={d0.ndim}")
        return int(Z), int(Y), int(X)

    def __iter__(self) -> Iterable[Tuple[Tuple[slice, slice, slice], Dict[str, "lib.ndarray"]]]:
        Z, Y, X = self._zyx_shape()
        bz, by, bx = self.block_size

        for z0 in range(0, Z, bz):
            z1 = min(z0 + bz, Z)
            zsl = slice(z0, z1)

            for y0 in range(0, Y, by):
                y1 = min(y0 + by, Y)
                ysl = slice(y0, y1)

                for x0 in range(0, X, bx):
                    x1 = min(x0 + bx, X)
                    xsl = slice(x0, x1)

                    batch: Dict[str, "lib.ndarray"] = {}
                    for k, dset in self.ds.items():
                        if dset.ndim == 3:
                            arr = dset[zsl, ysl, xsl]              # (bz,by,bx)
                        elif dset.ndim == 4:
                            arr = dset[:, zsl, ysl, xsl]           # (C,bz,by,bx)
                        else:
                            raise ValueError(f"Unsupported ndim={dset.ndim} for key='{k}'")

                        # important: keep backend consistent (numpy vs cupy)
                        batch[k] = lib.asarray(arr, dtype=self.dtype)

                    yield (zsl, ysl, xsl), batch


class H5BlockWriter:
    """
    Create output datasets and write true 3D blocks.

    Supports:
      - (Z, Y, X)
      - (C, Z, Y, X)   components-first

    Usage:
      writer.write_block("vol", zsl, ysl, xsl, vol_block)
      writer.write_block("vec", zsl, ysl, xsl, vec_block)
    """

    def __init__(
        self,
        out_path: Path | str,
        specs: Dict[str, dict],   # per key: shape, dtype, chunks, compression, etc.
        mode: str = "w",
    ):
        self.out_path = str(out_path)
        self.specs = specs
        self.mode = mode
        self.F: Optional[h5.File] = None
        self.ds: Dict[str, h5.Dataset] = {}

    def __enter__(self):
        self.F = h5.File(self.out_path, self.mode)
        for k, sp in self.specs.items():
            self.ds[k] = self.F.create_dataset(k, **sp)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.F is not None:
            self.F.close()
        self.F = None
        self.ds = {}

    @staticmethod
    def _to_numpy(x):
        """h5py needs NumPy; convert CuPy arrays if needed."""
        # CuPy arrays have .get(); NumPy arrays don't.
        get = getattr(x, "get", None)
        return get() if callable(get) else x

    def write_block(self, key: str, zsl: slice, ysl: slice, xsl: slice, data):
        dset = self.ds[key]
        data_np = self._to_numpy(data)

        if dset.ndim == 3:
            # data shape expected: (bz, by, bx)
            dset[zsl, ysl, xsl] = data_np
        elif dset.ndim == 4:
            # data shape expected: (C, bz, by, bx)
            dset[:, zsl, ysl, xsl] = data_np
        else:
            raise ValueError(f"Unsupported ndim={dset.ndim} for key='{key}'")
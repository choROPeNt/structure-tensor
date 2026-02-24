from __future__ import annotations

from typing import Dict, Iterable, Tuple, Optional
from pathlib import Path

import h5py as h5
import numpy as np


def load_h5_datasets(
    file_path: Path | str,
    keys: Tuple[str, ...],
    slices: Optional[Dict[str, tuple]] = None,
    dtype: np.dtype | type = np.float32,
    strict: bool = True,
) -> Dict[str, np.ndarray]:
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



class H5BlockReader:
    """
    Stream blocks from HDF5 datasets (typically chunk along Z).

    Example shapes:
      volume: (Z, Y, X)
      vec:    (3, Z, Y, X)
    """

    def __init__(
        self,
        file_path: Path | str,
        keys: Tuple[str, ...],
        z_block: int = 64,
        dtype: np.dtype | type =np.float32,
        strict: bool = True,
    ):
        self.file_path = str(file_path)
        self.keys = keys
        self.z_block = int(z_block)
        self.dtype = dtype
        self.strict = strict

        self.F = None
        self.ds = {}

    def __enter__(self):
        self.F = h5.File(self.file_path, "r")
        for k in self.keys:
            if k not in self.F:
                if self.strict:
                    raise KeyError(f"Key '{k}' not found in {self.file_path}")
                else:
                    continue
            if not isinstance(self.F[k], h5.Dataset):
                if self.strict:
                    raise TypeError(f"Key '{k}' is not a Dataset (got {type(self.F[k])})")
                else:
                    continue
            self.ds[k] = self.F[k]
        if not self.ds:
            raise RuntimeError("No datasets opened. Check keys/strict.")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.F is not None:
            self.F.close()
        self.F = None
        self.ds = {}

    def _z_size(self) -> int:
        # determine Z from first dataset (supports (Z,Y,X) or (C,Z,Y,X))
        d0 = next(iter(self.ds.values()))
        return d0.shape[0] if d0.ndim == 3 else d0.shape[1]

    def __iter__(self) -> Iterable[Tuple[slice, Dict[str, np.ndarray]]]:
        Z = self._z_size()
        for z0 in range(0, Z, self.z_block):
            z1 = min(z0 + self.z_block, Z)
            zsl = slice(z0, z1)

            batch = {}
            for k, dset in self.ds.items():
                if dset.ndim == 3:
                    arr = dset[zsl, :, :]
                elif dset.ndim == 4:
                    arr = dset[:, zsl, :, :]
                else:
                    raise ValueError(f"Unsupported ndim={dset.ndim} for key='{k}'")
                batch[k] = np.asarray(arr, dtype=self.dtype)
            yield zsl, batch


class H5BlockWriter:
    """
    Create output datasets and write blocks by Z-slice.
    """
    def __init__(
        self,
        out_path: Path | str,
        specs: Dict[str, dict],   # per key: shape, dtype, chunks, compression, etc.
        mode="w",
    ):
        self.out_path = str(out_path)
        self.specs = specs
        self.mode = mode
        self.F = None
        self.ds = {}

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

    def write_block(self, key: str, zsl: slice, data: np.ndarray):
        dset = self.ds[key]
        if dset.ndim == 3:
            dset[zsl, :, :] = data
        elif dset.ndim == 4:
            dset[:, zsl, :, :] = data
        else:
            raise ValueError(f"Unsupported ndim={dset.ndim} for key='{key}'")
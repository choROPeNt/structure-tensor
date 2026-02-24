from __future__ import annotations
from pathlib import Path
from typing import Iterable, Mapping, Optional
import h5py as h5


def write_xdmf_for_h5(
    h5_path: str | Path,
    keys: Iterable[str],
    *,
    grid_key: str = "volume",
    spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    center: str = "Node",
    make_vec_vds_if_needed: bool = True,
    vds_suffix: str = "_zyx3",
    attribute_type_override: Optional[Mapping[str, str]] = None,
) -> Path:
    """
    Create XDMF file referencing selected datasets.

    Parameters
    ----------
    spacing_xyz : (dx, dy, dz)
        Physical voxel spacing.
    origin_xyz : (ox, oy, oz)
        Physical origin.
    """

    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    if center not in {"Node", "Cell"}:
        raise ValueError("center must be 'Node' or 'Cell'")

    keys = list(keys)
    attribute_type_override = dict(attribute_type_override or {})

    dx, dy, dz = map(float, spacing_xyz)
    ox, oy, oz = map(float, origin_xyz)

    with h5.File(h5_path, "a" if make_vec_vds_if_needed else "r") as f:

        # ---- grid ----
        if grid_key not in f:
            raise KeyError(f"Missing grid_key '{grid_key}'")

        grid = f[grid_key]

        if not isinstance(grid, h5.Dataset):
            raise TypeError(f"'{grid_key}' is not an h5.Dataset")

        if grid.ndim < 3:
            raise ValueError(f"'{grid_key}' must have at least 3 dims, got {grid.shape}")

        # Take last three dims as spatial
        Z, Y, X = grid.shape



        # ---- topology dims (XDMF order) ----

        if center == "Cell":
            topoX, topoY, topoZ = X + 1, Y + 1, Z + 1
        else:  # Node
            topoX, topoY, topoZ = X, Y, Z


        attr_blocks = []

        for key in keys:
            if key not in f:
                raise KeyError(f"Missing dataset '{key}'")

            ds = f[key]
            target_key = key
            atype = attribute_type_override.get(key)
            
            if not isinstance(ds, h5.Dataset):
                raise TypeError(f"'{key}' is not an h5.Dataset")

            if ds.ndim == 3 and ds.shape == (X, Y, X):
                atype = atype or "Scalar"
                dims = f"{Z} {Y} {X}"              # X Y Z

            elif ds.ndim == 4 and ds.shape == (Z, Y, X, 3):
                atype = atype or "Vector"
                dims = f"{Z} {Y} {X} 3"            # X Y Z 3

            elif ds.ndim == 4 and ds.shape[0] == 3 and ds.shape[1:] == (Z, Y, X):
                atype = atype or "Vector"
                dims = f"{X} {Y} {Z} 3"            # X Y Z 3

                if not make_vec_vds_if_needed:
                    raise ValueError(f"{key} is (3,Z,Y,X) but VDS disabled.")

                vds_key = f"{key}{vds_suffix}"
                if vds_key in f:
                    del f[vds_key]

                # Keep actual HDF5 layout as (Z,Y,X,3) (no copy), but XDMF will read with X Y Z dims.
                layout = h5.VirtualLayout(shape=(Z, Y, X, 3), dtype=ds.dtype)
                vsrc = h5.VirtualSource(ds)
                for c in range(3):
                    layout[:, :, :, c] = vsrc[c, :, :, :]
                f.create_virtual_dataset(vds_key, layout)

                target_key = vds_key

            else:
                raise ValueError(f"Unsupported shape for '{key}': {ds.shape}")

            attr_blocks.append(
                f"""
      <Attribute Name="{key}" AttributeType="{atype}" Center="{center}">
        <DataItem Dimensions="{dims}" Format="HDF">
          {h5_path.name}:/{target_key}
        </DataItem>
      </Attribute>""".rstrip()
            )

    xmf_path = h5_path.with_suffix(".xdmf")

    xml = f"""<?xml version="1.0" ?>
<Xdmf Version="3.0">
  <Domain>
    <Grid Name="ImageData" GridType="Uniform">
      <Topology TopologyType="3DCoRectMesh" Dimensions="{topoX} {topoY} {topoZ}"/>
      <Geometry GeometryType="ORIGIN_DXDYDZ">
        <DataItem Dimensions="3" Format="XML">{ox} {oy} {oz}</DataItem>
        <DataItem Dimensions="3" Format="XML">{dx} {dy} {dz}</DataItem>
      </Geometry>
{chr(10).join(attr_blocks)}
    </Grid>
  </Domain>
</Xdmf>
"""

    xmf_path.write_text(xml, encoding="utf-8")
    return xmf_path
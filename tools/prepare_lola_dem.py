"""Extract and reproject an official LOLA DEM window for research runs."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_ingestion.provenance import sha256_file, write_provenance

DEFAULT_SOURCE = (
    "https://planetarymaps.usgs.gov/mosaic/"
    "Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif"
)
PRODUCT_ID = "Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014"
MOON_RADIUS_M = 1_737_400.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a metric south-polar LOLA subset with provenance."
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default="data/processed/shackleton_lola_240m.tif")
    parser.add_argument("--lat", type=float, default=-89.67)
    parser.add_argument("--lon", type=float, default=129.78)
    parser.add_argument("--extent-km", type=float, default=30.0)
    parser.add_argument("--resolution-m", type=float, default=240.0)
    args = parser.parse_args()

    import rasterio
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT
    from rasterio.warp import transform

    source_text = str(args.source)
    source_path = Path(source_text)
    source_is_local = source_path.is_file()
    raster_source = source_text if source_is_local else f"/vsicurl/{source_text}"
    target_crs = CRS.from_proj4(
        "+proj=stere +lat_0=-90 +lat_ts=-90 +lon_0=0 "
        f"+a={MOON_RADIUS_M} +b={MOON_RADIUS_M} +units=m +no_defs"
    )
    geographic_moon_crs = CRS.from_proj4(
        f"+proj=longlat +a={MOON_RADIUS_M} +b={MOON_RADIUS_M} +no_defs"
    )
    extent_m = args.extent_km * 1000.0
    size = int(math.ceil(extent_m / args.resolution_m))

    with rasterio.open(raster_source) as src:
        if src.crs is None:
            raise ValueError("Source LOLA product has no CRS metadata.")
        center_x, center_y = transform(
            geographic_moon_crs, target_crs, [args.lon], [args.lat]
        )
        dst_transform = from_origin(
            center_x[0] - extent_m / 2.0,
            center_y[0] + extent_m / 2.0,
            args.resolution_m,
            args.resolution_m,
        )
        with WarpedVRT(
            src,
            crs=target_crs,
            transform=dst_transform,
            width=size,
            height=size,
            resampling=Resampling.bilinear,
        ) as vrt:
            data = vrt.read(1, masked=True).astype(np.float64)
        scale = float(src.scales[0]) if src.scales else 1.0
        offset = float(src.offsets[0]) if src.offsets else 0.0
        source_crs = str(src.crs)

    data = data * scale + offset
    if data.mask.all():
        raise ValueError("Requested LOLA window contains no valid pixels.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    nodata = -9999.0
    profile = {
        "driver": "GTiff",
        "height": size,
        "width": size,
        "count": 1,
        "dtype": "float32",
        "crs": target_crs,
        "transform": dst_transform,
        "nodata": nodata,
        "compress": "deflate",
        "tiled": True,
    }
    with rasterio.open(output, "w", **profile) as dst:
        dst.write(data.filled(nodata).astype(np.float32), 1)
        dst.update_tags(
            source_product_id=PRODUCT_ID,
            source_url=source_text,
            vertical_units="metre relative to 1737.4 km sphere",
            longitude_direction="positive east",
            reference_frame="Mean Earth/polar axis (ME)",
        )

    sidecar = write_provenance(
        output,
        source_url=source_text,
        source_product_id=PRODUCT_ID,
        source_sha256=sha256_file(source_path) if source_is_local else None,
        processing={
            "operation": "bilinear warp and spatial subset",
            "center_latitude_deg": args.lat,
            "center_longitude_deg": args.lon,
            "extent_km": args.extent_km,
            "resolution_m": args.resolution_m,
            "source_crs": source_crs,
            "target_crs": target_crs.to_string(),
            "source_scale": scale,
            "source_offset": offset,
            "remote_source_hash_note": (
                None if source_is_local else "Full 8 GB source was accessed by HTTPS range; "
                "the derived product hash is verified, but the full source hash was not computed."
            ),
        },
    )
    print(f"Wrote {output}")
    print(f"Wrote {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

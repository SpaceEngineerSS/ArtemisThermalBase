from __future__ import annotations

import json

import numpy as np
import pytest

from data_ingestion.lola_loader import LOLALoader
from data_ingestion.provenance import verify_provenance, write_provenance
from validation.diviner import temperature_metrics


def test_provenance_detects_file_change(tmp_path) -> None:
    product = tmp_path / "product.bin"
    product.write_bytes(b"official-derived-data")
    write_provenance(
        product,
        source_url="https://planetarymaps.usgs.gov/example.tif",
        source_product_id="EXAMPLE",
        source_sha256=None,
        processing={"operation": "test"},
    )
    assert verify_provenance(product)["verified"] is True
    product.write_bytes(b"changed")
    with pytest.raises(ValueError, match="integrity"):
        verify_provenance(product)


def test_provenance_sidecar_is_machine_readable(tmp_path) -> None:
    product = tmp_path / "product.bin"
    product.write_bytes(b"data")
    sidecar = write_provenance(
        product,
        source_url="https://planetarymaps.usgs.gov/example.tif",
        source_product_id="EXAMPLE",
        source_sha256="abc",
        processing={},
    )
    assert json.loads(sidecar.read_text())["schema_version"] == 1


def test_diviner_missing_temperature_is_excluded() -> None:
    metrics = temperature_metrics(
        np.array([100.0, -9999.0, 120.0, np.nan]),
        np.array([102.0, 400.0, 116.0, 50.0]),
    )
    assert metrics["n"] == 2
    assert metrics["bias_k"] == pytest.approx(-1.0)
    assert metrics["rmse_k"] == pytest.approx(np.sqrt(10.0))


def test_lola_geotiff_scale_is_applied(tmp_path) -> None:
    rasterio = pytest.importorskip("rasterio")
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    path = tmp_path / "scaled_lola.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="int16",
        crs=CRS.from_proj4("+proj=eqc +a=1737400 +b=1737400 +units=m"),
        transform=from_origin(0.0, 2.0, 1.0, 1.0),
    ) as dataset:
        dataset.write(np.array([[0, 2], [4, 6]], dtype=np.int16), 1)
        dataset.scales = [0.5]

    dem = LOLALoader(center_elevation=False).load_dem(path)
    assert np.array_equal(dem.elevation, [[0.0, 1.0], [2.0, 3.0]])
    assert dem.metadata["band_scale"] == pytest.approx(0.5)

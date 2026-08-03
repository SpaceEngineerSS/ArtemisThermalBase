# Research Data Sources

Research mode accepts only traceable products. URLs and product identifiers are
machine-readable in `config/data_sources.yaml`; downloaded NAIF files are hashed
in `data/ephemeris.provenance.lock.json`.

## Topography

- Product: `Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014`
- Publisher: USGS Astrogeology / LOLA Science Team
- Instrument: LRO Lunar Orbiter Laser Altimeter (LOLA)
- Datum: elevation relative to the 1,737.4 km lunar sphere
- Reference frame: Mean Earth/polar axis, planetocentric latitude, positive-east longitude
- Storage: signed 16-bit DN with GeoTIFF scale 0.5 m/DN
- Official product page: https://astrogeology.usgs.gov/search/map/moon_lro_lola_dem_118m
- Official file: https://planetarymaps.usgs.gov/mosaic/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif

The global file is about 8 GB and is not committed. `tools/prepare_lola_dem.py`
extracts a Shackleton window, converts it to a metric lunar south-polar CRS,
applies the scale/offset, and writes a SHA-256 provenance sidecar. A remote
window does not establish the full source-file checksum; for archival
publication, download the source once and prepare from the local hashed file.

## Solar geometry

Research mode uses SPICE with pinned NAIF/JPL files: `naif0012.tls`,
`de440s.bsp`, `moon_pa_de440_200625.bpc`, and `moon_de440_250416.tf`.
The Sun vector is evaluated in `MOON_ME` with `LT+S` aberration correction.
There is no analytical fallback.

Official directories:

- https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/
- https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/
- https://naif.jpl.nasa.gov/pub/naif/generic_kernels/fk/satellites/

## Temperature validation

LRO Diviner calibrated RDR or derived polar products are the intended external
temperature reference. PDS documents missing brightness temperatures as
`-9999`; `validation.diviner.temperature_metrics` excludes those records.
Spatial/temporal/footprint/channel matching is not implemented yet, so this
release does not claim Diviner validation.

Official archive: https://pds-geosciences.wustl.edu/missions/lro/diviner.htm

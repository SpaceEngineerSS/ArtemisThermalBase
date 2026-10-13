---
title: Data and reproducibility
nav: Data
kicker: "08"
order: 8
description: Where every input comes from, how it is checked, and how to repeat a run.
---

## Sources

| input | product | from |
|---|---|---|
| leap seconds | `naif0012.tls` | NAIF generic kernels |
| planetary ephemeris | `de440s.bsp` (Park et al. 2021) | NAIF generic kernels |
| lunar orientation | `moon_pa_de440_200625.bpc` | NAIF generic kernels |
| lunar frames | `moon_de440_250416.tf` | NAIF generic kernels |
| topography | LRO LOLA global LDEM, 118 m, March 2014 | USGS Astrogeology |

All of them are listed with URLs in
[config/data_sources.yaml](../config/data_sources.yaml). Downloads are
restricted to the official NASA, JPL and USGS hosts.

```bash
python tools/fetch_research_data.py --group ephemeris
```

The four kernels are pinned by SHA-256 in that file. A download that does not
match is set aside as `.rejected` and the script stops. Interrupted downloads
resume. The result is recorded in `data/ephemeris.provenance.lock.json`.

## Topography

The LOLA product is a global equirectangular grid of about 8 GB. We never
download it whole:

```bash
python tools/prepare_lola_dem.py --lat -89.67 --lon 129.78 \
    --extent-km 40 --resolution-m 200 \
    --output data/processed/shackleton_lola_200m.tif
```

GDAL reads only the rows that cover the window over HTTPS (about 60 MB for
Shackleton) and warps them bilinearly to south polar stereographic on the
1737.4 km sphere. If the USGS address does not answer, the script switches to
the S3 bucket that USGS itself redirects to.

Next to the GeoTIFF it writes `*.provenance.json`: source URL and product,
processing parameters, and the SHA-256 of the derived file. Research mode
recomputes that hash and refuses the DEM if it changed.

Any other DEM works if it is a GeoTIFF in a metric projected CRS with a
north-up, unrotated grid. The loader applies the band's scale and offset,
fills NoData from the nearest valid cell and records what it did in the
run's metadata.

## Repeating a run

A run folder records what is needed to repeat it: the software version, all
settings that differ from the configuration file, the hashes of the DEM and
the kernels, and the SHA-256 of every output file.

1. Install the same version (`artemis --version` against
   `artemis_version` in `metadata.json`).
2. Fetch the kernels; the pinned hashes guarantee the same files.
3. Recreate the DEM with the processing parameters in its provenance sidecar,
   or use the file whose hash is recorded.
4. Run with the same configuration and command-line options.
5. Compare checksums. Results are deterministic for a given machine and Numba
   version; across CPUs the last digits of float64 sums can differ.

Before an expensive research run:

```bash
python -m validation.check_research_readiness --dem data/processed/shackleton_lola_200m.tif
```

It loads the research configuration, verifies the DEM's provenance and
relief, loads the kernels, evaluates the Sun at the start time on the DEM's
own centre and writes `output/research_readiness.json`.

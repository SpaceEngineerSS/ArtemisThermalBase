# Reproducible Research Workflow

## 1. Install

```bash
python -m pip install -e ".[dev]"
```

## 2. Fetch pinned ephemeris inputs

```bash
python tools/fetch_research_data.py --group ephemeris
```

This creates `data/ephemeris.provenance.lock.json`. Kernel binaries remain
ignored by Git.

## 3. Prepare LOLA topography

For limited bandwidth, extract a remote window:

```bash
python tools/prepare_lola_dem.py \
  --output data/processed/shackleton_lola_240m.tif \
  --extent-km 30 --resolution-m 240
```

For an archival run, first download the official global file and pass its local
path with `--source`; the sidecar then records the full source SHA-256.

## 4. Audit inputs

```bash
python -m validation.check_research_readiness \
  --dem data/processed/shackleton_lola_240m.tif
```

The audit rejects a missing sidecar, digest mismatch, non-metric DEM, suspicious
Shackleton relief, missing kernel, and implausible Sun-Moon distance.

## 5. Run

```bash
python main.py \
  --config config/research_shackleton.yaml \
  --dem data/processed/shackleton_lola_240m.tif \
  --duration 708.734 --output output/research_shackleton
```

The research configuration requires three spin-up lunar cycles, 64 solar-disk
samples, 512 view-factor rays per face, float64 state, and SPICE geometry. It is
computationally expensive. Publication results require convergence sweeps over
DEM resolution, timestep, solar samples, vertical grid, spin-up cycles, and
view-factor rays; retain every configuration and readiness report.

## 6. Quality gates

```bash
python -m pytest
python -m ruff check .
python -m mypy core_engine data_ingestion simulation thermal_solver validation main.py
```

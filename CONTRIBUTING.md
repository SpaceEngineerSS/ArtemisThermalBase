# Contributing

Thank you for taking the time. Bug reports with a reproducible command are the
most useful contribution there is; physics corrections with a reference come
right after.

## Setting up

```bash
git clone https://github.com/SpaceEngineerSS/ArtemisThermalBase.git
cd ArtemisThermalBase
python -m pip install -e ".[dev]"
python tools/fetch_research_data.py --group ephemeris   # lets the SPICE tests run
python -m pytest
```

The first test run compiles the Numba kernels; later runs take under a
minute. GUI tests run offscreen (`QT_QPA_PLATFORM=offscreen`).

## Before you open a pull request

- `python -m pytest`, `python -m ruff check .` and
  `python -m mypy -p artemis_thermal`
  pass.
- A change that alters results comes with a test that would have caught the
  old behaviour, and a line in `CHANGELOG.md` saying what changes and by how
  much.
- A new setting is added to `config/default_config.yaml` and to
  `docs/configuration.md` (a test checks they agree).
- Documentation examples in `docs/api.md` still run (a test executes them).

## Physics changes

Say where the equation comes from: a paper, a dataset, a derivation in the
pull request. If the change is a modelling choice rather than a fix, explain
what it trades. We prefer an honest limitation in `docs/limitations.md` to a
silent approximation.

## Style

Code follows the surrounding code: NumPy docstrings, type hints, names that
match the equations (`T`, `dz`, `Q_geo` are fine). Comments explain why, not
what. Documentation is plain English in the first person plural, with numbers
where we have them and without promotional words.

"""The ``artemis`` command.

    artemis run [options]      run a simulation (also the default command)
    artemis ui [RUN_FOLDER]    open the desktop app
    artemis render RUN_FOLDER  redraw the figures of a finished run
    artemis configs            list the bundled configurations
    artemis warmup             compile the Numba kernels ahead of time

``artemis --duration 6`` is the same as ``artemis run --duration 6``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from artemis_thermal import __version__

logger = logging.getLogger("artemis")

COMMANDS = ("run", "ui", "render", "configs", "warmup")


def setup_logging(level: str = "INFO") -> None:
    """Plain one-line log records on stdout."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def _parse_time(text: str) -> datetime:
    value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _positive(kind: type[float] | type[int]):
    def convert(text: str) -> float | int:
        value = kind(text)
        if value <= 0:
            raise argparse.ArgumentTypeError(f"must be positive, got {text}")
        return value

    return convert


def build_parser() -> argparse.ArgumentParser:
    """Argument parser for all subcommands."""
    parser = argparse.ArgumentParser(
        prog="artemis",
        description="Illumination and regolith temperature at the lunar poles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  artemis run --duration 24\n"
            "  artemis run --cratersize 2000 --dt 120 --output runs/small\n"
            "  artemis run --config research_shackleton --dem data/shackleton_40m.tif\n"
            "  artemis ui runs/small\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="run a simulation")
    run.add_argument("--config", default=None,
                     help="YAML file or bundled name (default: default_config)")
    run.add_argument("--dem", default=None, help="GeoTIFF DEM in a projected metric CRS")
    run.add_argument("--max-dem-size", type=_positive(int), default=None,
                     help="downsample the DEM so neither side exceeds this many pixels")
    run.add_argument("--cratersize", type=_positive(float), default=None,
                     help="synthetic crater radius [m]")
    run.add_argument("--duration", type=_positive(float), default=None,
                     help="production length in hours (default: config time_range)")
    run.add_argument("--dt", type=_positive(float), default=None, help="time step [s]")
    run.add_argument("--output-interval", type=_positive(float), default=None,
                     help="seconds between stored maps")
    run.add_argument("--start", default=None, help="UTC start, ISO-8601")
    run.add_argument("--lat", type=float, default=None, help="target latitude [deg]")
    run.add_argument("--lon", type=float, default=None, help="target longitude [deg E]")
    run.add_argument("--ephemeris", choices=["synthetic", "skyfield", "spice"], default=None)
    run.add_argument("--point-source", action=argparse.BooleanOptionalAction, default=None,
                     help="treat the Sun as a point (fast) or as a disk (penumbra)")
    run.add_argument("--view-factors", action=argparse.BooleanOptionalAction, default=None,
                     help="one bounce of terrain infrared between facets")
    run.add_argument("--disk-samples", type=_positive(int), default=None,
                     help="rays per face toward the solar disk")
    run.add_argument("--spinup-cycles", type=int, default=None,
                     help="lunar days to run before recording")
    run.add_argument("--output", default=None, help="run folder (default: config output)")
    run.add_argument("--no-figures", action="store_true", help="skip PNG figures")
    run.add_argument("--log-level", default="INFO",
                     choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    ui = sub.add_parser("ui", help="open the desktop app")
    ui.add_argument("run_folder", nargs="?", default=None)

    render = sub.add_parser("render", help="redraw the figures of a run folder")
    render.add_argument("run_folder")
    render.add_argument("--dpi", type=_positive(int), default=None)

    sub.add_parser("configs", help="list bundled configurations")
    sub.add_parser("warmup", help="compile Numba kernels into the on-disk cache")
    return parser


def _normalise_argv(argv: Sequence[str]) -> list[str]:
    args = list(argv)
    if not args or (args[0] not in COMMANDS and args[0] not in ("-h", "--help", "--version")):
        args.insert(0, "run")
    return args


def cmd_run(args: argparse.Namespace) -> int:
    """Run a simulation and write its folder and figures."""
    from core_engine.constants import load_config, log_assumptions, log_platform_info
    from simulation.runner import SimulationRunner

    config = load_config(args.config)
    if args.view_factors is not None:
        config = replace(config, view_factors=replace(config.view_factors,
                                                      enabled=args.view_factors))
    if args.disk_samples is not None:
        config = replace(config, illumination=replace(config.illumination,
                                                      solar_disk_samples=args.disk_samples))
    log_platform_info()
    log_assumptions(config)

    dem = None
    if args.dem:
        from data_ingestion.lola_loader import LOLALoader

        loader = LOLALoader(
            require_provenance=config.research.enabled and config.research.require_dem_provenance
        )
        dem = loader.load_dem(args.dem, max_size=args.max_dem_size)

    start = _parse_time(args.start or config.time_range.start)
    if args.duration is not None:
        duration_h = args.duration
    else:
        duration_h = (_parse_time(config.time_range.end) - start).total_seconds() / 3600.0
        if duration_h <= 0:
            raise SystemExit("time_range.end is not after the start time; pass --duration.")

    output = Path(args.output or config.output.directory)
    results = SimulationRunner(config, crater_radius_m=args.cratersize).run(
        start_time=start,
        duration_hours=duration_h,
        dt_s=args.dt,
        output_interval_s=args.output_interval,
        point_source_mode=args.point_source,
        ephemeris_mode=args.ephemeris,
        latitude_deg=args.lat,
        longitude_deg=args.lon,
        spinup_cycles=args.spinup_cycles,
        save_data=True,
        output_dir=output,
        external_dem=dem,
    )

    if not args.no_figures:
        from simulation.io_manager import ResultsBundle
        from visualization.figures import render_run_figures

        figures = render_run_figures(ResultsBundle.open(output), output, dpi=config.output.dpi,
                                     hero_dpi=config.output.hero_dpi)
        logger.info("Figures: %s", ", ".join(p.name for p in figures))

    T = results.final_surface_T
    logger.info(
        "Done: %d faces, %d steps in %.1f s. Surface %.1f–%.1f K (run max %.1f K). Output: %s",
        results.metadata["num_faces"], results.metadata["total_compute_steps"],
        results.metadata["wall_time_s"], T.min(), T.max(), results.max_surface_T.max(),
        output.resolve(),
    )
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    """Redraw the figures of an existing run folder."""
    from core_engine.constants import load_config
    from simulation.io_manager import ResultsBundle
    from visualization.figures import render_run_figures

    config = load_config()
    folder = Path(args.run_folder)
    dpi = args.dpi or config.output.dpi
    for path in render_run_figures(ResultsBundle.open(folder), folder, dpi=dpi,
                                   hero_dpi=max(dpi, config.output.hero_dpi)):
        logger.info("Wrote %s", path)
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    """Launch the desktop app."""
    try:
        from gui.app import main as gui_main
    except ImportError as exc:
        raise SystemExit(
            f"The desktop app needs the 'gui' extra ({exc.name} is missing):\n"
            '  pip install "artemis-thermal-base[gui]"'
        ) from exc
    return gui_main(args.run_folder)


def cmd_configs(_args: argparse.Namespace) -> int:
    """Print the bundled configuration names and paths."""
    from core_engine.constants import list_bundled_configs, resolve_config_path

    for name in list_bundled_configs():
        print(f"{name:24s} {resolve_config_path(name)}")
    return 0


def cmd_warmup(_args: argparse.Namespace) -> int:
    """Compile every hot kernel once so the first real run starts instantly."""
    from simulation.warmup import warm_up

    seconds = warm_up()
    logger.info("Kernels compiled and cached in %.1f s.", seconds)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``artemis`` command."""
    args = build_parser().parse_args(_normalise_argv(sys.argv[1:] if argv is None else argv))
    setup_logging(getattr(args, "log_level", "INFO"))
    handlers = {
        "run": cmd_run,
        "ui": cmd_ui,
        "render": cmd_render,
        "configs": cmd_configs,
        "warmup": cmd_warmup,
    }
    try:
        return handlers[args.command](args)
    except (ValueError, MemoryError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())

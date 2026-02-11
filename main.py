"""ArtemisThermalBase — CLI entry point.

Runs the lunar south pole micro-illumination and thermal simulation.

Usage
-----
    python main.py --duration 6 --cratersize 2000 --lat -89.54
    python main.py --dem data/shackleton.tif --duration 6
    python main.py --render-only           # re-render hero image from saved data

Author: Mehmet Gümüş (github.com/SpaceEngineerSS)
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging."""
    fmt = "%(name)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        stream=sys.stdout,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="artemis",
        description=(
            "ArtemisThermalBase — Lunar South Pole "
            "micro-illumination & thermal simulation"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --duration 6\n"
            "  python main.py --cratersize 2000 --duration 12 --dt 300\n"
            "  python main.py --dem data/sample_lola_dem.tif --duration 6\n"
            "  python main.py --point-source --duration 24\n"
            "  python main.py --render-only --output output\n"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to simulation config YAML (default: config/default_config.yaml)",
    )
    parser.add_argument(
        "--cratersize",
        type=float,
        default=None,
        help="Override crater radius in meters (default: from config)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=24.0,
        help="Simulation duration in hours (default: 24)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Time step in seconds (default: from config, typically 600s)",
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=-89.54,
        help="Target latitude in degrees (default: -89.54, Shackleton)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory for plots and data (default: output/)",
    )
    parser.add_argument(
        "--point-source",
        action="store_true",
        default=False,
        help="Use point-source mode (faster, no penumbra)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    parser.add_argument(
        "--output-interval",
        type=float,
        default=3600.0,
        help="Save output snapshots every N seconds (default: 3600)",
    )
    parser.add_argument(
        "--dem",
        type=str,
        default=None,
        help="Path to a real GeoTIFF DEM (e.g. LOLA). Bypasses synthetic DEM.",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        default=False,
        help="Skip simulation; render hero image from existing saved data",
    )
    parser.add_argument(
        "--hero-dpi",
        type=int,
        default=300,
        help="DPI for the hero image (default: 300)",
    )

    return parser.parse_args()


def main() -> int:
    """Main simulation entry point."""
    args = parse_args()
    setup_logging(args.log_level)

    logger = logging.getLogger("artemis")
    logger.info("=" * 60)
    logger.info("  ArtemisThermalBase — Lunar Thermal Simulation")
    logger.info("=" * 60)

    output_dir = Path(args.output)

    # --render-only mode: skip simulation, just re-render from saved data
    if args.render_only:
        logger.info("Render-only mode: loading saved data from %s/", output_dir)
        from visualization.hero_renderer import render_from_saved_data

        hero_path = render_from_saved_data(
            data_dir=output_dir,
            dpi=args.hero_dpi,
        )
        logger.info("Hero image rendered: %s", hero_path)
        return 0

    # Full simulation mode
    from core_engine.constants import load_config
    from simulation.runner import SimulationRunner
    from visualization.plotter import generate_all_plots
    from visualization.hero_renderer import render_hero_image

    # Load configuration
    config_path = Path(args.config)
    logger.info("Loading config: %s", config_path)
    config = load_config(config_path)

    # Load external DEM if provided
    external_dem = None
    if args.dem:
        from data_ingestion.lola_loader import LOLALoader
        logger.info("Loading external DEM: %s", args.dem)
        loader = LOLALoader()
        external_dem = loader.load_dem(args.dem)
        logger.info(
            "External DEM loaded: %d x %d, z=[%.1f, %.1f] m",
            external_dem.elevation.shape[1],
            external_dem.elevation.shape[0],
            external_dem.elevation.min(),
            external_dem.elevation.max(),
        )

    # Build runner
    runner = SimulationRunner(
        config=config,
        crater_radius_m=args.cratersize,
    )

    # Run simulation
    start_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    logger.info(
        "Running: start=%s, duration=%.1f hrs, dt=%s s",
        start_time.isoformat(),
        args.duration,
        args.dt or config.solver.dt_s,
    )

    results = runner.run(
        start_time=start_time,
        duration_hours=args.duration,
        dt_s=args.dt,
        output_interval_s=args.output_interval,
        point_source_mode=args.point_source if args.point_source else None,
        save_data=True,
        output_dir=output_dir,
        external_dem=external_dem,
    )

    # Generate standard debug plots
    logger.info("Generating plots → %s/", output_dir)
    saved = generate_all_plots(results, output_dir=output_dir)

    # Generate hero image
    if results.surface_temps and results.illumination_maps:
        logger.info("Rendering hero image...")
        hero_path = render_hero_image(
            face_centroids=results.face_centroids,
            thermal_grid=results.surface_temps[-1],
            illumination_grid=results.illumination_maps[-1],
            output_path=output_dir / "hero_artemis.png",
            dpi=args.hero_dpi,
        )
        saved.append(hero_path)

    # Summary
    logger.info("=" * 60)
    logger.info("  SIMULATION COMPLETE")
    logger.info("=" * 60)
    logger.info("  Duration: %.1f hours (%d steps)", args.duration, results.metadata["num_steps"])
    logger.info("  Wall time: %.1f s", results.metadata.get("wall_time_s", 0))
    if results.surface_temps:
        final_T = results.surface_temps[-1]
        logger.info(
            "  Final T: min=%.1f K, max=%.1f K, mean=%.1f K",
            final_T.min(), final_T.max(), final_T.mean(),
        )
    logger.info("  Output files (%d):", len(saved))
    for p in saved:
        logger.info("    → %s", p)
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())

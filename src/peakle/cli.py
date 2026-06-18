"""Peakle command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from peakle.config import AppSettings, load_settings
from peakle.demo.pipeline import DemoOptions, run_demo
from peakle.web.server import serve


def main(argv: list[str] | None = None) -> int:
    """Runs the Peakle CLI.

    Args:
        argv: Optional argument vector.

    Returns:
        Process exit code.
    """

    config_file = _parse_config_file(argv)
    settings = load_settings(config_file)
    parser = _build_parser(settings)
    args = parser.parse_args(argv)

    if args.command == "demo" and args.demo_command == "run":
        options = DemoOptions.from_settings(
            settings,
            output_dir=args.output,
            seed=args.seed,
            grid_width=args.grid_width,
            grid_height=args.grid_height,
            image_width=args.image_width,
            image_height=args.image_height,
            optimization_max_iterations=args.optimization_max_iterations,
        )
        result = run_demo(options)
        print(f"Artifacts: {result.output_dir.resolve()}")
        print(f"Scene: {result.scene_path.resolve()}")
        print(f"Position error: {_format_optional(result.position_error_m, 'm')}")
        print(f"Yaw error: {_format_optional(result.yaw_error_deg, 'deg')}")
        print(f"Contour MAE: {result.contour_mae_px:.2f} px")
        print(f"Visible labels: {result.visible_labels}")
        return 0

    if args.command == "web" and args.web_command == "serve":
        serve(settings, host=args.host, port=args.port)
        return 0

    parser.print_help()
    return 2


def _build_parser(settings: AppSettings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="peakle")
    parser.add_argument("--config", type=Path, help="Path to a Peakle YAML settings file")
    subparsers = parser.add_subparsers(dest="command")

    demo_parser = subparsers.add_parser("demo", help="Synthetic demo commands")
    demo_subparsers = demo_parser.add_subparsers(dest="demo_command")
    run_parser = demo_subparsers.add_parser("run", help="Run the synthetic demo")
    run_parser.add_argument("--output", type=Path, default=settings.artifact_dir)
    run_parser.add_argument("--seed", type=int, default=settings.random_seed)
    run_parser.add_argument("--grid-width", type=int, default=settings.terrain.grid_width)
    run_parser.add_argument("--grid-height", type=int, default=settings.terrain.grid_height)
    run_parser.add_argument("--image-width", type=int, default=settings.render.image_width)
    run_parser.add_argument("--image-height", type=int, default=settings.render.image_height)
    run_parser.add_argument(
        "--optimization-max-iterations",
        type=int,
        default=settings.optimization.max_iterations,
    )

    web_parser = subparsers.add_parser("web", help="Browser viewer commands")
    web_subparsers = web_parser.add_subparsers(dest="web_command")
    serve_parser = web_subparsers.add_parser("serve", help="Serve the live viewer (computes views on demand)")
    serve_parser.add_argument("--host", default=settings.web.host)
    serve_parser.add_argument("--port", type=int, default=settings.web.port)

    return parser


def _parse_config_file(argv: list[str] | None) -> Path | None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path)
    args, _unknown = pre_parser.parse_known_args(argv)
    return args.config


def _format_optional(value: float | None, unit: str) -> str:
    if value is None:
        return "-"
    return f"{value:.2f} {unit}"


if __name__ == "__main__":
    raise SystemExit(main())

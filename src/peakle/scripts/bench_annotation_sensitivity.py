"""Publish the shared-renderer Gate-1 annotation-sensitivity diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import math
import platform
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from peakle.annotation.sensitivity_study import run_annotation_sensitivity_suite as _run_suite
from peakle.config import AppSettings, load_settings, settings_payload
from peakle.rendering.terrain_view import terrain_fingerprint
from peakle.research.experiment import canonical_json_bytes, publish_flat_run, whole_worktree_provenance
from peakle.scene.state import SceneState

RUN_SCHEMA = "peakle_annotation_sensitivity_run_v2"


def main() -> None:
    args = _parser().parse_args()
    _validate_args(args)
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing artifact directory: {output}")
    settings = _effective_settings(args)
    started = datetime.now(UTC)
    state = SceneState.from_settings(settings)
    suite = _run_suite(
        state,
        camera_indices=range(len(state.true_cameras)),
        terrain_stride=args.terrain_stride,
        max_labels=args.max_labels,
    )
    results_bytes = canonical_json_bytes(suite.model_dump(mode="json", by_alias=True))
    finished = datetime.now(UTC)
    run = {
        "schema": RUN_SCHEMA,
        "status": "complete",
        "run_id": output.name,
        "created_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "wall_runtime_s": round((finished - started).total_seconds(), 3),
        "case_count": sum(len(study.cases) for study in suite.studies),
        "view_count": len(suite.studies),
        "results_sha256": hashlib.sha256(results_bytes).hexdigest(),
        "terrain": terrain_fingerprint(state.terrain),
        "effective_settings": settings_payload(settings),
        "benchmark_args": {
            "config_file": str(Path(args.config).resolve()) if args.config else None,
            "seed": args.seed,
            "terrain_extent_m": args.terrain_extent_m,
            "terrain_grid": args.terrain_grid,
            "image_width": args.image_width,
            "image_height": args.image_height,
            "max_views": args.max_views,
            "camera_ground_clearance_m": args.camera_ground_clearance_m,
            "terrain_stride": args.terrain_stride,
            "max_labels": args.max_labels,
        },
        "code": whole_worktree_provenance(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    publish_flat_run(output, run, {"results.json": results_bytes})
    print(f"Committed {len(suite.studies)} views / {run['case_count']} cases to {output}.", flush=True)


def _effective_settings(args: argparse.Namespace) -> AppSettings:
    settings = load_settings(Path(args.config) if args.config else None)
    terrain = settings.terrain.model_copy(
        update={
            "seed": args.seed,
            "width_m": args.terrain_extent_m,
            "height_m": args.terrain_extent_m,
            "grid_width": args.terrain_grid,
            "grid_height": args.terrain_grid,
        }
    )
    render = settings.render.model_copy(update={"image_width": args.image_width, "image_height": args.image_height})
    camera = settings.camera.model_copy(
        update={
            "overlook_height_m": args.camera_ground_clearance_m,
            "view_count": min(settings.camera.view_count, args.max_views),
        }
    )
    return settings.model_copy(
        update={"random_seed": args.seed, "terrain": terrain, "render": render, "camera": camera}
    )


def _validate_args(args: argparse.Namespace) -> None:
    minimums = {
        "terrain_extent_m": 2_000.0,
        "terrain_grid": 64,
        "image_width": 160,
        "image_height": 120,
        "max_views": 1,
        "terrain_stride": 1,
        "max_labels": 1,
    }
    for name, minimum in minimums.items():
        if getattr(args, name) < minimum:
            raise SystemExit(f"--{name.replace('_', '-')} must be at least {minimum}")
    if args.seed < 0:
        raise SystemExit("--seed must be non-negative")
    for name in ("terrain_extent_m", "camera_ground_clearance_m"):
        if not math.isfinite(getattr(args, name)):
            raise SystemExit(f"--{name.replace('_', '-')} must be finite")
    if args.camera_ground_clearance_m <= 0.0:
        raise SystemExit("--camera-ground-clearance-m must be positive")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--terrain-extent-m", type=float, default=16_000.0)
    parser.add_argument("--terrain-grid", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=360)
    parser.add_argument("--max-views", type=int, default=4)
    parser.add_argument("--camera-ground-clearance-m", type=float, default=2.5)
    parser.add_argument("--terrain-stride", type=int, default=2)
    parser.add_argument("--max-labels", type=int, default=8)
    return parser


if __name__ == "__main__":
    main()

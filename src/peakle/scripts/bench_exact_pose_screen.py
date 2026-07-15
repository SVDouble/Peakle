"""Run the registered independent exact-pose representation screen."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from peakle.domain.camera import CameraIntrinsics
from peakle.localize.correspondence import SiftMatcher, WorkerMatcher, load_model_manifest
from peakle.localize.paths import BASE
from peakle.research.exact_pose_screen import ExactPoseScreenResult, run_exact_pose_screen
from peakle.research.experiment import canonical_json_bytes, publish_flat_run, whole_worktree_provenance
from peakle.research.synthetic_scenes import rugged_scenes
from peakle.research.webgl_query import render_terrain_webgl_query

RUN_SCHEMA = "peakle_exact_pose_representation_screen_run_v1"
SUPPORT_PATHS = "domain/peaks.py rendering/skyline.py localize/paths.py".split()


def main() -> None:
    args = _parser().parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing artifact directory: {output}")
    if args.timeout_s <= 0.0:
        raise SystemExit("--timeout-s must be positive")
    implementation = tuple(
        BASE / path
        for path in (
            "docs/research-and-development.md",
            "src/peakle/io/artifacts.py",
            "src/peakle/domain/angles.py",
            "src/peakle/domain/camera.py",
            "src/peakle/domain/coordinates.py",
            "src/peakle/domain/terrain.py",
            *(f"src/peakle/{path}" for path in SUPPORT_PATHS),
            "src/peakle/research/exact_pose_screen.py",
            "src/peakle/research/exact_pose_correspondence.py",
            "src/peakle/research/experiment.py",
            "src/peakle/research/synthetic_scenes.py",
            "src/peakle/research/webgl_contract.py",
            "src/peakle/research/webgl_query.py",
            "src/peakle/research/webgl_query.html",
            "src/peakle/rendering/terrain_view.py",
            "src/peakle/rendering/rasterizer.py",
            "src/peakle/rendering/pinhole.py",
            "src/peakle/localize/correspondence.py",
            "src/peakle/localize/pnp.py",
            "src/peakle/scripts/bench_exact_pose_screen.py",
            "src/peakle/scripts/roma_match_worker.py",
            "src/peakle/config.py",
            "src/peakle/default_settings.yaml",
            "src/peakle/scene/state.py",
            "src/peakle/terrain/generator.py",
            "src/peakle/terrain/peak_detection.py",
            "pyproject.toml",
            "uv.lock",
        )
    )
    before = whole_worktree_provenance(implementation)
    if before["dirty"] is not False or before["git_sha"] is None:
        raise SystemExit("registered screen requires a clean Git worktree with a resolved HEAD")
    started_at, started = datetime.now(UTC), time.perf_counter()
    scenes = rugged_scenes(
        (31, 47),
        views_per_scene=2,
        terrain_width_m=14_000.0,
        terrain_height_m=10_000.0,
        terrain_grid_width=97,
        terrain_grid_height=73,
        eye_height_m=2.5,
    )
    intrinsics = CameraIntrinsics.from_horizontal_fov(320, 180, 55.0)
    worker = (sys.executable, str(BASE / "src/peakle/scripts/roma_match_worker.py"))

    def learned(matcher_id: str, manifest_path: Path):
        return lambda: WorkerMatcher(
            command=worker,
            matcher_id=matcher_id,
            model_manifest=load_model_manifest(manifest_path.expanduser().resolve()),
            timeout_s=args.timeout_s,
        )

    results, blobs = run_exact_pose_screen(
        scenes,
        intrinsics,
        {
            "sift": SiftMatcher,
            "roma_outdoor": learned("roma_outdoor", args.roma_manifest),
            "minima_roma": learned("minima_roma", args.minima_manifest),
        },
        lambda terrain, camera, pose: render_terrain_webgl_query(
            terrain, camera, pose, chromium_path=args.chromium_path, timeout_s=args.timeout_s
        ),
    )
    results = ExactPoseScreenResult.model_validate({**results, "run_id": output.name}).model_dump(
        mode="json", by_alias=True
    )
    results_bytes = canonical_json_bytes(results)
    summary_bytes = _summary(results).encode()
    after = whole_worktree_provenance(implementation)
    if after["dirty"] is not False or after["git_sha"] != before["git_sha"]:
        raise RuntimeError("source worktree changed during the registered screen")
    run = {
        "schema": RUN_SCHEMA,
        "status": "complete",
        "run_id": output.name,
        "created_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "wall_runtime_s": round(time.perf_counter() - started, 6),
        "results_sha256": hashlib.sha256(results_bytes).hexdigest(),
        "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "blob_count": len(blobs),
        "blob_bytes": sum(map(len, blobs.values())),
        "code": after,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "scikit_image": importlib.metadata.version("scikit-image"),
            "pydantic": importlib.metadata.version("pydantic"),
        },
    }
    publish_flat_run(output, run, {**blobs, "results.json": results_bytes, "summary.md": summary_bytes})
    print(f"Published {output} with {len(results['case_evaluations'])} graded cases.", flush=True)


def _summary(results: dict) -> str:
    aggregates = results["aggregates"]
    return (
        "# Exact-pose representation screen\n\n"
        f"- Harness passed: `{aggregates['harness_passed']}`\n"
        f"- PnP survivors: `{', '.join(aggregates['surviving_pairs']) or 'none'}`\n"
        f"- Advanced pairs: `{', '.join(aggregates['advanced_pairs']) or 'none'}`\n"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chromium-path", type=Path)
    parser.add_argument("--roma-manifest", type=Path, default=BASE / "local/models/roma/manifest.json")
    parser.add_argument("--minima-manifest", type=Path, default=BASE / "local/models/minima/manifest.json")
    parser.add_argument("--timeout-s", type=float, default=300.0)
    return parser


if __name__ == "__main__":
    main()

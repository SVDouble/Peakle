"""Rerank a completed pose-atlas artifact with analysis-only PFM geometry.

The source PFM is itself reference-pose-rendered evidence.  This script measures
the geometry ceiling beyond skyline-only ranking; it does not produce a
production-eligible localization result.  It writes and hashes ``rerank.json``
before numeric pose truth is passed to the post-hoc evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from peakle.domain.camera import CameraExtrinsics
from peakle.domain.coordinates import GeoPoint, LocalPoint
from peakle.domain.pose import PosePrior
from peakle.localize.bench import find_sample_dirs
from peakle.localize.geopose import read_pfm
from peakle.localize.paths import BASE, COP_TILES_DIR
from peakle.localize.pfm_geometry_rerank import (
    PFM_RERANK_ARCHIVE_SCHEMA,
    PfmGeometryRerankArchive,
    PfmGeometryRerankConfig,
    build_pfm_geometry_rerank,
    evaluate_pfm_geometry_rerank,
)
from peakle.localize.strategy_bench import (
    PriorScenario,
    default_terrain_cache_inventory,
    file_sha256,
    provision_estimator_terrain,
)
from peakle.scripts.bench_pose_atlas import ATLAS_STUDY_SCHEMA
from peakle.terrain.copernicus import load_copernicus_terrain

PFM_GEOMETRY_STUDY_SCHEMA = "peakle_pose_atlas_pfm_geometry_study_v1"
ESTIMATOR_FILE = "rerank.json"


def main() -> None:
    args = _parser().parse_args()
    atlas_path = Path(args.atlas)
    output_dir = Path(args.output)
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing artifact directory: {output_dir}")
    source_bytes = atlas_path.read_bytes()
    source_results_sha256 = hashlib.sha256(source_bytes).hexdigest()
    source_results = _json_object(source_bytes, atlas_path)
    source_run_path = atlas_path.with_name("run.json")
    source_run_bytes = source_run_path.read_bytes()
    source_run_sha256 = hashlib.sha256(source_run_bytes).hexdigest()
    source_run = _json_object(source_run_bytes, source_run_path)
    _validate_source_artifact(source_results, source_run, source_bytes)
    selected = _selected_records(source_results, args.samples, args.candidate_track)
    estimator_specs = _estimator_specs(selected, args.candidate_track)
    sample_dirs = {path.name: path for path in find_sample_dirs()}
    missing = [record["name"] for record in selected if record["name"] not in sample_dirs]
    if missing:
        raise SystemExit(f"GeoPose inputs are unavailable for: {', '.join(missing)}")
    _validate_input_depth_hashes(selected, source_run, sample_dirs)
    expected_cache_sha = _expected_cache_sha(source_run)
    cache_before = _compact_cache_inventory(default_terrain_cache_inventory())
    if cache_before["aggregate_sha256"] != expected_cache_sha:
        raise SystemExit("terrain cache differs from the source atlas run; rerun the atlas before geometry reranking")

    config = PfmGeometryRerankConfig(subsample=args.subsample)
    started_at = datetime.now(UTC)
    implementation = _implementation_record()
    archives: dict[str, PfmGeometryRerankArchive] = {}
    estimator_samples: list[dict[str, Any]] = []
    print(
        f"PFM geometry rerank: {len(selected)} sample(s), track {args.candidate_track}, subsample {config.subsample}",
        flush=True,
    )
    for index, estimator_spec in enumerate(estimator_specs, start=1):
        sample_started = time.perf_counter()
        name = str(estimator_spec["name"])
        atlas_archive = estimator_spec["estimator_archive"]
        estimator_terrain = _rehydrate_estimator_terrain(estimator_spec, source_results["config"])
        source_depth_path = sample_dirs[name] / "cyl" / "distance_crop.pfm"
        rerank = build_pfm_geometry_rerank(
            estimator_terrain.terrain,
            estimator_terrain.high_resolution_patch,
            read_pfm(source_depth_path),
            atlas_archive,
            config=config,
        )
        archives[name] = rerank
        estimator_samples.append(
            {
                "name": name,
                "candidate_track": args.candidate_track,
                "source_depth_file_sha256": file_sha256(source_depth_path),
                "terrain_inputs": estimator_terrain.provenance,
                "rerank_archive": rerank.to_record(),
                "runtime_s": round(time.perf_counter() - sample_started, 4),
            }
        )
        print(
            f"[{index}/{len(selected)}] {name}: selected original atlas rank "
            f"{rerank.selected.original_estimator_rank}; {estimator_samples[-1]['runtime_s']:.1f}s",
            flush=True,
        )

    _verify_stability(
        atlas_path=atlas_path,
        source_results_sha256=source_results_sha256,
        source_run_path=source_run_path,
        source_run_sha256=source_run_sha256,
        selected=selected,
        source_run=source_run,
        sample_dirs=sample_dirs,
        cache_before=cache_before,
        implementation_before=implementation,
    )
    estimator_study = {
        "schema": PFM_GEOMETRY_STUDY_SCHEMA,
        "analysis_only": True,
        "production_eligible": False,
        "numeric_evaluation_reference_used": False,
        "source_atlas": {
            "path": str(atlas_path),
            "results_sha256": source_results_sha256,
            "run_sha256": source_run_sha256,
            "schema": source_results["schema"],
            "candidate_track": args.candidate_track,
        },
        "config": config.to_record(),
        "implementation": implementation,
        "terrain_cache": cache_before,
        "samples": estimator_samples,
    }
    estimator_bytes = _json_bytes(estimator_study)
    estimator_sha = hashlib.sha256(estimator_bytes).hexdigest()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    committed = False
    try:
        _write_once(staging_dir / ESTIMATOR_FILE, estimator_bytes)
        persisted = (staging_dir / ESTIMATOR_FILE).read_bytes()
        if hashlib.sha256(persisted).hexdigest() != estimator_sha:
            raise RuntimeError("persisted estimator archive failed its post-write digest check")
        _fsync_directory(staging_dir)

        # Numeric reference pose was already present in the source artifact, but the whitelist above
        # prevented ranker access.  It is first passed to the evaluator below this durable freeze boundary.
        evaluation_samples = _evaluate_frozen_archives(selected, archives, args.candidate_track)
        results = {
            "schema": PFM_GEOMETRY_STUDY_SCHEMA,
            "analysis_only": True,
            "production_eligible": False,
            "source_atlas": {
                "results_sha256": source_results_sha256,
                "run_sha256": source_run_sha256,
            },
            "estimator_archive": {
                "file": ESTIMATOR_FILE,
                "sha256": estimator_sha,
                "frozen_before_numeric_evaluation": True,
            },
            "samples": evaluation_samples,
            "aggregates": _aggregates(evaluation_samples),
        }
        results_bytes = _json_bytes(results)
        summary_bytes = _summary_markdown(results).encode()
        _verify_stability(
            atlas_path=atlas_path,
            source_results_sha256=source_results_sha256,
            source_run_path=source_run_path,
            source_run_sha256=source_run_sha256,
            selected=selected,
            source_run=source_run,
            sample_dirs=sample_dirs,
            cache_before=cache_before,
            implementation_before=implementation,
        )
        finished_at = datetime.now(UTC)
        run = {
            "schema": PFM_GEOMETRY_STUDY_SCHEMA,
            "run_id": output_dir.name,
            "status": "complete",
            "created_at": started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "wall_runtime_s": round((finished_at - started_at).total_seconds(), 3),
            "source_atlas_results_sha256": source_results_sha256,
            "source_atlas_run_sha256": source_run_sha256,
            "estimator_archive_sha256": estimator_sha,
            "results_sha256": hashlib.sha256(results_bytes).hexdigest(),
            "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
            "truth_separation": {
                "source_depth_pfm_used_by_ranker": True,
                "source_depth_generated_at_reference_pose": True,
                "numeric_reference_pose_used_by_ranker": False,
                "estimator_archive_frozen_before_numeric_evaluation": True,
            },
            "provenance_stability": {
                "captured_before_first_sample": True,
                "rechecked_before_estimator_freeze": True,
                "rechecked_before_artifact_commit": True,
                "changed_sections": [],
            },
            "environment": {"python": platform.python_version(), "platform": platform.platform()},
            "limitations": [
                "the source PFM is reference-pose-rendered oracle evidence",
                "PFM distance is treated as camera-ray range; DEM horizontal range is converted by ray elevation",
                "one median log-depth scale nuisance is fit independently per candidate",
                "the fixed fusion weights are heuristic and were frozen before post-hoc numeric pose evaluation",
                "the study reranks only three yaw-separated modes retained per source-atlas position",
            ],
        }
        _write_once(staging_dir / "results.json", results_bytes)
        _write_once(staging_dir / "summary.md", summary_bytes)
        _write_once(staging_dir / "run.json", _json_bytes(run))
        _fsync_directory(staging_dir)
        if output_dir.exists():
            raise RuntimeError(f"output directory appeared during execution: {output_dir}")
        os.rename(staging_dir, output_dir)
        _fsync_directory(output_dir.parent)
        committed = True
    finally:
        if not committed:
            shutil.rmtree(staging_dir, ignore_errors=True)
    print(f"Committed {output_dir} (estimator archive sha256 {estimator_sha[:12]}…).", flush=True)


def _rehydrate_estimator_terrain(sample: dict[str, Any], study_config: dict[str, Any]):
    """Reconstruct only the terrain stack declared by the archived prior."""

    terrain_config = study_config.get("terrain")
    if not isinstance(terrain_config, dict):
        raise ValueError("source atlas terrain configuration is missing")
    extent = float(terrain_config["extent_m"])
    grid = int(terrain_config["grid"])
    origin = GeoPoint.model_validate(sample["coordinate_frame_origin"])
    terrain = load_copernicus_terrain(
        origin.latitude_deg,
        origin.longitude_deg,
        extent_m=extent,
        grid=grid,
        tile_dir=COP_TILES_DIR,
    )
    prior_record = sample.get("prior")
    if not isinstance(prior_record, dict):
        raise ValueError("source atlas prior record is missing")
    prior = PosePrior(
        position=LocalPoint.model_validate(prior_record["position"]),
        yaw_deg=float(prior_record["yaw_deg"]),
        pitch_deg=float(prior_record["pitch_deg"]),
        horizontal_sigma_m=float(prior_record["horizontal_sigma_m"]),
        vertical_sigma_m=float(prior_record["vertical_sigma_m"]),
        yaw_sigma_deg=float(prior_record["yaw_sigma_deg"]),
        pitch_sigma_deg=float(prior_record["pitch_sigma_deg"]),
    )
    terrain_inputs = sample.get("terrain_inputs")
    native_record = terrain_inputs.get("native_patch", {}) if isinstance(terrain_inputs, dict) else {}
    scenario = PriorScenario(
        name=cast(Any, str(prior_record["regime"])),
        prior=prior,
        use_position_prior=True,
        use_orientation_prior=True,
        perturbation={"replicate": int(prior_record["terrain_cache_replicate"])},
        contains_exact_reference=bool(native_record.get("prior_contains_exact_reference", False)),
        constructed_from_reference=bool(prior_record.get("constructed_from_reference_for_controlled_perturbation")),
    )
    return provision_estimator_terrain(terrain, scenario)


def _estimator_specs(samples: list[dict[str, Any]], candidate_track: str) -> list[dict[str, Any]]:
    """Whitelist only inputs allowed above the numeric-evaluation boundary."""

    specs: list[dict[str, Any]] = []
    prior_fields = (
        "position",
        "yaw_deg",
        "pitch_deg",
        "horizontal_sigma_m",
        "vertical_sigma_m",
        "yaw_sigma_deg",
        "pitch_sigma_deg",
        "regime",
        "constructed_from_reference_for_controlled_perturbation",
    )
    for sample in samples:
        prior = sample["prior"]
        perturbation = prior.get("perturbation")
        replicate = perturbation.get("replicate") if isinstance(perturbation, dict) else None
        if isinstance(replicate, bool) or not isinstance(replicate, int) or replicate < 0:
            raise ValueError("source atlas prior has no valid terrain-cache replicate")
        whitelisted_prior = {field: prior[field] for field in prior_fields}
        # Only the replicate indexes a deterministic terrain-provisioning cache. The source
        # perturbation also stores signed offsets that reconstruct numeric reference truth and
        # therefore must never cross the estimator boundary.
        whitelisted_prior["terrain_cache_replicate"] = replicate
        spec = {
            "name": sample["name"],
            "coordinate_frame_origin": sample["coordinate_frame_origin"],
            "prior": whitelisted_prior,
            "terrain_inputs": sample["terrain_inputs"],
            "estimator_archive": sample["tracks"][candidate_track]["estimator_archive"],
        }
        encoded = json.dumps(spec, sort_keys=True)
        if '"reference"' in encoded or '"errors"' in encoded or '"evaluation"' in encoded:
            raise ValueError("numeric evaluation data crossed the estimator input whitelist")
        specs.append(spec)
    return specs


def _evaluate_frozen_archives(
    sample_records: list[dict[str, Any]],
    archives: dict[str, PfmGeometryRerankArchive],
    candidate_track: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sample in sample_records:
        name = str(sample["name"])
        archive = archives[name]
        truth = CameraExtrinsics.model_validate(sample["reference"])
        requested = (1, 5, 10, 25, 50, 100, 250, 500, 1000, len(archive.candidates))
        evaluation = evaluate_pfm_geometry_rerank(archive, truth, top_ks=requested)
        source_evaluation = sample["tracks"][candidate_track].get("evaluation")
        atlas_winner = source_evaluation.get("winner_errors") if isinstance(source_evaluation, dict) else None
        records.append(
            {
                "name": name,
                "candidate_track": candidate_track,
                "rerank_archive_schema": PFM_RERANK_ARCHIVE_SCHEMA,
                "rerank_archive_sha256": archive.archive_sha256,
                "atlas_skyline_winner": atlas_winner,
                "evaluation": evaluation.to_record(),
            }
        )
    return records


def _aggregates(samples: list[dict[str, Any]]) -> dict[str, Any]:
    winner = [sample["evaluation"]["winner_errors"] for sample in samples]
    oracle = [sample["evaluation"]["candidate_pool_gt_oracle"] for sample in samples]
    component_names = sorted(
        {component for sample in samples for component in sample["evaluation"].get("component_winner_errors", {})}
    )
    component_winners = {
        component: _evaluated_candidate_aggregate(
            [sample["evaluation"]["component_winner_errors"][component] for sample in samples]
        )
        for component in component_names
    }
    return {
        "samples": len(samples),
        "winner_successes": sum(item["reaches_target"] for item in winner),
        "candidate_pool_oracle_successes": sum(item["reaches_target"] for item in oracle),
        "median_winner_horizontal_m": _median_error(winner, "horizontal_position_m"),
        "median_winner_yaw_deg": _median_error(winner, "yaw_deg"),
        "median_candidate_pool_oracle_horizontal_m": _median_error(oracle, "horizontal_position_m"),
        "median_candidate_pool_oracle_yaw_deg": _median_error(oracle, "yaw_deg"),
        "component_winners": component_winners,
    }


def _evaluated_candidate_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "samples": len(records),
        "successes": sum(record["reaches_target"] is True for record in records),
        "median_horizontal_m": _median_error(records, "horizontal_position_m"),
        "median_yaw_deg": _median_error(records, "yaw_deg"),
    }


def _median_error(records: list[dict[str, Any]], field: str) -> float | None:
    values = [float(record["errors"][field]) for record in records if record.get("errors", {}).get(field) is not None]
    return round(statistics.median(values), 5) if values else None


def _summary_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# PFM geometry rerank ceiling",
        "",
        "Analysis-only: the PFM was rendered at the reference pose. Numeric pose truth was evaluated "
        "only after `rerank.json` was frozen.",
        "",
        "| sample | skyline-only winner | PFM geometry winner | candidate-pool oracle | original atlas rank |",
        "|---|---:|---:|---:|---:|",
    ]
    for sample in results["samples"]:
        atlas = sample.get("atlas_skyline_winner")
        atlas_text = _format_evaluated(atlas) if isinstance(atlas, dict) else "-"
        evaluation = sample["evaluation"]
        winner = evaluation["winner_errors"]
        oracle = evaluation["candidate_pool_gt_oracle"]
        lines.append(
            f"| {sample['name']} | {atlas_text} | {_format_evaluated(winner)} | "
            f"{_format_evaluated(oracle)} | {winner['original_estimator_rank']} |"
        )
    lines.extend(
        [
            "",
            "## Fixed component-winner ablations",
            "",
            "Each row evaluates the frozen winner of one predeclared score component; numeric truth did not "
            "choose these candidates.",
            "",
            "| component | target successes | median position | median yaw |",
            "|---|---:|---:|---:|",
        ]
    )
    for component, aggregate in results["aggregates"]["component_winners"].items():
        lines.append(
            f"| {component} | {aggregate['successes']}/{aggregate['samples']} | "
            f"{aggregate['median_horizontal_m']:.1f} m | {aggregate['median_yaw_deg']:.1f}° |"
        )
    return "\n".join(lines) + "\n"


def _format_evaluated(record: dict[str, Any]) -> str:
    errors = record["errors"]
    return f"{errors['horizontal_position_m']:.1f} m / {errors['yaw_deg']:.1f}°"


def _selected_records(results: dict[str, Any], samples_value: str, candidate_track: str) -> list[dict[str, Any]]:
    raw_samples = results.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise SystemExit("source atlas contains no samples")
    by_name = {str(sample.get("name")): sample for sample in raw_samples if isinstance(sample, dict)}
    requested = [item.strip() for item in samples_value.split(",") if item.strip()]
    if len(requested) != len(set(requested)):
        raise SystemExit("duplicate sample names are not allowed")
    if not requested:
        requested = list(by_name)
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise SystemExit(f"source atlas does not contain: {', '.join(missing)}")
    selected = [by_name[name] for name in requested]
    unavailable = []
    for sample in selected:
        tracks = sample.get("tracks")
        track = tracks.get(candidate_track) if isinstance(tracks, dict) else None
        if (
            not isinstance(track, dict)
            or track.get("status") != "ok"
            or not isinstance(track.get("estimator_archive"), dict)
        ):
            unavailable.append(str(sample["name"]))
    if unavailable:
        raise SystemExit(f"candidate track {candidate_track!r} is unavailable for: {', '.join(unavailable)}")
    return selected


def _validate_source_artifact(results: dict[str, Any], run: dict[str, Any], results_bytes: bytes) -> None:
    if results.get("schema") != ATLAS_STUDY_SCHEMA or run.get("schema") != ATLAS_STUDY_SCHEMA:
        raise SystemExit(f"source must be a completed {ATLAS_STUDY_SCHEMA} artifact")
    if run.get("status") != "complete":
        raise SystemExit("source atlas run is not complete")
    actual = hashlib.sha256(results_bytes).hexdigest()
    if run.get("results_sha256") != actual:
        raise SystemExit("source atlas results.json does not match run.json")


def _validate_input_depth_hashes(
    samples: list[dict[str, Any]],
    source_run: dict[str, Any],
    sample_dirs: dict[str, Path],
) -> None:
    inputs = source_run.get("inputs")
    files = inputs.get("files") if isinstance(inputs, dict) else None
    if not isinstance(files, list):
        raise SystemExit("source atlas input fingerprint is missing")
    expected = {str(record.get("path")): str(record.get("sha256")) for record in files if isinstance(record, dict)}
    for sample in samples:
        name = str(sample["name"])
        relative = f"{name}/cyl/distance_crop.pfm"
        actual = file_sha256(sample_dirs[name] / "cyl" / "distance_crop.pfm")
        if expected.get(relative) != actual:
            raise SystemExit(f"source PFM changed since the atlas run: {name}")


def _expected_cache_sha(source_run: dict[str, Any]) -> str:
    terrain_cache = source_run.get("terrain_cache")
    value = terrain_cache.get("aggregate_sha256") if isinstance(terrain_cache, dict) else None
    if not isinstance(value, str):
        raise SystemExit("source atlas terrain-cache fingerprint is missing")
    return value


def _compact_cache_inventory(inventory: dict[str, Any]) -> dict[str, Any]:
    caches: dict[str, Any] = {}
    for name, record in sorted(inventory.items()):
        files = record.get("files", []) if isinstance(record, dict) else []
        caches[name] = {
            "file_count": len(files),
            "total_size_bytes": sum(int(item.get("size", 0)) for item in files),
            "inventory_sha256": hashlib.sha256(_json_bytes(files)).hexdigest(),
        }
    return {
        "scope": "available_cache_inventory_not_exact_consumption",
        "caches": caches,
        "aggregate_sha256": hashlib.sha256(_json_bytes(inventory)).hexdigest(),
    }


def _implementation_record() -> dict[str, Any]:
    paths = [
        Path(__file__),
        BASE / "src/peakle/scripts/bench_pose_atlas.py",
        BASE / "src/peakle/localize/bench.py",
        BASE / "src/peakle/localize/pfm_geometry_rerank.py",
        BASE / "src/peakle/localize/geopose.py",
        BASE / "src/peakle/localize/gtrefine.py",
        BASE / "src/peakle/localize/typed_outlines.py",
        BASE / "src/peakle/localize/raycast.py",
        BASE / "src/peakle/localize/strategy_bench.py",
        BASE / "src/peakle/localize/swissdem.py",
        BASE / "src/peakle/localize/copdem.py",
        BASE / "src/peakle/localize/paths.py",
        BASE / "src/peakle/terrain/copernicus.py",
        BASE / "src/peakle/domain/camera.py",
        BASE / "src/peakle/domain/coordinates.py",
        BASE / "src/peakle/domain/pose.py",
        BASE / "src/peakle/domain/projection.py",
        BASE / "src/peakle/domain/terrain.py",
    ]
    files = [{"path": str(path.relative_to(BASE)), "sha256": file_sha256(path)} for path in paths]
    relative = [record["path"] for record in files]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=BASE, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--", *relative], cwd=BASE, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    diff = subprocess.run(
        ["git", "diff", "--binary", "--", *relative], cwd=BASE, check=True, capture_output=True
    ).stdout
    return {
        "git_revision": revision,
        "aggregate_sha256": hashlib.sha256(_json_bytes(files)).hexdigest(),
        "files": files,
        "source_worktree_status": status,
        "tracked_source_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _verify_stability(
    *,
    atlas_path: Path,
    source_results_sha256: str,
    source_run_path: Path,
    source_run_sha256: str,
    selected: list[dict[str, Any]],
    source_run: dict[str, Any],
    sample_dirs: dict[str, Path],
    cache_before: dict[str, Any],
    implementation_before: dict[str, Any],
) -> None:
    """Refuse to freeze or publish if any estimator input changed during the run."""

    changed: list[str] = []
    if file_sha256(atlas_path) != source_results_sha256:
        changed.append("source_atlas_results")
    if file_sha256(source_run_path) != source_run_sha256:
        changed.append("source_atlas_run_metadata")
    try:
        _validate_input_depth_hashes(selected, source_run, sample_dirs)
    except SystemExit:
        changed.append("source_pfm_inputs")
    if _compact_cache_inventory(default_terrain_cache_inventory()) != cache_before:
        changed.append("terrain_cache")
    if _implementation_record() != implementation_before:
        changed.append("selected_implementation_paths")
    if changed:
        raise RuntimeError(f"rerank inputs changed during execution: {', '.join(changed)}")


def _json_object(content: bytes, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError as error:
        raise SystemExit(f"invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"expected a JSON object in {path}")
    return value


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()


def _write_once(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", required=True, help="completed pose-atlas results.json")
    parser.add_argument("--candidate-track", choices=("pfm_oracle", "photo_auto"), default="pfm_oracle")
    parser.add_argument("--samples", default="", help="comma-separated subset; default is every source sample")
    parser.add_argument("--subsample", type=_positive_int, default=4)
    parser.add_argument("--output", required=True, help="new write-once output directory")
    return parser


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


if __name__ == "__main__":
    main()

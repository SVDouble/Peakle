"""Run a compute-heavy local skyline horizontal-position/yaw ceiling study.

This is an analysis tool, not a production solver or leaderboard entry.  It
constructs a reproducible synthetic prior from the GeoPose reference, then
freezes and hashes the complete estimator score lattice before numeric reference
evaluation.  The PFM track is a reference-pose-rendered geometry ceiling; the
automatic-photo track measures current end-to-end skyline evidence under the
same search budget.  Neither oracle replaces the blind estimator winner.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from peakle.domain.camera import CameraExtrinsics
from peakle.localize.bench import find_sample_dirs
from peakle.localize.extract import best_skyline_candidate, extract_candidates
from peakle.localize.geopose import load_sample, resampled_oracle_skyline
from peakle.localize.paths import BASE, STD_WIDTH
from peakle.localize.skyline_atlas import (
    SkylineAtlasConfig,
    build_skyline_atlas,
    evaluate_skyline_atlas,
)
from peakle.localize.strategy_bench import (
    MATRIX_EXTRACTORS,
    EvidenceTrack,
    MatrixConfig,
    assess_pre_solve_quality,
    build_prior_scenario,
    default_terrain_cache_inventory,
    file_sha256,
    input_fingerprint,
    load_benchmark_terrain,
    provision_estimator_terrain,
)

ATLAS_STUDY_SCHEMA = "peakle_pose_atlas_study_v2"
DEFAULT_SAMPLES = (
    "eth_ch1_IMG_4948_01024",
    "eth_ch1_IMG_5143_01024",
    "eth_ch1_IMG_5145_01024",
)
TRACKS = ("pfm_oracle", "photo_auto")


def main() -> None:
    args = _parser().parse_args()
    sample_dirs = _selected_samples(args.samples)
    tracks = _csv_tracks(args.tracks)
    atlas_config = SkylineAtlasConfig(
        radius_m=args.radius_m,
        spacing_m=args.spacing_m,
        yaw_step_deg=args.yaw_step_deg,
        yaw_modes_per_position=args.yaw_modes_per_position,
        yaw_mode_separation_deg=args.yaw_mode_separation_deg,
        max_observed_columns=args.max_observed_columns,
        residual_cap_px=args.residual_cap_px,
        high_outlier_trim_fraction=args.high_outlier_trim_fraction,
        high_outlier_weight=args.high_outlier_weight,
        max_abs_roll_nuisance_deg=args.max_abs_roll_nuisance_deg,
        eye_height_m=args.eye_height_m,
        ray_step_m=args.ray_step_m,
    )
    matrix_config = MatrixConfig(
        algorithms=("keep-prior",),
        evidence_tracks=("pfm_oracle",),
        prior_regimes=("perturbed_metadata",),
        perturbation_bucket=args.perturbation,
        root_seed=args.seed,
        extent_m=args.extent_km * 1000.0,
        terrain_grid=args.grid,
        extractor=args.extractor,
        map_center_offset_fraction=args.map_offset_fraction,
    )
    matrix_config.validate()
    output_dir = Path(args.output)
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing artifact directory: {output_dir}")

    provenance_started = _provenance_snapshot(sample_dirs)
    started_at = datetime.now(UTC)
    samples: list[dict[str, Any]] = []
    print(
        f"Pose atlas: {len(sample_dirs)} sample(s), {len(tracks)} track(s), "
        f"{_position_count(atlas_config)} positions, 360° scored, "
        f"{atlas_config.yaw_modes_per_position} yaw modes/position shortlisted per track",
        flush=True,
    )
    for index, sample_dir in enumerate(sample_dirs, start=1):
        sample_started = time.perf_counter()
        record = _run_sample(
            sample_dir,
            tracks=tracks,
            matrix_config=matrix_config,
            atlas_config=atlas_config,
            replicate=args.replicate,
        )
        samples.append(record)
        print(
            f"[{index}/{len(sample_dirs)}] {sample_dir.name}: {time.perf_counter() - sample_started:.1f}s",
            flush=True,
        )

    finished_at = datetime.now(UTC)
    provenance_finished = _provenance_snapshot(sample_dirs)
    changed_provenance = [key for key in provenance_started if provenance_started[key] != provenance_finished[key]]
    if changed_provenance:
        changed = ", ".join(changed_provenance)
        raise RuntimeError(f"study inputs changed during execution; refusing artifact commit: {changed}")
    results = {
        "schema": ATLAS_STUDY_SCHEMA,
        "run_id": output_dir.name,
        "config": {
            "tracks": list(tracks),
            "root_seed": args.seed,
            "perturbation_bucket": args.perturbation,
            "replicate": args.replicate,
            "terrain": {
                "extent_m": matrix_config.extent_m,
                "grid": matrix_config.terrain_grid,
                "map_center_offset_fraction": matrix_config.map_center_offset_fraction,
                "native_patch_policy": "cached_swissALTI3D_2m_centred_on_supplied_position_prior",
                "evaluation_patch_isolated_from_estimator": True,
            },
            "atlas": atlas_config.to_record(),
        },
        "samples": samples,
        "aggregates": _aggregates(samples, tracks),
    }
    result_bytes = _json_bytes(results)
    summary_bytes = _summary_markdown(results).encode()
    run = {
        "schema": ATLAS_STUDY_SCHEMA,
        "run_id": output_dir.name,
        "status": "complete",
        "created_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "wall_runtime_s": round((finished_at - started_at).total_seconds(), 3),
        "results_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "inputs": provenance_started["inputs"],
        "implementation": provenance_started["implementation"],
        "environment": _environment_record(),
        "terrain_cache": provenance_started["terrain_cache"],
        "provenance_stability": {
            "captured_before_first_sample": True,
            "rechecked_before_artifact_commit": True,
            "changed_sections": [],
        },
        "limitations": [
            "PFM oracle uses source depth and is analysis-only",
            "the PFM source depth was generated at the reference pose",
            "automatic-photo evidence is a hard skyline selected without reference truth",
            "the atlas fixes eye height and supplied FOV while fitting bounded pitch/roll crop nuisances",
            "the reported ceiling covers horizontal position and yaw; pitch and roll are not graded",
            "full-lattice and shortlist GT oracles never replace the estimator-ranked winner",
            "the regional terrain frame is sample-centred for controlled data provisioning",
            "the three-photo control is diagnostic and not a calibrated dataset-wide rate",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_once(output_dir / "results.json", result_bytes)
    _write_once(output_dir / "summary.md", summary_bytes)
    _write_once(output_dir / "run.json", _json_bytes(run))
    print(
        f"Committed pose-atlas artifact to {output_dir} (results sha256 {run['results_sha256'][:12]}…).",
        flush=True,
    )


def _run_sample(
    sample_dir: Path,
    *,
    tracks: tuple[str, ...],
    matrix_config: MatrixConfig,
    atlas_config: SkylineAtlasConfig,
    replicate: int,
) -> dict[str, Any]:
    sample = load_sample(sample_dir)
    terrain_selection = load_benchmark_terrain(sample, matrix_config)
    rgb = _load_photo(sample.photo_path)
    height, width = rgb.shape[:2]
    oracle_rows = resampled_oracle_skyline(sample.depth_path, width, height)
    compatibility, photo_edge_support = assess_pre_solve_quality(
        sample,
        terrain_selection,
        [
            EvidenceTrack(
                "pfm_oracle",
                oracle_rows,
                {
                    "source": "source_depth_pfm",
                    "evidence_generated_at_reference_pose": True,
                    "numeric_reference_pose_parameters_used_for_ranking": False,
                },
            )
        ],
        width,
        height,
    )
    scenario = build_prior_scenario(
        sample,
        terrain_selection.terrain,
        terrain_selection.truth,
        "perturbed_metadata",
        matrix_config.perturbation_bucket,
        replicate,
        matrix_config.root_seed,
    )
    estimator_terrain = provision_estimator_terrain(terrain_selection.terrain, scenario)
    evidence = _evidence_records(rgb, oracle_rows, tracks, matrix_config.extractor)
    track_records: dict[str, Any] = {}
    for track in tracks:
        evidence_record = evidence[track]
        rows = evidence_record.pop("_rows")
        if rows is None:
            track_records[track] = {
                "status": "evidence_rejected",
                "evidence": evidence_record,
                "estimator_archive": None,
                "evaluation": None,
            }
            continue
        track_started = time.perf_counter()
        archive = build_skyline_atlas(
            estimator_terrain.terrain,
            rows,
            height,
            sample.fov_deg,
            scenario.prior.position,
            native_patch=estimator_terrain.high_resolution_patch,
            config=atlas_config,
        )
        # Freeze the complete estimator score lattice first.  The controlled synthetic prior was
        # derived from truth, but numeric evaluation truth is only consumed after this boundary.
        archive_record = archive.to_record()
        evaluation = evaluate_skyline_atlas(
            archive,
            terrain_selection.truth,
            top_ks=(1, 5, 10, 25, 50, 100, 250, 500, 1000, len(archive.candidates)),
        )
        reference_probe = _evaluation_reference_probe(
            estimator_terrain.terrain,
            estimator_terrain.high_resolution_patch,
            rows,
            height,
            sample.fov_deg,
            terrain_selection.truth,
            atlas_config,
            blind_winner_score=archive.selected.score,
        )
        track_records[track] = {
            "status": "ok",
            "evidence": evidence_record,
            "runtime_s": round(time.perf_counter() - track_started, 4),
            "estimator_archive": archive_record,
            "evaluation": {
                **evaluation.to_record(),
                "evaluation_only_reference_position_probe": reference_probe,
            },
        }
        winner = evaluation.winner_errors.errors
        oracle = evaluation.full_lattice_gt_oracle.errors
        print(
            f"  {sample.name:<40} {track:<11} winner {winner.horizontal_position_m:7.1f}m/"
            f"{winner.yaw_deg:5.1f}°; full-lattice oracle {oracle.horizontal_position_m:6.1f}m/"
            f"{oracle.yaw_deg:4.1f}°; {track_records[track]['runtime_s']:7.1f}s",
            flush=True,
        )

    return {
        "name": sample.name,
        "manual": sample.manual,
        "reference": terrain_selection.truth.model_dump(mode="json"),
        "reference_source": "refined_geopose_metadata_info_lines_2_6",
        "coordinate_frame_origin": terrain_selection.terrain.spec.origin.model_dump(mode="json"),
        "compatibility": compatibility,
        "photo_edge_support": photo_edge_support,
        "prior": {
            **scenario.prior.model_dump(mode="json"),
            "regime": scenario.name,
            "constructed_from_reference_for_controlled_perturbation": True,
            "perturbation": scenario.perturbation,
            "errors": _pose_errors(
                CameraExtrinsics(
                    position=scenario.prior.position,
                    yaw_deg=scenario.prior.yaw_deg,
                    pitch_deg=scenario.prior.pitch_deg,
                    roll_deg=0.0,
                ),
                terrain_selection.truth,
            ),
        },
        "terrain_inputs": estimator_terrain.provenance,
        "tracks": track_records,
    }


def _evaluation_reference_probe(
    terrain,
    native_patch,
    observed_rows: np.ndarray,
    height_px: int,
    horizontal_fov_deg: float,
    truth: CameraExtrinsics,
    config: SkylineAtlasConfig,
    *,
    blind_winner_score: float,
) -> dict[str, Any]:
    """Score the reference position after freezing the blind archive."""

    probe = build_skyline_atlas(
        terrain,
        observed_rows,
        height_px,
        horizontal_fov_deg,
        truth.position,
        native_patch=native_patch,
        config=replace(config, radius_m=0.0),
    )
    evaluated = evaluate_skyline_atlas(probe, truth, top_ks=(1, len(probe.candidates)))
    return {
        "reference_data_used": True,
        "used_by_estimator": False,
        "included_in_estimator_archive": False,
        "position_source": "refined_geopose_metadata_info_lines_3_6",
        "best_yaw_mode": probe.selected.to_record(),
        "errors": evaluated.winner_errors.errors.to_record(),
        "score_delta_reference_minus_blind_winner": round(probe.selected.score - blind_winner_score, 12),
    }


def _evidence_records(
    rgb: np.ndarray,
    oracle_rows: np.ndarray,
    tracks: tuple[str, ...],
    extractor: str,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {
        "pfm_oracle": {
            "_rows": oracle_rows,
            "source": "source_depth_pfm",
            "analysis_only": True,
            "evidence_generated_at_reference_pose": True,
            "numeric_reference_pose_parameters_used_for_ranking": False,
            "coverage": round(float(np.isfinite(oracle_rows).mean()), 6),
        }
    }
    if "photo_auto" in tracks:
        candidates = extract_candidates(rgb, backend=extractor)
        chosen = best_skyline_candidate(candidates, min_coverage=0.25)
        if chosen is None:
            records["photo_auto"] = {
                "_rows": None,
                "source": extractor,
                "available": False,
                "reason": "no_candidate_with_25pct_coverage",
                "selection_uses_reference_truth": False,
                "evidence_generated_at_reference_pose": False,
            }
        else:
            name, candidate = chosen
            records["photo_auto"] = {
                "_rows": candidate.rows,
                "source": extractor,
                "available": True,
                "candidate": name,
                "coverage": round(candidate.coverage, 6),
                "agreement": round(candidate.agreement, 6),
                "detected_candidates": sorted(candidates),
                "selection_uses_reference_truth": False,
                "evidence_generated_at_reference_pose": False,
            }
    return records


def _aggregates(samples: list[dict[str, Any]], tracks: tuple[str, ...]) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    for track in tracks:
        records = [sample["tracks"].get(track) for sample in samples]
        completed = [record for record in records if isinstance(record, dict) and record.get("status") == "ok"]
        winner_errors = [record["evaluation"]["winner_errors"] for record in completed]
        oracle_errors = [record["evaluation"]["full_lattice_gt_oracle"] for record in completed]
        aggregates.append(
            {
                "track": track,
                "samples": len(records),
                "completed": len(completed),
                "evidence_rejected": len(records) - len(completed),
                "blind_winner_successes": sum(item["reaches_target"] for item in winner_errors),
                "full_lattice_oracle_successes": sum(item["reaches_target"] for item in oracle_errors),
                "median_blind_winner_horizontal_m": _median_nested(winner_errors, "horizontal_position_m"),
                "median_blind_winner_yaw_deg": _median_nested(winner_errors, "yaw_deg"),
                "median_full_lattice_oracle_horizontal_m": _median_nested(oracle_errors, "horizontal_position_m"),
                "median_full_lattice_oracle_yaw_deg": _median_nested(oracle_errors, "yaw_deg"),
                "runtime_s": round(sum(float(record.get("runtime_s") or 0.0) for record in completed), 4),
            }
        )
    return aggregates


def _median_nested(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record["errors"][key]) for record in records if record.get("errors", {}).get(key) is not None]
    return round(statistics.median(values), 5) if values else None


def _summary_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# Peakle horizontal-position/yaw atlas ceiling study",
        "",
        "Blind winners are estimator-selected. Full-lattice oracles are evaluation-only and never replace them.",
        "",
        "| sample | track | blind winner | full-lattice oracle | shortlist top-100 reaches target | runtime s |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for sample in results["samples"]:
        for track, record in sample["tracks"].items():
            if record["status"] != "ok":
                lines.append(f"| {sample['name']} | {track} | rejected | - | - | - |")
                continue
            evaluation = record["evaluation"]
            winner = evaluation["winner_errors"]["errors"]
            oracle = evaluation["full_lattice_gt_oracle"]["errors"]
            top100 = next(item for item in evaluation["shortlist_top_k"] if item["requested_k"] == 100)
            lines.append(
                f"| {sample['name']} | {track} | {winner['horizontal_position_m']:.1f} m / "
                f"{winner['yaw_deg']:.1f}° | {oracle['horizontal_position_m']:.1f} m / "
                f"{oracle['yaw_deg']:.1f}° | {'yes' if top100['reaches_target'] else 'no'} | "
                f"{record['runtime_s']:.1f} |"
            )
    return "\n".join(lines) + "\n"


def _pose_errors(estimate: CameraExtrinsics, truth: CameraExtrinsics) -> dict[str, float | None]:
    east = estimate.position.east_m - truth.position.east_m
    north = estimate.position.north_m - truth.position.north_m
    up = estimate.position.up_m - truth.position.up_m
    return {
        "horizontal_position_m": round(math.hypot(east, north), 6),
        "vertical_m": round(abs(up), 6),
        "yaw_deg": round(abs((estimate.yaw_deg - truth.yaw_deg + 180.0) % 360.0 - 180.0), 6),
        "pitch_deg": None,
    }


def _load_photo(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if image.width > STD_WIDTH:
        scale = STD_WIDTH / image.width
        image = image.resize((STD_WIDTH, max(1, round(image.height * scale))), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def _selected_samples(value: str) -> list[Path]:
    requested = [item.strip() for item in value.split(",") if item.strip()]
    duplicates = sorted({name for name in requested if requested.count(name) > 1})
    if duplicates:
        raise SystemExit(f"duplicate GeoPose samples: {', '.join(duplicates)}")
    by_name = {path.name: path for path in find_sample_dirs()}
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise SystemExit(f"unknown or incomplete GeoPose samples: {', '.join(missing)}")
    if not requested:
        raise SystemExit("at least one sample is required")
    return [by_name[name] for name in requested]


def _csv_tracks(value: str) -> tuple[str, ...]:
    tracks = tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    unknown = set(tracks) - set(TRACKS)
    if unknown:
        raise SystemExit(f"unknown evidence tracks: {', '.join(sorted(unknown))}")
    if not tracks:
        raise SystemExit("at least one evidence track is required")
    return tracks


def _position_count(config: SkylineAtlasConfig) -> int:
    side = 2 * int(math.floor(config.radius_m / config.spacing_m + 1e-12)) + 1
    return side * side


def _implementation_record() -> dict[str, Any]:
    paths = [
        Path(__file__),
        BASE / "src/peakle/localize/skyline_atlas.py",
        BASE / "src/peakle/localize/solve.py",
        BASE / "src/peakle/localize/raycast.py",
        BASE / "src/peakle/localize/strategy_bench.py",
        BASE / "src/peakle/localize/extract.py",
        BASE / "src/peakle/localize/geopose.py",
        BASE / "src/peakle/localize/compatibility.py",
        BASE / "src/peakle/localize/swissdem.py",
        BASE / "src/peakle/domain/camera.py",
        BASE / "src/peakle/domain/coordinates.py",
        BASE / "src/peakle/domain/projection.py",
        BASE / "src/peakle/domain/terrain.py",
    ]
    files = [{"path": str(path.relative_to(BASE)), "sha256": file_sha256(path)} for path in paths]
    aggregate = hashlib.sha256(_json_bytes(files)).hexdigest()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=BASE,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    relative_paths = [item["path"] for item in files]
    status = subprocess.run(
        ["git", "status", "--short", "--", *relative_paths],
        cwd=BASE,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    diff = subprocess.run(
        ["git", "diff", "--binary", "--", *relative_paths],
        cwd=BASE,
        check=True,
        capture_output=True,
    ).stdout
    return {
        "git_revision": revision,
        "aggregate_sha256": aggregate,
        "files": files,
        "source_worktree_status": status,
        "tracked_source_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _provenance_snapshot(sample_dirs: list[Path]) -> dict[str, Any]:
    return {
        "inputs": input_fingerprint(sample_dirs),
        "implementation": _implementation_record(),
        "terrain_cache": _compact_cache_inventory(default_terrain_cache_inventory()),
    }


def _compact_cache_inventory(inventory: dict[str, Any]) -> dict[str, Any]:
    """Attest the available cache without embedding thousands of unrelated entries."""

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


def _environment_record() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "scipy", "pillow", "pydantic"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default=",".join(DEFAULT_SAMPLES))
    parser.add_argument("--tracks", default=",".join(TRACKS))
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--perturbation", choices=("mild", "standard", "hard"), default="standard")
    parser.add_argument("--replicate", type=_nonnegative_int, default=0)
    parser.add_argument("--extent-km", type=float, default=40.0)
    parser.add_argument("--grid", type=int, default=1335)
    parser.add_argument("--map-offset-fraction", type=float, default=0.16)
    parser.add_argument("--extractor", choices=MATRIX_EXTRACTORS, default="color")
    parser.add_argument("--radius-m", type=float, default=500.0)
    parser.add_argument("--spacing-m", type=float, default=50.0)
    parser.add_argument("--yaw-step-deg", type=float, default=1.0)
    parser.add_argument("--yaw-modes-per-position", type=int, default=3)
    parser.add_argument("--yaw-mode-separation-deg", type=float, default=8.0)
    parser.add_argument("--max-observed-columns", type=int, default=192)
    parser.add_argument("--residual-cap-px", type=float, default=60.0)
    parser.add_argument("--high-outlier-trim-fraction", type=float, default=0.10)
    parser.add_argument("--high-outlier-weight", type=float, default=0.02)
    parser.add_argument("--max-abs-roll-nuisance-deg", type=float, default=10.0)
    parser.add_argument("--eye-height-m", type=float, default=2.5)
    parser.add_argument("--ray-step-m", type=float, default=10.0)
    parser.add_argument("--output", required=True, help="new write-once output directory")
    return parser


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


if __name__ == "__main__":
    main()

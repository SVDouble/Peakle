"""Read-only discovery and normalization of immutable pose benchmark artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from peakle.localize.atlas_dashboard import (
    ATLAS_DASHBOARD_FILENAME,
    ATLAS_DASHBOARD_SCHEMA,
    ATLAS_STUDY_SCHEMA,
    build_atlas_dashboard,
    canonical_json_bytes,
    validated_atlas_dashboard,
)
from peakle.localize.strategy_bench import aggregate_matrix

router = APIRouter(tags=["benchmarks"])

BASE = Path(__file__).resolve().parents[4]
OUTPUT = BASE / "local/output"
RESULT_GLOB = "*-geopose-bench/results.json"
ATLAS_ALGORITHM = "skyline-atlas"
ATLAS_PRIOR_REGIME = "controlled_reference_perturbation"
ATLAS_CACHE_ROOT = BASE / "local/cache/pose_atlas_dashboard"
_ATLAS_LOAD_LOCK = threading.Lock()


@dataclass(frozen=True)
class _AtlasArtifact:
    payload: dict[str, Any]
    results_sha256: str
    compact_source: str
    dashboard_hash_verified: bool | None


def _discover_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for result_path in OUTPUT.glob(RESULT_GLOB):
        payload = _read_result(result_path)
        if payload is None:
            continue
        rows = payload["rows"]
        matrix_cases = payload["matrix_cases"]
        stored_metadata = _read_json(result_path.parent / "run.json")
        metadata: dict[str, Any] = stored_metadata if stored_metadata is not None else {}
        matrix_value = metadata.get("matrix")
        matrix_metadata: dict[str, Any] = matrix_value if isinstance(matrix_value, dict) else {}
        render_matching_value = metadata.get("render_matching")
        render_matching: dict[str, Any] = render_matching_value if isinstance(render_matching_value, dict) else {}
        candidate_validation_value = render_matching.get("candidate_validation")
        candidate_validation = candidate_validation_value if isinstance(candidate_validation_value, dict) else None
        algorithms = sorted({str(case.get("algorithm")) for case in matrix_cases if case.get("algorithm")})
        if not algorithms:
            configured = matrix_metadata.get("algorithms") if isinstance(matrix_metadata, dict) else None
            algorithms = [str(value) for value in configured] if isinstance(configured, list) else ["horizon"]
        evidence = sorted({str(case.get("evidence_track")) for case in matrix_cases if case.get("evidence_track")})
        regimes = sorted({str(case.get("prior_regime")) for case in matrix_cases if case.get("prior_regime")})
        if not evidence:
            evidence = ["pfm_oracle", "photo_auto"]
        if not regimes:
            regimes = ["known_position_no_orientation_prior"]
        expected_hash = metadata.get("results_sha256")
        actual_hash = _sha256(result_path) if expected_hash else None
        hash_verified = actual_hash == expected_hash if expected_hash else None
        code_value = metadata.get("code")
        code_metadata: dict[str, Any] = code_value if isinstance(code_value, dict) else {}
        implementation_value = code_metadata.get("implementation")
        implementation: dict[str, Any] = implementation_value if isinstance(implementation_value, dict) else {}
        attempted_cases = [case for case in matrix_cases if case.get("status") != "skipped"]
        primary_cases = [case for case in attempted_cases if case.get("ranking_eligible") is True]
        valid_rows = [row for row in rows if "error" not in row]
        run_id = result_path.parent.name
        run = {
            "id": run_id,
            "created_at": metadata.get("created_at") or _legacy_created_at(run_id),
            "status": metadata.get("status", "legacy"),
            "schema_version": payload["schema_version"],
            "kind": "strategy_matrix" if payload["schema_version"] >= 2 else "legacy_orientation",
            "sample_count": len(rows),
            "completed_sample_count": len(valid_rows),
            "error_count": len(rows) - len(valid_rows),
            "matrix_case_count": len(matrix_cases),
            "attempted_case_count": len(attempted_cases),
            "primary_case_count": len(primary_cases),
            "algorithm_count": len(algorithms),
            "algorithms": algorithms,
            "evidence_tracks": evidence,
            "prior_regimes": regimes,
            "compatibility_policy": (
                "gt_dem_compat_v1"
                if any(isinstance(row.get("gt_dem_compatibility"), dict) for row in valid_rows)
                else "legacy_vertical_shift_chamfer_px"
            ),
            "has_provenance": bool(metadata),
            "hash_verified": hash_verified,
            "results_sha256": expected_hash,
            "git_sha": code_metadata.get("git_sha"),
            "git_diff_sha256": code_metadata.get("git_diff_sha256"),
            "implementation_sha256": implementation.get("aggregate_sha256"),
            "dirty_code": code_metadata.get("dirty"),
            "candidate_validation": _json_safe(candidate_validation),
            "_path": result_path,
            "_metadata": metadata,
        }
        runs.append(run)
    discovered_paths = {run["_path"] for run in runs}
    for metadata_path in OUTPUT.glob("*/run.json"):
        atlas_metadata = _read_json(metadata_path)
        if atlas_metadata is None or atlas_metadata.get("schema") != ATLAS_STUDY_SCHEMA:
            continue
        result_path = metadata_path.parent / "results.json"
        if atlas_metadata.get("run_id") not in {None, result_path.parent.name}:
            continue
        if result_path in discovered_paths:
            continue
        artifact = _read_atlas_artifact(result_path, atlas_metadata)
        if artifact is None:
            continue
        payload = artifact.payload
        if payload.get("run_id") not in {None, result_path.parent.name}:
            continue
        runs.append(_atlas_run(result_path, atlas_metadata, artifact))
    runs.sort(key=lambda run: str(run["created_at"]), reverse=True)
    eligible = [
        run
        for run in runs
        if run["kind"] == "strategy_matrix"
        and run["status"] == "complete"
        and run["primary_case_count"] > 0
        and run["hash_verified"] is not False
    ]
    if eligible:
        eligible[0]["recommended"] = True
    return runs


def _atlas_run(
    result_path: Path,
    metadata: dict[str, Any],
    artifact: _AtlasArtifact,
) -> dict[str, Any]:
    payload = artifact.payload
    samples = payload["samples"]
    tracks = _atlas_track_names(payload)
    track_records = [
        sample.get("tracks", {}).get(track)
        for sample in samples
        for track in tracks
        if isinstance(sample.get("tracks"), dict)
    ]
    completed_tracks = sum(isinstance(record, dict) and record.get("status") == "ok" for record in track_records)
    evidence_rejected = sum(
        isinstance(record, dict) and record.get("status") == "evidence_rejected" for record in track_records
    )
    track_errors = sum(isinstance(record, dict) and record.get("status") == "error" for record in track_records)
    implementation_value = metadata.get("implementation")
    implementation = implementation_value if isinstance(implementation_value, dict) else {}
    expected_hash = metadata.get("results_sha256")
    dashboard_value = metadata.get("dashboard")
    dashboard = dashboard_value if isinstance(dashboard_value, dict) else {}
    atlas_config = payload.get("config", {}).get("atlas")
    return {
        "id": result_path.parent.name,
        "created_at": metadata.get("created_at") or _legacy_created_at(result_path.parent.name),
        "status": metadata.get("status", "legacy"),
        "schema_version": 2,
        "artifact_schema": ATLAS_STUDY_SCHEMA,
        "kind": "pose_atlas",
        "sample_count": len(samples),
        "completed_sample_count": len(samples),
        "error_count": track_errors,
        "matrix_case_count": 0,
        "attempted_case_count": len(samples) * len(tracks),
        "primary_case_count": 0,
        "algorithm_count": 1,
        "algorithms": [ATLAS_ALGORITHM],
        "evidence_tracks": tracks,
        "track_count": len(tracks),
        "completed_track_count": completed_tracks,
        "evidence_rejected_count": evidence_rejected,
        "prior_regimes": [ATLAS_PRIOR_REGIME],
        "compatibility_policy": "gt_dem_compat_v1",
        "has_provenance": bool(metadata),
        "hash_verified": artifact.results_sha256 == expected_hash if expected_hash else None,
        "results_sha256": expected_hash,
        "dashboard_sha256": dashboard.get("sha256"),
        "dashboard_hash_verified": artifact.dashboard_hash_verified,
        "compact_source": artifact.compact_source,
        "git_sha": implementation.get("git_revision"),
        "git_diff_sha256": implementation.get("tracked_source_diff_sha256"),
        "implementation_sha256": implementation.get("aggregate_sha256"),
        "dirty_code": bool(implementation.get("source_worktree_status")),
        "candidate_validation": None,
        "atlas_config": _json_safe(atlas_config),
        "_path": result_path,
        "_metadata": metadata,
        "_payload": payload,
    }


def _public_run(run: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in run.items() if not key.startswith("_")}


def _run(run_id: str) -> dict[str, Any]:
    for run in _discover_runs():
        if run["id"] == run_id:
            return run
    raise HTTPException(status_code=404, detail=f"unknown benchmark run {run_id!r}")


@router.get("/bench/runs")
def list_benchmark_runs() -> list[dict[str, Any]]:
    """Available immutable benchmark artifacts, newest first."""

    return [_public_run(run) for run in _discover_runs()]


@router.get("/bench/compatibility")
async def get_compatibility_policy() -> dict[str, Any]:
    """Machine-readable contracts for the two independent dataset gates."""

    return {
        "policy": "gt_dem_compat_v1",
        "status": "provisional_initial_thresholds",
        "inputs": ["source_depth_pfm", "raw_metadata_pose", "selected_terrain_stack"],
        "forbidden_inputs": ["gt_v2_refined_pose", "photo_skyline", "solver_output"],
        "fixed": ["position", "yaw", "fov", "terrain"],
        "nuisance": "cross_fitted_global_vertical_crop_shift",
        "metrics": [
            "trimmed_symmetric_angular_chamfer_deg",
            "median_angular_error_deg",
            "p90_angular_error_deg",
            "valid_column_coverage",
            "crop_shift_px",
        ],
        "tiers": {
            "MAP_A": {"median_deg_lte": 0.25, "p90_deg_lte": 0.75, "coverage_gte": 0.90},
            "MAP_B": {"median_deg_lte": 0.50, "p90_deg_lte": 1.50, "coverage_gte": 0.80},
            "MAP_C": {"description": "outside MAP_A/MAP_B; excluded from primary solver ranking"},
        },
        "height_gate": {
            "policy": "raw_camera_clearance_v1",
            "status": "provisional_initial_thresholds",
            "metric": "raw_metadata_camera_elevation_minus_dem_ground_m",
            "tiers": {
                "HEIGHT_A": {"clearance_m_gte": -2.0, "clearance_m_lte": 10.0},
                "HEIGHT_B": {"clearance_m_gte": -10.0, "clearance_m_lte": 50.0},
                "HEIGHT_C": {"description": "outside HEIGHT_A/HEIGHT_B"},
            },
            "note": "This physical altitude/datum check is never fitted and does not alter the skyline MAP tier.",
        },
    }


@router.get("/bench/runs/{run_id}")
def get_benchmark_run(run_id: str) -> dict[str, Any]:
    run = _run(run_id)
    return _public_run(run) | {"provenance": _json_safe(run["_metadata"])}


@router.get("/bench/runs/{run_id}/summary")
def get_benchmark_summary(run_id: str) -> dict[str, Any]:
    run = _run(run_id)
    payload = _payload(run)
    if run["kind"] == "strategy_matrix":
        return _matrix_summary(run, payload)
    if run["kind"] == "pose_atlas":
        return _atlas_summary(run, payload)
    return _legacy_summary(run, payload["rows"])


@router.get("/bench/runs/{run_id}/cases")
def get_benchmark_cases(
    run_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=5000),
    subset: str = Query(default="all"),
    query: str = Query(default=""),
    algorithm: str | None = Query(default=None),
    evidence: str | None = Query(default=None),
    prior_regime: str | None = Query(default=None),
) -> dict[str, Any]:
    run = _run(run_id)
    payload = _payload(run)
    needle = query.casefold().strip()
    if run["kind"] == "pose_atlas":
        samples = _filter_atlas_samples(payload["samples"], subset)
        tracks = _atlas_track_names(payload)
        cases = [_atlas_case(sample, track) for sample in samples for track in tracks]
        if algorithm and algorithm != ATLAS_ALGORITHM:
            cases = []
        if evidence:
            cases = [case for case in cases if case.get("evidence_track") == evidence]
        if prior_regime and prior_regime != ATLAS_PRIOR_REGIME:
            cases = []
        if needle:
            cases = [case for case in cases if needle in str(case.get("name", "")).casefold()]
        cases.sort(key=lambda case: (str(case["name"]), str(case["evidence_track"])))
        return {
            "mode": "atlas",
            "total": len(cases),
            "offset": offset,
            "limit": limit,
            "rows": cases[offset : offset + limit],
        }
    if run["kind"] == "strategy_matrix":
        compatibility = _compatibility_by_name(payload["rows"])
        cases = _filter_matrix_cases(payload["matrix_cases"], subset, compatibility)
        cases = [case for case in cases if case.get("status") != "skipped"]
        if algorithm:
            cases = [case for case in cases if case.get("algorithm") == algorithm]
        if evidence:
            cases = [case for case in cases if case.get("evidence_track") == evidence]
        if prior_regime:
            cases = [case for case in cases if case.get("prior_regime") == prior_regime]
        if needle:
            cases = [case for case in cases if needle in str(case.get("name", "")).casefold()]
        public = [_matrix_case(case, compatibility.get(str(case.get("name")), {})) for case in cases]
        public.sort(
            key=lambda case: (
                case["status"] == "skipped",
                case["success"] is True,
                case["name"],
                case["algorithm"],
                case["evidence_track"],
            )
        )
        return {
            "mode": "matrix",
            "total": len(public),
            "offset": offset,
            "limit": limit,
            "rows": public[offset : offset + limit],
        }

    rows = _filter_legacy_rows(payload["rows"], subset)
    if needle:
        rows = [row for row in rows if needle in str(row.get("name", "")).casefold()]
    public = [_legacy_case(row) for row in rows]
    public.sort(key=lambda case: (_compatibility_sort(case["compatibility"]), case["name"]))
    return {
        "mode": "legacy",
        "total": len(public),
        "offset": offset,
        "limit": limit,
        "rows": public[offset : offset + limit],
    }


def _atlas_summary(run: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    samples = payload["samples"]
    tracks = _atlas_track_names(payload)
    subsets: dict[str, Any] = {}
    for key in ("all", "manual", "map_ab", "map_ab_height_a", "map_a", "map_a_height_a"):
        selected = _filter_atlas_samples(samples, key)
        if not selected and key not in {"all", "manual"}:
            continue
        aggregates = [_atlas_track_aggregate(selected, track) for track in tracks]
        subsets[key] = {
            "sample_count": len(selected),
            "requested_track_count": len(selected) * len(tracks),
            "completed_track_count": sum(item["completed"] for item in aggregates),
            "evidence_rejected_count": sum(item["evidence_rejected"] for item in aggregates),
            "error_count": sum(item["errors"] for item in aggregates),
            "aggregates": aggregates,
        }
    default_subset = (
        "map_ab_height_a"
        if subsets.get("map_ab_height_a", {}).get("sample_count", 0)
        else "map_ab"
        if subsets.get("map_ab", {}).get("sample_count", 0)
        else "all"
    )
    warnings = [
        "This is a diagnostic search-ceiling study, not a ranking-eligible strategy-matrix run.",
        "Blind winners are estimator-selected. Full-lattice GT oracles use reference truth only after the score "
        "lattice is frozen and never replace the estimator output.",
        "PFM / source-depth evidence was generated at the reference pose and is analysis-only.",
        "The atlas prior is a controlled, reference-derived horizontal perturbation. Only east/north centres "
        "the grid; recorded yaw, pitch and altitude perturbations do not constrain scoring.",
        "GeoPose crop pitch and the fitted roll nuisance are not graded.",
    ]
    if run.get("dirty_code"):
        warnings.append(
            "This artifact was produced from modified attested source paths; "
            "use its implementation hash for comparison."
        )
    return {
        "mode": "atlas",
        "run": _public_run(run),
        "default_subset": default_subset,
        "warnings": warnings,
        "success_contract": {
            "horizontal_position_error_m_lte": 100.0,
            "absolute_yaw_error_deg_lte": 5.0,
            "pitch_scored": False,
        },
        "subsets": subsets,
    }


def _filter_atlas_samples(samples: list[dict[str, Any]], subset: str) -> list[dict[str, Any]]:
    if subset == "all":
        return list(samples)
    if subset == "manual":
        return [sample for sample in samples if sample.get("manual")]
    if subset not in {"map_ab", "map_ab_height_a", "map_a", "map_a_height_a"}:
        raise HTTPException(status_code=400, detail=f"unknown subset {subset!r}")
    selected: list[dict[str, Any]] = []
    for sample in samples:
        compatibility = sample.get("compatibility")
        compat = compatibility if isinstance(compatibility, dict) else {}
        tier = compat.get("tier")
        if subset in {"map_ab", "map_ab_height_a"} and tier not in {"MAP_A", "MAP_B"}:
            continue
        if subset not in {"map_ab", "map_ab_height_a"} and tier != "MAP_A":
            continue
        if subset in {"map_ab_height_a", "map_a_height_a"} and _height_tier_from_compat(compat) != "HEIGHT_A":
            continue
        selected.append(sample)
    return selected


def _atlas_track_names(payload: dict[str, Any]) -> list[str]:
    config_value = payload.get("config")
    config = config_value if isinstance(config_value, dict) else {}
    configured = config.get("tracks")
    if isinstance(configured, list):
        names = [str(value) for value in configured if value]
        if names:
            return list(dict.fromkeys(names))
    return sorted(
        {
            str(track)
            for sample in payload.get("samples", [])
            if isinstance(sample, dict) and isinstance(sample.get("tracks"), dict)
            for track in sample["tracks"]
        }
    )


def _atlas_track_aggregate(samples: list[dict[str, Any]], track: str) -> dict[str, Any]:
    records = [sample.get("tracks", {}).get(track) for sample in samples if isinstance(sample.get("tracks"), dict)]
    completed = [record for record in records if isinstance(record, dict) and record.get("status") == "ok"]
    winner = [
        evaluation["blind_winner"]
        for record in completed
        if isinstance((evaluation := record.get("evaluation")), dict)
        and isinstance(evaluation.get("blind_winner"), dict)
    ]
    oracle = [
        evaluation["full_lattice_oracle"]
        for record in completed
        if isinstance((evaluation := record.get("evaluation")), dict)
        and isinstance(evaluation.get("full_lattice_oracle"), dict)
    ]
    requested = len(samples)
    blind_successes = sum(item.get("reaches_target") is True for item in winner)
    oracle_successes = sum(item.get("reaches_target") is True for item in oracle)
    top_100_successes = 0
    for record in completed:
        evaluation = record.get("evaluation")
        if not isinstance(evaluation, dict):
            continue
        top_100 = next(
            (
                item
                for item in evaluation.get("shortlist_top_k", [])
                if isinstance(item, dict) and item.get("requested_k") == 100
            ),
            None,
        )
        top_100_successes += isinstance(top_100, dict) and top_100.get("reaches_target") is True
    return {
        "track": track,
        "requested": requested,
        "completed": len(completed),
        "evidence_rejected": sum(
            isinstance(record, dict) and record.get("status") == "evidence_rejected" for record in records
        ),
        "errors": sum(isinstance(record, dict) and record.get("status") == "error" for record in records),
        "missing": sum(not isinstance(record, dict) for record in records),
        "blind_winner_successes": blind_successes,
        "blind_winner_success_rate": round(blind_successes / requested, 4) if requested else None,
        "full_lattice_oracle_successes": oracle_successes,
        "full_lattice_oracle_success_rate": round(oracle_successes / requested, 4) if requested else None,
        "median_blind_winner_horizontal_m": _atlas_candidate_median(winner, "horizontal_position_m"),
        "median_blind_winner_yaw_deg": _atlas_candidate_median(winner, "yaw_deg"),
        "median_full_lattice_oracle_horizontal_m": _atlas_candidate_median(oracle, "horizontal_position_m"),
        "median_full_lattice_oracle_yaw_deg": _atlas_candidate_median(oracle, "yaw_deg"),
        "median_full_lattice_oracle_estimator_rank": _median_numbers([item.get("estimator_rank") for item in oracle]),
        "shortlist_top_100_successes": top_100_successes,
        "runtime_s": round(
            sum(_json_number(record.get("runtime_s")) or 0.0 for record in completed if isinstance(record, dict)), 4
        ),
    }


def _atlas_candidate_median(candidates: list[dict[str, Any]], key: str) -> float | None:
    return _median_numbers(
        [candidate.get("errors", {}).get(key) for candidate in candidates if isinstance(candidate.get("errors"), dict)]
    )


def _median_numbers(values: list[Any]) -> float | None:
    numbers = [number for value in values if (number := _json_number(value)) is not None]
    return round(median(numbers), 5) if numbers else None


def _atlas_case(sample: dict[str, Any], track: str) -> dict[str, Any]:
    tracks_value = sample.get("tracks")
    tracks = tracks_value if isinstance(tracks_value, dict) else {}
    record_value = tracks.get(track)
    record = record_value if isinstance(record_value, dict) else {}
    evaluation_value = record.get("evaluation")
    evaluation = evaluation_value if isinstance(evaluation_value, dict) else {}
    shortlist = [item for item in evaluation.get("shortlist_top_k", []) if isinstance(item, dict)]
    top_100 = next(
        (item for item in shortlist if item.get("requested_k") == 100),
        None,
    )
    reached = [
        item
        for item in shortlist
        if item.get("reaches_target") is True and _json_number(item.get("requested_k")) is not None
    ]
    first_reach = min(reached, key=lambda item: float(item["requested_k"])) if reached else None
    return _json_safe(
        {
            "id": f"{sample.get('name', 'unknown')}:{track}",
            "name": str(sample.get("name", "unknown")),
            "manual": bool(sample.get("manual")),
            "algorithm": ATLAS_ALGORITHM,
            "evidence_track": track,
            "prior_regime": ATLAS_PRIOR_REGIME,
            "status": record.get("status", "missing"),
            "runtime_s": record.get("runtime_s"),
            "compatibility": sample.get("compatibility"),
            "photo_edge_support": sample.get("photo_edge_support"),
            "prior_errors": sample.get("prior_errors"),
            "prior_context": sample.get("prior_context"),
            "evidence": record.get("evidence"),
            "blind_winner": evaluation.get("blind_winner"),
            "full_lattice_oracle": evaluation.get("full_lattice_oracle"),
            "shortlist_top_100": top_100,
            "shortlist_first_reach": first_reach,
            "selection_regret": evaluation.get("selection_regret"),
            "target": evaluation.get("target"),
            "reference_position_probe": evaluation.get("reference_position_probe"),
            "archive": record.get("archive"),
        }
    )


def _matrix_summary(run: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    cases = payload["matrix_cases"]
    compatibility = _compatibility_by_name(payload["rows"])
    subset_keys = (
        "all",
        "manual",
        "primary",
        "primary_height_a",
        "map_ab",
        "map_a",
        "map_a_height_a",
        "map_a_photo",
    )
    subsets: dict[str, Any] = {}
    for key in subset_keys:
        selected = _filter_matrix_cases(cases, key, compatibility)
        if not selected and key not in {"all", "manual"}:
            continue
        attempted = [case for case in selected if case.get("status") != "skipped"]
        subsets[key] = {
            "sample_count": len({str(case.get("name")) for case in attempted}),
            "requested_case_count": len(selected),
            "attempted_case_count": len(attempted),
            "error_count": sum(case.get("status") == "error" for case in attempted),
            "abstention_count": sum(case.get("outcome") == "abstained" for case in attempted),
            "evidence_rejected_count": sum(case.get("outcome") == "evidence_rejected" for case in attempted),
            "aggregates": _json_safe(aggregate_matrix(selected)),
        }
    default_subset = "all"
    # Prefer a clean physical-height stratum when it contains ranking-eligible
    # attempts. Otherwise fall back to the complete primary set before any
    # diagnostic MAP-only subset, which may contain only excluded strategies.
    for candidate in ("primary_height_a", "primary", "map_a_height_a", "map_a"):
        if subsets.get(candidate, {}).get("attempted_case_count", 0) > 0:
            default_subset = candidate
            break
    warnings = [
        "Primary ranking is restricted to manual MAP_A/MAP_B cases and honors every recorded exclusion. "
        "Solver errors remain failures in the denominator.",
        "GeoPose crop pitch is an uncalibrated vertical registration nuisance and is not ranked.",
    ]
    if run.get("dirty_code"):
        warnings.append(
            "This artifact was produced from a dirty worktree; inspect its implementation hash before comparison."
        )
    if run.get("primary_case_count", 0) == 0:
        warnings.append("This run has no ranking-eligible cases; its rows are diagnostic only.")
    return {
        "mode": "matrix",
        "run": _public_run(run),
        "default_subset": default_subset,
        "warnings": warnings,
        "success_contract": {
            "horizontal_position_error_m_lte": 100.0,
            "absolute_yaw_error_deg_lte": 5.0,
            "pitch_scored": False,
        },
        "subsets": subsets,
    }


def _legacy_summary(run: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    subsets = {
        "all": rows,
        "manual": [row for row in rows if row.get("manual")],
        "map_proxy_5px": [row for row in rows if _finite(row.get("gt_consistency_px")) <= 5.0],
        "map_proxy_10px": [row for row in rows if _finite(row.get("gt_consistency_px")) <= 10.0],
        "map_a": [row for row in rows if _compat_tier(row) == "MAP_A"],
        "map_a_height_a": [row for row in rows if _compat_tier(row) == "MAP_A" and _height_tier(row) == "HEIGHT_A"],
        "map_a_photo": [
            row for row in rows if _compat_tier(row) == "MAP_A" and _photo_edge_support(row).get("usable") is True
        ],
        "map_b": [row for row in rows if _compat_tier(row) in {"MAP_A", "MAP_B"}],
    }
    available = {
        name: {
            "sample_count": len(subset),
            "pfm_oracle": _legacy_track_summary(subset, "oracle"),
            "photo_auto": _legacy_track_summary(subset, "extracted"),
        }
        for name, subset in subsets.items()
        if subset or name in {"all", "manual"}
    }
    default_subset = "map_a_height_a" if "map_a_height_a" in available else "map_a" if "map_a" in available else "all"
    return {
        "mode": "legacy",
        "run": _public_run(run),
        "default_subset": default_subset,
        "warning": (
            "Legacy orientation-only artifact at a known position. It cannot rank full-pose strategies; "
            "the old CONFIRMED calibration is retired and is not reported."
        ),
        "warnings": [
            "Legacy orientation-only artifact at a known position.",
            "Old confidence labels were calibrated on retired GT-v2 data and are ignored.",
        ],
        "subsets": available,
    }


def _filter_matrix_cases(
    cases: list[dict[str, Any]],
    subset: str,
    compatibility: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if subset == "all":
        return list(cases)
    if subset == "manual":
        return [case for case in cases if case.get("manual")]
    if subset == "primary":
        return [case for case in cases if case.get("ranking_eligible") is True]
    if subset not in {"primary_height_a", "map_ab", "map_a", "map_a_height_a", "map_a_photo"}:
        raise HTTPException(status_code=400, detail=f"unknown subset {subset!r}")
    selected: list[dict[str, Any]] = []
    for case in cases:
        name = str(case.get("name"))
        compat = compatibility.get(name, {})
        tier = case.get("compatibility_tier") or compat.get("tier")
        if subset == "primary_height_a":
            if (
                case.get("ranking_eligible") is True
                and tier == "MAP_A"
                and _height_tier_from_compat(compat) == "HEIGHT_A"
            ):
                selected.append(case)
            continue
        if subset == "map_ab" and tier not in {"MAP_A", "MAP_B"}:
            continue
        if subset != "map_ab" and tier != "MAP_A":
            continue
        if subset == "map_a_height_a" and _height_tier_from_compat(compat) != "HEIGHT_A":
            continue
        if subset == "map_a_photo" and _case_photo_edge_supported(case) is not True:
            continue
        selected.append(case)
    return selected


def _filter_legacy_rows(rows: list[dict[str, Any]], subset: str) -> list[dict[str, Any]]:
    if subset == "all":
        return list(rows)
    if subset == "manual":
        return [row for row in rows if row.get("manual")]
    if subset == "map_proxy_5px":
        return [row for row in rows if _finite(row.get("gt_consistency_px")) <= 5.0]
    if subset == "map_proxy_10px":
        return [row for row in rows if _finite(row.get("gt_consistency_px")) <= 10.0]
    if subset == "map_a":
        return [row for row in rows if _compat_tier(row) == "MAP_A"]
    if subset == "map_a_height_a":
        return [row for row in rows if _compat_tier(row) == "MAP_A" and _height_tier(row) == "HEIGHT_A"]
    if subset == "map_a_photo":
        return [row for row in rows if _compat_tier(row) == "MAP_A" and _photo_edge_support(row).get("usable") is True]
    if subset == "map_b":
        return [row for row in rows if _compat_tier(row) in {"MAP_A", "MAP_B"}]
    raise HTTPException(status_code=400, detail=f"unknown subset {subset!r}")


def _matrix_case(case: dict[str, Any], compatibility: dict[str, Any]) -> dict[str, Any]:
    errors = case.get("errors") if isinstance(case.get("errors"), dict) else None
    baseline = case.get("baseline") if isinstance(case.get("baseline"), dict) else None
    result = case.get("result") if isinstance(case.get("result"), dict) else None
    diagnostics = result.get("diagnostics") if isinstance(result, dict) else None
    candidate_validation = diagnostics.get("candidate_validation") if isinstance(diagnostics, dict) else None
    baseline_errors = (
        baseline.get("errors") if isinstance(baseline, dict) and isinstance(baseline.get("errors"), dict) else None
    )
    deltas = None
    if errors and baseline_errors:
        deltas = {
            "horizontal_position_m": _difference(
                errors.get("horizontal_position_m"), baseline_errors.get("horizontal_position_m")
            ),
            "yaw_deg": _difference(errors.get("yaw_deg"), baseline_errors.get("yaw_deg")),
        }
    success_record = case.get("success")
    success = success_record.get("value") if isinstance(success_record, dict) else None
    return _json_safe(
        {
            "id": case.get("id"),
            "name": case.get("name"),
            "manual": bool(case.get("manual")),
            "algorithm": case.get("algorithm"),
            "evidence_track": case.get("evidence_track"),
            "prior_regime": case.get("prior_regime"),
            "status": case.get("status"),
            "outcome": case.get("outcome"),
            "success": success,
            "runtime_s": case.get("runtime_s"),
            "ranking_eligible": bool(case.get("ranking_eligible")),
            "ranking_exclusions": case.get("ranking_exclusions") or [],
            "skip_reason": case.get("skip_reason"),
            "error": case.get("error"),
            "errors": errors,
            "baseline_errors": baseline_errors,
            "delta_vs_prior": deltas,
            "compatibility": compatibility or {"tier": case.get("compatibility_tier")},
            "photo_edge_supported": _case_photo_edge_supported(case),
            "evidence": case.get("evidence"),
            "original_metadata_diagnostic": case.get("original_metadata_diagnostic"),
            "candidate_validation": candidate_validation,
        }
    )


def _legacy_case(row: dict[str, Any]) -> dict[str, Any]:
    compatibility = row.get("gt_dem_compatibility")
    if isinstance(compatibility, dict):
        compat = _json_safe(compatibility)
    else:
        proxy = _json_number(row.get("gt_consistency_px"))
        compat = {
            "policy": "legacy_vertical_shift_chamfer_px",
            "tier": (
                "PROXY_A"
                if proxy is not None and proxy <= 5
                else "PROXY_B"
                if proxy is not None and proxy <= 10
                else "PROXY_C"
            ),
            "proxy_px": proxy,
        }
    return {
        "name": str(row.get("name", "unknown")),
        "manual": bool(row.get("manual")),
        "status": "error" if "error" in row else "ok",
        "error": row.get("error"),
        "fov_deg": _json_number(row.get("fov_deg")),
        "gt_yaw_deg": _json_number(row.get("gt_yaw")),
        "compatibility": compat,
        "pfm_oracle": _legacy_track_case(row.get("oracle")),
        "photo_auto": _legacy_track_case(row.get("extracted")),
        "extraction_error_px": _json_number(row.get("extraction_err_px")),
        "photo_edge_support": _photo_edge_support(row),
        "original_metadata_diagnostic": _json_safe(row.get("original_metadata_diagnostic")),
    }


def _legacy_track_case(track: Any) -> dict[str, Any] | None:
    if not isinstance(track, dict):
        return None
    return {
        "success": bool(track.get("correct")),
        "yaw_error_deg": _json_number(track.get("yaw_err")),
        "fit_px": _json_number(track.get("chamfer_px")),
        "verdict": "UNCALIBRATED" if track.get("verdict") == "CONFIRMED" else track.get("verdict"),
        "coverage": _json_number(track.get("coverage")),
        "alias_ratio": _json_number(track.get("alias_ratio")),
    }


def _legacy_track_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    tracks: list[dict[str, Any]] = []
    for row in rows:
        track = row.get(key)
        tracks.append(track if isinstance(track, dict) else {})
    successes = sum(bool(track.get("correct")) for track in tracks)
    yaw_errors = [abs(value) for track in tracks if (value := _json_number(track.get("yaw_err"))) is not None]
    fits = [value for track in tracks if (value := _json_number(track.get("chamfer_px"))) is not None]
    return {
        "attempts": len(rows),
        "reported_results": sum(bool(track) for track in tracks),
        "errors_or_missing": len(rows) - sum(bool(track) for track in tracks),
        "successes": successes,
        "success_rate": round(successes / len(rows), 4) if rows else None,
        "median_abs_yaw_error_deg": round(median(yaw_errors), 3) if yaw_errors else None,
        "median_fit_px": round(median(fits), 3) if fits else None,
        "confidence_policy": "retired_gt_v2_calibration_not_reported",
    }


def _payload(run: dict[str, Any]) -> dict[str, Any]:
    if run["kind"] == "pose_atlas":
        payload = run.get("_payload")
        if isinstance(payload, dict):
            return payload
        artifact = _read_atlas_artifact(run["_path"], run["_metadata"])
        if artifact is not None:
            return artifact.payload
        raise HTTPException(status_code=500, detail="pose-atlas artifact became unreadable")
    payload = _read_result(run["_path"])
    if payload is None:
        raise HTTPException(status_code=500, detail="benchmark artifact became unreadable")
    return payload


def _read_atlas_artifact(path: Path, metadata: dict[str, Any]) -> _AtlasArtifact | None:
    result_identity = _file_identity(path)
    if result_identity is None:
        return None
    run_id = path.parent.name
    results_hash_declared = "results_sha256" in metadata
    expected_results_sha256 = _sha256_digest(metadata.get("results_sha256"))
    if results_hash_declared and expected_results_sha256 is None:
        return None
    dashboard_declared = "dashboard" in metadata
    dashboard_identity: tuple[str, int, int, int, int, int] | None = None
    expected_dashboard_sha256: str | None = None
    if dashboard_declared:
        dashboard_value = metadata.get("dashboard")
        if not isinstance(dashboard_value, dict):
            return None
        expected_dashboard_sha256 = _sha256_digest(dashboard_value.get("sha256"))
        dashboard_source_sha256 = _sha256_digest(dashboard_value.get("source_results_sha256"))
        if (
            dashboard_value.get("schema") != ATLAS_DASHBOARD_SCHEMA
            or dashboard_value.get("path") != ATLAS_DASHBOARD_FILENAME
            or expected_dashboard_sha256 is None
            or expected_results_sha256 is None
            or dashboard_source_sha256 != expected_results_sha256
        ):
            return None
        dashboard_identity = _file_identity(path.parent / ATLAS_DASHBOARD_FILENAME)
        if dashboard_identity is None:
            return None
        declared_size = dashboard_value.get("size_bytes")
        if isinstance(declared_size, int) and declared_size != dashboard_identity[3]:
            return None
    cache_root = str(ATLAS_CACHE_ROOT.resolve())
    # functools' cache does not coalesce concurrent misses. Hold the lock around
    # the cached invocation so only one thread can enter a dense legacy parse.
    with _ATLAS_LOAD_LOCK:
        return _read_atlas_artifact_cached(
            result_identity,
            dashboard_identity,
            expected_results_sha256,
            expected_dashboard_sha256,
            run_id,
            cache_root,
        )


@lru_cache(maxsize=128)
def _read_atlas_artifact_cached(
    result_identity: tuple[str, int, int, int, int, int],
    dashboard_identity: tuple[str, int, int, int, int, int] | None,
    expected_results_sha256: str | None,
    expected_dashboard_sha256: str | None,
    run_id: str,
    cache_root: str,
) -> _AtlasArtifact | None:
    result_path = Path(result_identity[0])
    actual_results_sha256 = _stream_sha256(result_path)
    if actual_results_sha256 is None:
        return None
    # A declared result digest is the immutable run commit.  Cache entries are
    # keyed by the bytes we observed, but must never legitimize bytes that no
    # longer match that commit.
    if expected_results_sha256 is not None and actual_results_sha256 != expected_results_sha256:
        return None

    if dashboard_identity is not None:
        # A declared sidecar is part of the immutable commit. Never silently
        # downgrade a missing/tampered sidecar into the legacy dense path.
        if expected_dashboard_sha256 is None:
            return None
        dashboard_path = Path(dashboard_identity[0])
        try:
            encoded = dashboard_path.read_bytes()
            if hashlib.sha256(encoded).hexdigest() != expected_dashboard_sha256:
                return None
            dashboard_raw = json.loads(encoded)
        except OSError, json.JSONDecodeError:
            return None
        payload = validated_atlas_dashboard(
            dashboard_raw,
            expected_results_sha256=actual_results_sha256,
            expected_run_id=run_id,
        )
        if payload is None:
            return None
        return _AtlasArtifact(
            payload=payload,
            results_sha256=actual_results_sha256,
            compact_source="artifact_sidecar",
            dashboard_hash_verified=True,
        )

    cache_path = _atlas_cache_path(Path(cache_root), actual_results_sha256)
    cached_payload = _read_cached_atlas_dashboard(
        cache_path,
        expected_results_sha256=actual_results_sha256,
        expected_run_id=run_id,
    )
    if cached_payload is not None:
        return _AtlasArtifact(
            payload=cached_payload,
            results_sha256=actual_results_sha256,
            compact_source="content_cache",
            dashboard_hash_verified=None,
        )

    raw = _parse_dense_atlas_results(result_path)
    if raw is None:
        return None
    try:
        dashboard = build_atlas_dashboard(raw, actual_results_sha256)
    except ValueError:
        return None
    _write_atlas_cache(cache_path, canonical_json_bytes(dashboard))
    return _AtlasArtifact(
        payload=dashboard["payload"],
        results_sha256=actual_results_sha256,
        compact_source="results_fallback",
        dashboard_hash_verified=None,
    )


def _file_identity(path: Path) -> tuple[str, int, int, int, int, int] | None:
    try:
        stat = path.stat()
        resolved = str(path.resolve())
    except OSError:
        return None
    return (resolved, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _stream_sha256(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _parse_dense_atlas_results(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
    except OSError, json.JSONDecodeError:
        return None
    return raw if isinstance(raw, dict) else None


def _atlas_cache_path(cache_root: Path, results_sha256: str) -> Path:
    return cache_root / ATLAS_DASHBOARD_SCHEMA / results_sha256[:2] / f"{results_sha256}.json"


def _read_cached_atlas_dashboard(
    path: Path,
    *,
    expected_results_sha256: str,
    expected_run_id: str,
) -> dict[str, Any] | None:
    try:
        raw = json.loads(path.read_bytes())
    except OSError, json.JSONDecodeError:
        return None
    return validated_atlas_dashboard(
        raw,
        expected_results_sha256=expected_results_sha256,
        expected_run_id=expected_run_id,
    )


def _write_atlas_cache(path: Path, content: bytes) -> None:
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _sha256_digest(value: Any) -> str | None:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        return None
    try:
        int(value, 16)
    except ValueError:
        return None
    return value


def _read_result(path: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads(path.read_text())
    except OSError, json.JSONDecodeError:
        return None
    if isinstance(raw, list):
        return {
            "schema_version": 0,
            "rows": [row for row in raw if isinstance(row, dict)],
            "matrix_cases": [],
            "aggregates": [],
        }
    if not isinstance(raw, dict):
        return None
    if raw.get("schema") == ATLAS_STUDY_SCHEMA:
        return None
    rows = raw.get("rows", [])
    cases = raw.get("matrix_cases", [])
    aggregates = raw.get("aggregates", [])
    if not isinstance(rows, list) or not isinstance(cases, list) or not isinstance(aggregates, list):
        return None
    try:
        schema_version = int(raw.get("schema_version", 1))
    except TypeError, ValueError:
        schema_version = 1
    return {
        "schema_version": schema_version,
        "rows": [row for row in rows if isinstance(row, dict)],
        "matrix_cases": [case for case in cases if isinstance(case, dict)],
        "aggregates": [aggregate for aggregate in aggregates if isinstance(aggregate, dict)],
    }


def _compatibility_by_name(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("name")): row["gt_dem_compatibility"]
        for row in rows
        if row.get("name") is not None and isinstance(row.get("gt_dem_compatibility"), dict)
    }


def _compat_tier(row: dict[str, Any]) -> str | None:
    compatibility = row.get("gt_dem_compatibility")
    return str(compatibility.get("tier")) if isinstance(compatibility, dict) and compatibility.get("tier") else None


def _height_tier(row: dict[str, Any]) -> str | None:
    compatibility = row.get("gt_dem_compatibility")
    return _height_tier_from_compat(compatibility if isinstance(compatibility, dict) else {})


def _height_tier_from_compat(compatibility: dict[str, Any]) -> str | None:
    height = compatibility.get("height")
    return str(height.get("tier")) if isinstance(height, dict) and height.get("tier") else None


def _photo_edge_support(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("photo_edge_support", row.get("photo_reproducibility"))
    return value if isinstance(value, dict) else {}


def _case_photo_edge_supported(case: dict[str, Any]) -> bool | None:
    value = case.get("photo_edge_supported", case.get("photo_reproducible"))
    return value if isinstance(value, bool) else None


def _difference(value: Any, baseline: Any) -> float | None:
    left = _json_number(value)
    right = _json_number(baseline)
    return round(left - right, 5) if left is not None and right is not None else None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except OSError, json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _legacy_created_at(run_id: str) -> str:
    stamp = run_id.removesuffix("-geopose-bench").removesuffix("-matrix")
    try:
        return datetime.strptime(stamp, "%Y%m%d-%H%M%S").replace(tzinfo=UTC).isoformat()
    except ValueError:
        return stamp


def _json_number(value: Any) -> float | None:
    try:
        number = float(value)
    except TypeError, ValueError:
        return None
    return number if math.isfinite(number) else None


def _finite(value: Any) -> float:
    number = _json_number(value)
    return number if number is not None else math.inf


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _compatibility_sort(compatibility: dict[str, Any]) -> float:
    for key in ("p90_deg", "chamfer_deg", "proxy_px"):
        value = _json_number(compatibility.get(key))
        if value is not None:
            return value
    return math.inf

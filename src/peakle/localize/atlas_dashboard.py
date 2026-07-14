"""Compact, hash-linked projections of immutable pose-atlas artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

ATLAS_STUDY_SCHEMA = "peakle_pose_atlas_study_v2"
ATLAS_DASHBOARD_SCHEMA = "peakle_pose_atlas_dashboard_v1"
ATLAS_DASHBOARD_FILENAME = "dashboard.json"


def canonical_json_bytes(value: Any) -> bytes:
    """Return the canonical on-disk JSON encoding used by atlas artifacts."""

    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()


def compact_atlas_results(raw: Any) -> dict[str, Any] | None:
    """Project a dense v2 result onto the bounded dashboard contract."""

    if not isinstance(raw, dict) or raw.get("schema") != ATLAS_STUDY_SCHEMA:
        return None
    config = raw.get("config")
    samples = raw.get("samples")
    if (
        not isinstance(config, dict)
        or not isinstance(samples, list)
        or not all(isinstance(sample, dict) for sample in samples)
    ):
        return None
    return {
        "schema": ATLAS_STUDY_SCHEMA,
        "run_id": raw.get("run_id"),
        "config": _json_safe(config),
        "samples": [_compact_sample(sample) for sample in samples],
    }


def build_atlas_dashboard(raw: Any, results_sha256: str) -> dict[str, Any]:
    """Build a deterministic sidecar linked to the exact dense result bytes."""

    if not _is_sha256(results_sha256):
        raise ValueError("results_sha256 must be a lowercase SHA-256 digest")
    payload = compact_atlas_results(raw)
    if payload is None:
        raise ValueError(f"results do not satisfy {ATLAS_STUDY_SCHEMA}")
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("atlas results must contain a non-empty run_id")
    return {
        "schema": ATLAS_DASHBOARD_SCHEMA,
        "run_id": run_id,
        "source_results_schema": ATLAS_STUDY_SCHEMA,
        "source_results_sha256": results_sha256,
        "payload_sha256": hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
        "payload": payload,
    }


def validated_atlas_dashboard(
    raw: Any,
    *,
    expected_results_sha256: str,
    expected_run_id: str,
) -> dict[str, Any] | None:
    """Validate a sidecar's complete hash/run/schema chain and return its payload."""

    if (
        not isinstance(raw, dict)
        or raw.get("schema") != ATLAS_DASHBOARD_SCHEMA
        or raw.get("run_id") != expected_run_id
        or raw.get("source_results_schema") != ATLAS_STUDY_SCHEMA
        or raw.get("source_results_sha256") != expected_results_sha256
    ):
        return None
    payload = raw.get("payload")
    if not _is_compact_payload(payload, expected_run_id=expected_run_id):
        return None
    expected_payload_hash = raw.get("payload_sha256")
    if not _is_sha256(expected_payload_hash):
        return None
    actual_payload_hash = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return payload if actual_payload_hash == expected_payload_hash else None


def _is_compact_payload(value: Any, *, expected_run_id: str) -> bool:
    if not isinstance(value, dict):
        return False
    samples = value.get("samples")
    return (
        value.get("schema") == ATLAS_STUDY_SCHEMA
        and value.get("run_id") == expected_run_id
        and isinstance(value.get("config"), dict)
        and isinstance(samples, list)
        and all(isinstance(sample, dict) for sample in samples)
    )


def _compact_sample(sample: dict[str, Any]) -> dict[str, Any]:
    prior_value = sample.get("prior")
    prior = prior_value if isinstance(prior_value, dict) else {}
    tracks_value = sample.get("tracks")
    tracks = tracks_value if isinstance(tracks_value, dict) else {}
    return {
        "name": str(sample.get("name", "unknown")),
        "manual": bool(sample.get("manual")),
        "compatibility": _json_safe(sample.get("compatibility")),
        "photo_edge_support": _json_safe(sample.get("photo_edge_support")),
        "prior_errors": _json_safe(prior.get("errors")),
        "prior_context": _json_safe(
            {
                "regime": prior.get("regime"),
                "constructed_from_reference_for_controlled_perturbation": prior.get(
                    "constructed_from_reference_for_controlled_perturbation"
                ),
                "atlas_constraints": "east_north_grid_center_only",
                "recorded_yaw_pitch_altitude_used_by_atlas": False,
            }
        ),
        "tracks": {str(track): _compact_track(record) for track, record in tracks.items() if isinstance(record, dict)},
    }


def _compact_track(record: dict[str, Any]) -> dict[str, Any]:
    evaluation_value = record.get("evaluation")
    evaluation = evaluation_value if isinstance(evaluation_value, dict) else None
    return {
        "status": str(record.get("status", "missing")),
        "runtime_s": _json_number(record.get("runtime_s")),
        "evidence": _json_safe(record.get("evidence")),
        "archive": _compact_archive(record.get("estimator_archive")),
        "evaluation": _compact_evaluation(evaluation) if evaluation is not None else None,
    }


def _compact_archive(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    lattice_value = value.get("full_score_lattice")
    lattice = lattice_value if isinstance(lattice_value, dict) else {}
    return _json_safe(
        {
            "archive_sha256": value.get("archive_sha256"),
            "candidate_count": value.get("candidate_count"),
            "selected_candidate_id": value.get("selected_candidate_id"),
            "grid": value.get("grid"),
            "query_geometry": value.get("query_geometry"),
            "native_patch_supplied": value.get("native_patch_supplied"),
            "full_lattice": {
                "yaw_count": lattice.get("yaw_count"),
                "hypothesis_count": lattice.get("hypothesis_count"),
            },
        }
    )


def _compact_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    top_k = evaluation.get("shortlist_top_k")
    top_k_records = top_k if isinstance(top_k, list) else []
    reference_value = evaluation.get("evaluation_only_reference_position_probe")
    reference = reference_value if isinstance(reference_value, dict) else None
    return {
        "archive_sha256": evaluation.get("archive_sha256"),
        "reference_data_used": evaluation.get("reference_data_used"),
        "used_by_estimator": evaluation.get("used_by_estimator"),
        "target": _json_safe(evaluation.get("target")),
        "blind_winner": _compact_candidate(evaluation.get("winner_errors")),
        "full_lattice_oracle": _compact_candidate(evaluation.get("full_lattice_gt_oracle")),
        "selection_regret": _json_safe(evaluation.get("selection_regret")),
        "shortlist_top_k": [
            _json_safe(
                {
                    "candidate_pool": item.get("candidate_pool"),
                    "requested_k": item.get("requested_k"),
                    "evaluated_k": item.get("evaluated_k"),
                    "reaches_target": item.get("reaches_target"),
                    "recall": item.get("recall"),
                }
            )
            for item in top_k_records
            if isinstance(item, dict)
        ],
        "reference_position_probe": (
            _json_safe(
                {
                    "used_by_estimator": reference.get("used_by_estimator"),
                    "included_in_estimator_archive": reference.get("included_in_estimator_archive"),
                    "errors": reference.get("errors"),
                    "score_delta_reference_minus_blind_winner": reference.get(
                        "score_delta_reference_minus_blind_winner"
                    ),
                }
            )
            if reference is not None
            else None
        ),
    }


def _compact_candidate(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    hypothesis_value = value.get("hypothesis")
    hypothesis = hypothesis_value if isinstance(hypothesis_value, dict) else {}
    return _json_safe(
        {
            "candidate_id": value.get("candidate_id"),
            "estimator_rank": value.get("estimator_rank"),
            "estimator_rank_scope": value.get("estimator_rank_scope"),
            "estimator_score": hypothesis.get("estimator_score"),
            "errors": value.get("errors"),
            "normalized_joint_error": value.get("normalized_joint_error"),
            "reaches_target": value.get("reaches_target"),
        }
    )


def _json_number(value: Any) -> float | None:
    try:
        number = float(value)
    except TypeError, ValueError:
        return None
    return number if math.isfinite(number) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True

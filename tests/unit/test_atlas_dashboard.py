"""Compact pose-atlas dashboard projection contracts."""

from __future__ import annotations

import copy
import hashlib

from peakle.localize.atlas_dashboard import (
    ATLAS_DASHBOARD_SCHEMA,
    ATLAS_STUDY_SCHEMA,
    build_atlas_dashboard,
    canonical_json_bytes,
    validated_atlas_dashboard,
)


def _dense_results() -> dict:
    candidate = {
        "candidate_id": "candidate-1",
        "estimator_rank": 1,
        "estimator_rank_scope": "full_score_lattice",
        "hypothesis": {"estimator_score": 0.25, "pose": {"yaw_deg": 12.0}},
        "errors": {"horizontal_position_m": 20.0, "yaw_deg": 1.0, "pitch_deg": None},
        "normalized_joint_error": 0.2,
        "reaches_target": True,
    }
    return {
        "schema": ATLAS_STUDY_SCHEMA,
        "run_id": "atlas-control",
        "config": {"tracks": ["pfm_oracle"]},
        "samples": [
            {
                "name": "synthetic-control",
                "manual": False,
                "prior": {
                    "regime": "perturbed_metadata",
                    "constructed_from_reference_for_controlled_perturbation": True,
                    "errors": {"horizontal_position_m": 200.0, "yaw_deg": 15.0},
                },
                "tracks": {
                    "pfm_oracle": {
                        "status": "ok",
                        "runtime_s": 2.0,
                        "estimator_archive": {
                            "candidate_count": 3,
                            "full_score_lattice": {
                                "yaw_count": 360,
                                "hypothesis_count": 1080,
                                "positions": [{"yaw_scores": [0.1, 0.2]}],
                            },
                            "candidates": [{"private_candidate": True}],
                        },
                        "evaluation": {
                            "reference_data_used": True,
                            "used_by_estimator": False,
                            "winner_errors": candidate,
                            "full_lattice_gt_oracle": candidate,
                            "shortlist_top_k": [],
                            "evaluation_only_reference_position_probe": {
                                "used_by_estimator": False,
                                "included_in_estimator_archive": False,
                                "errors": candidate["errors"],
                                "score_delta_reference_minus_blind_winner": 0.5,
                                "private_mode": True,
                            },
                        },
                    }
                },
            }
        ],
    }


def test_dashboard_projection_is_bounded_and_keeps_truth_audit_labels() -> None:
    results = _dense_results()
    results_sha256 = hashlib.sha256(canonical_json_bytes(results)).hexdigest()

    dashboard = build_atlas_dashboard(results, results_sha256)
    payload = validated_atlas_dashboard(
        dashboard,
        expected_results_sha256=results_sha256,
        expected_run_id="atlas-control",
    )

    assert payload is not None
    encoded = canonical_json_bytes(payload)
    assert b"yaw_scores" not in encoded
    assert b"private_candidate" not in encoded
    assert b"private_mode" not in encoded
    sample = payload["samples"][0]
    assert sample["prior_context"] == {
        "regime": "perturbed_metadata",
        "constructed_from_reference_for_controlled_perturbation": True,
        "atlas_constraints": "east_north_grid_center_only",
        "recorded_yaw_pitch_altitude_used_by_atlas": False,
    }
    evaluation = sample["tracks"]["pfm_oracle"]["evaluation"]
    assert evaluation["used_by_estimator"] is False
    assert evaluation["blind_winner"]["candidate_id"] == "candidate-1"
    assert evaluation["full_lattice_oracle"]["candidate_id"] == "candidate-1"
    assert evaluation["reference_position_probe"]["included_in_estimator_archive"] is False


def test_dashboard_validation_rejects_every_broken_hash_link() -> None:
    results = _dense_results()
    results_sha256 = hashlib.sha256(canonical_json_bytes(results)).hexdigest()
    dashboard = build_atlas_dashboard(results, results_sha256)

    broken_source = copy.deepcopy(dashboard)
    broken_source["source_results_sha256"] = "0" * 64
    broken_payload = copy.deepcopy(dashboard)
    broken_payload["payload"]["samples"][0]["name"] = "tampered"

    assert dashboard["schema"] == ATLAS_DASHBOARD_SCHEMA
    assert (
        validated_atlas_dashboard(
            broken_source,
            expected_results_sha256=results_sha256,
            expected_run_id="atlas-control",
        )
        is None
    )
    assert (
        validated_atlas_dashboard(
            broken_payload,
            expected_results_sha256=results_sha256,
            expected_run_id="atlas-control",
        )
        is None
    )
    assert (
        validated_atlas_dashboard(
            dashboard,
            expected_results_sha256=results_sha256,
            expected_run_id="different-run",
        )
        is None
    )

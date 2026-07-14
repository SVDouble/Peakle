from __future__ import annotations

import hashlib
import json

import pytest

from peakle.scripts import bench_pose_atlas_pfm_geometry as module
from peakle.scripts.bench_pose_atlas import ATLAS_STUDY_SCHEMA


def _sample(name: str = "alpha") -> dict:
    return {
        "name": name,
        "reference": {
            "position": {"east_m": 0.0, "north_m": 0.0, "up_m": 1000.0},
            "yaw_deg": 0.0,
            "pitch_deg": 0.0,
            "roll_deg": 0.0,
        },
        "coordinate_frame_origin": {"latitude_deg": 46.0, "longitude_deg": 8.0, "elevation_m": 500.0},
        "prior": {
            "position": {"east_m": 200.0, "north_m": 0.0, "up_m": 1002.5},
            "yaw_deg": 15.0,
            "pitch_deg": 8.0,
            "horizontal_sigma_m": 200.0,
            "vertical_sigma_m": 75.0,
            "yaw_sigma_deg": 15.0,
            "pitch_sigma_deg": 8.0,
            "regime": "perturbed_metadata",
            "constructed_from_reference_for_controlled_perturbation": True,
            "perturbation": {
                "bucket": "standard",
                "replicate": 0,
                "seed": 123,
                "realized": {
                    "east_m": -18.482648,
                    "north_m": 199.144148,
                    "up_m": 75.0,
                    "yaw_deg": -15.0,
                },
                "requested": {
                    "east_m": -18.482648,
                    "north_m": 199.144148,
                    "up_m": 75.0,
                    "yaw_deg": -15.0,
                },
            },
            "errors": {"horizontal_position_m": 200.0},
        },
        "terrain_inputs": {
            "base_source": "Copernicus GLO-30",
            "uses_reference_truth": False,
            "native_patch": {"prior_contains_exact_reference": False},
        },
        "tracks": {
            "pfm_oracle": {
                "status": "ok",
                "estimator_archive": {"schema": "peakle_skyline_atlas_archive_v2"},
                "evaluation": {"winner_errors": {"errors": {"horizontal_position_m": 500.0}}},
            },
            "photo_auto": {"status": "evidence_rejected", "estimator_archive": None},
        },
    }


def test_estimator_whitelist_excludes_numeric_reference_and_evaluation() -> None:
    specs = module._estimator_specs([_sample()], "pfm_oracle")
    encoded = json.dumps(specs, sort_keys=True)

    assert '"reference"' not in encoded
    assert '"errors"' not in encoded
    assert '"evaluation"' not in encoded
    assert '"perturbation"' not in encoded
    assert '"realized"' not in encoded
    assert '"requested"' not in encoded
    assert "-18.482648" not in encoded
    assert specs[0]["prior"]["position"]["east_m"] == 200.0
    assert specs[0]["prior"]["terrain_cache_replicate"] == 0
    assert specs[0]["estimator_archive"]["schema"] == "peakle_skyline_atlas_archive_v2"


def test_sample_selection_requires_completed_candidate_track() -> None:
    results = {"samples": [_sample("alpha"), _sample("beta")]}

    selected = module._selected_records(results, "beta,alpha", "pfm_oracle")
    assert [record["name"] for record in selected] == ["beta", "alpha"]
    with pytest.raises(SystemExit, match="duplicate"):
        module._selected_records(results, "alpha,alpha", "pfm_oracle")
    with pytest.raises(SystemExit, match="unavailable"):
        module._selected_records(results, "alpha", "photo_auto")


def test_source_artifact_digest_and_schema_are_enforced() -> None:
    results = {"schema": ATLAS_STUDY_SCHEMA, "samples": []}
    content = module._json_bytes(results)
    run = {
        "schema": ATLAS_STUDY_SCHEMA,
        "status": "complete",
        "results_sha256": hashlib.sha256(content).hexdigest(),
    }

    module._validate_source_artifact(results, run, content)
    with pytest.raises(SystemExit, match="does not match"):
        module._validate_source_artifact(results, run, content + b" ")


def test_cli_requires_positive_subsample() -> None:
    with pytest.raises(SystemExit):
        module._parser().parse_args(["--atlas", "a.json", "--subsample", "0", "--output", "out"])


def test_aggregates_keep_fixed_component_winners_separate() -> None:
    def evaluated(horizontal_m: float, yaw_deg: float, reaches_target: bool) -> dict:
        return {
            "errors": {"horizontal_position_m": horizontal_m, "yaw_deg": yaw_deg},
            "reaches_target": reaches_target,
        }

    samples = [
        {
            "evaluation": {
                "winner_errors": evaluated(20.0, 1.0, True),
                "candidate_pool_gt_oracle": evaluated(10.0, 0.5, True),
                "component_winner_errors": {
                    "fusion": evaluated(20.0, 1.0, True),
                    "skyline": evaluated(300.0, 2.0, False),
                },
            }
        },
        {
            "evaluation": {
                "winner_errors": evaluated(200.0, 3.0, False),
                "candidate_pool_gt_oracle": evaluated(15.0, 0.25, True),
                "component_winner_errors": {
                    "fusion": evaluated(200.0, 3.0, False),
                    "skyline": evaluated(400.0, 4.0, False),
                },
            }
        },
    ]

    aggregates = module._aggregates(samples)

    assert aggregates["component_winners"]["fusion"] == {
        "samples": 2,
        "successes": 1,
        "median_horizontal_m": 110.0,
        "median_yaw_deg": 2.0,
    }
    assert aggregates["component_winners"]["skyline"]["successes"] == 0

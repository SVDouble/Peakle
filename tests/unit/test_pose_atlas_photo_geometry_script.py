from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from peakle.scripts import bench_pose_atlas_photo_geometry as module
from peakle.scripts.bench_pose_atlas import ATLAS_STUDY_SCHEMA


def _sample(name: str = "alpha") -> dict:
    return {
        "name": name,
        "manual": True,
        "reference": {
            "position": {"east_m": 0.0, "north_m": 0.0, "up_m": 1000.0},
            "yaw_deg": 0.0,
            "pitch_deg": 0.0,
            "roll_deg": 0.0,
        },
        "reference_source": "truth metadata",
        "coordinate_frame_origin": {"latitude_deg": 46.0, "longitude_deg": 8.0, "elevation_m": 500.0},
        "compatibility": {
            "policy": "gt_dem_compat_v1",
            "tier": "MAP_B",
            "median_deg": 0.2,
            "p90_deg": 0.8,
            "coverage": 1.0,
            "terrain_inputs": {"evaluation_reference_patch": {"center_local_m": {"east": -12.0}}},
            "height": {
                "tier": "HEIGHT_A",
                "physically_plausible": True,
                "raw_camera_clearance_m": 2.5,
            },
        },
        "photo_edge_support": {"reference_support": 0.8},
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
                "replicate": 3,
                "realized": {"east_m": -18.482648, "north_m": 199.144148},
                "requested": {"east_m": -18.482648, "north_m": 199.144148},
            },
            "errors": {"horizontal_position_m": 200.0},
        },
        "terrain_inputs": {
            "base_source": "Copernicus GLO-30",
            "uses_reference_truth": False,
            "native_patch": {
                "source": "cached swissALTI3D 2 m",
                "center_source": "supplied_position_prior",
                "center_local_m": {"east": 200.0, "north": 0.0},
                "center_geo_deg": {"latitude": 46.0, "longitude": 8.0},
                "network_allowed": False,
                "radius_m": 4500.0,
                "status": "available",
                "coverage": {"available": True, "shape": [2, 2]},
                "used_by_estimator": True,
                "uses_reference_truth": False,
                "prior_constructed_from_reference": True,
                "prior_contains_exact_reference": False,
            },
            "regular_grid_fused_cells": 4,
            "regular_grid_is_per_cell_copy": True,
        },
        "tracks": {
            "pfm_oracle": {
                "status": "ok",
                "evidence": {"source": "alpha/cyl/distance_crop.pfm"},
                "evaluation": {"winner_errors": {"horizontal_position_m": 1.0}},
            },
            "photo_auto": {
                "status": "ok",
                "evidence": {
                    "source": "color",
                    "available": True,
                    "candidate": "color",
                    "coverage": 0.75,
                    "agreement": 0.25,
                    "detected_candidates": ["blue", "color"],
                    "selection_uses_reference_truth": False,
                    "evidence_generated_at_reference_pose": False,
                },
                "estimator_archive": {
                    "schema": "peakle_skyline_atlas_archive_v2",
                    "numeric_evaluation_reference_used": False,
                    "archive_sha256": "a" * 64,
                    "query_geometry": {
                        "width_px": 8,
                        "height_px": 4,
                        "observed_skyline_sha256": "b" * 64,
                    },
                },
                "evaluation": {"winner_errors": {"errors": {"horizontal_position_m": 500.0}}},
            },
        },
    }


def test_estimator_whitelist_is_photo_only_and_drops_signed_truth_leaks() -> None:
    spec = module._estimator_specs([_sample()])[0]
    encoded = json.dumps(spec, sort_keys=True).lower()

    assert '"reference":' not in encoded
    assert '"evaluation":' not in encoded
    assert '"errors":' not in encoded
    assert '"compatibility":' not in encoded
    assert '"photo_edge_support":' not in encoded
    assert '"pfm_oracle":' not in encoded
    assert '"perturbation":' not in encoded
    assert "distance_crop.pfm" not in encoded
    assert "-18.482648" not in encoded
    assert spec["prior"]["terrain_cache_replicate"] == 3
    assert spec["photo_evidence"]["candidate"] == "color"
    assert "uses_reference_truth" not in spec["terrain_expected_identity"]["native_patch"]
    assert "prior_constructed_from_reference" not in spec["terrain_expected_identity"]["native_patch"]


def test_estimator_phase_guard_rejects_late_truth_or_pfm_regressions() -> None:
    clean = module._estimator_specs([_sample()])
    module._assert_estimator_phase_inputs(clean)

    with pytest.raises(ValueError, match="crossed the estimator whitelist"):
        module._assert_estimator_phase_inputs([{"name": "alpha", "evaluation": {"score": 1.0}}])
    with pytest.raises(ValueError, match="crossed the estimator whitelist"):
        module._assert_estimator_phase_inputs([{"name": "alpha", "input": "cyl/distance_crop.pfm"}])


def test_whitelist_requires_source_photo_truth_blind_attestations() -> None:
    sample = _sample()
    sample["tracks"]["photo_auto"]["evidence"]["selection_uses_reference_truth"] = True
    with pytest.raises(ValueError, match="truth-blind"):
        module._estimator_specs([sample])

    sample = _sample()
    sample["tracks"]["photo_auto"]["evidence"]["evidence_generated_at_reference_pose"] = True
    with pytest.raises(ValueError, match="reference pose"):
        module._estimator_specs([sample])


def test_selection_requires_a_completed_photo_auto_track() -> None:
    rejected = _sample("rejected")
    rejected["tracks"]["photo_auto"]["status"] = "evidence_rejected"
    results = {"samples": [_sample("alpha"), rejected]}

    assert module._selected_records(results, "alpha")[0]["name"] == "alpha"
    with pytest.raises(SystemExit, match="duplicate"):
        module._selected_records(results, "alpha,alpha")
    with pytest.raises(SystemExit, match="unavailable"):
        module._selected_records(results, "rejected")


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


def test_source_photo_is_resized_to_the_exact_frozen_query(tmp_path: Path) -> None:
    path = tmp_path / "photo.jpg"
    Image.fromarray(np.arange(8 * 6 * 3, dtype=np.uint8).reshape(6, 8, 3), mode="RGB").save(path)

    loaded = module._load_photo_for_query(path, {"width_px": 5, "height_px": 3})

    with Image.open(path) as source:
        expected = np.asarray(source.convert("RGB").resize((5, 3), Image.Resampling.BILINEAR), np.uint8)
    assert loaded.shape == (3, 5, 3)
    np.testing.assert_array_equal(loaded, expected)


def test_photo_input_discovery_does_not_require_reference_depth(tmp_path: Path) -> None:
    sample = tmp_path / "photo-only"
    (sample / "cyl").mkdir(parents=True)
    (sample / "cyl" / "photo_crop.jpg").write_bytes(b"rgb input")

    assert module._find_photo_sample_dirs(tmp_path) == [sample]


def test_reconstructed_skyline_must_match_selection_metadata_and_atlas_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = np.asarray([1.0, np.nan, 2.0, 3.0])
    candidate = SimpleNamespace(rows=rows, coverage=0.75, agreement=0.25)
    candidates = {"blue": object(), "color": candidate}
    monkeypatch.setattr(module, "extract_candidates", lambda *_args, **_kwargs: candidates)
    monkeypatch.setattr(module, "best_skyline_candidate", lambda *_args, **_kwargs: ("color", candidate))
    evidence = {
        "source": "color",
        "candidate": "color",
        "coverage": 0.75,
        "agreement": 0.25,
        "detected_candidates": ["blue", "color"],
    }
    atlas = {"query_geometry": {"observed_skyline_sha256": module._observed_skyline_sha256(rows)}}

    reconstructed, record = module._reconstruct_photo_skyline(np.zeros((3, 4, 3), dtype=np.uint8), evidence, atlas)

    np.testing.assert_array_equal(reconstructed, rows)
    assert record["matches_source_atlas"] is True
    atlas["query_geometry"]["observed_skyline_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="frozen atlas hash"):
        module._reconstruct_photo_skyline(np.zeros((3, 4, 3), dtype=np.uint8), evidence, atlas)


def test_model_files_are_content_addressed_including_cached_symlinks(tmp_path: Path) -> None:
    checkpoint = tmp_path / "dexined.pth"
    checkpoint.write_bytes(b"edge weights")
    edge = module._file_model_provenance("dexined", checkpoint, kind="checkpoint")
    assert edge["files"][0]["sha256"] == hashlib.sha256(b"edge weights").hexdigest()

    model_dir = tmp_path / "depth"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    (model_dir / "model.safetensors").write_bytes(b"depth weights")
    before = module._directory_model_provenance("depth-anything", model_dir)
    (model_dir / "model.safetensors").write_bytes(b"changed weights")
    after = module._directory_model_provenance("depth-anything", model_dir)
    assert before["aggregate_sha256"] != after["aggregate_sha256"]

    (model_dir / "linked.bin").symlink_to(model_dir / "model.safetensors")
    linked = module._directory_model_provenance("depth-anything", model_dir)
    linked_record = next(record for record in linked["files"] if record["path"] == "linked.bin")
    assert linked_record["sha256"] == hashlib.sha256(b"changed weights").hexdigest()
    assert linked_record["symlink_target"] == str(model_dir / "model.safetensors")


def test_offline_context_blocks_sockets_and_restores_environment() -> None:
    before = os.environ.get("HF_HUB_OFFLINE")
    original_create_connection = socket.create_connection

    with module._offline_inference_environment():
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        with pytest.raises(RuntimeError, match="network access"):
            socket.create_connection(("example.invalid", 443))

    assert socket.create_connection is original_create_connection
    assert os.environ.get("HF_HUB_OFFLINE") == before


def test_state_dict_accepts_plain_or_wrapped_checkpoints_and_strips_data_parallel_prefix() -> None:
    assert module._state_dict({"module.layer": 1}) == {"layer": 1}
    assert module._state_dict({"state_dict": {"module.layer": 2}}) == {"layer": 2}
    with pytest.raises(ValueError, match="mapping"):
        module._state_dict([])


def test_aggregates_separate_ranking_selection_beam_and_abstention() -> None:
    def evaluated(horizontal: float, yaw: float, success: bool) -> dict:
        return {
            "errors": {"horizontal_position_m": horizontal, "yaw_deg": yaw},
            "reaches_target": success,
        }

    samples = [
        {
            "evaluation_only_compatibility": {"fit_bucket": "MAP_A"},
            "evaluation": {
                "ranked_winner_errors": evaluated(20.0, 1.0, True),
                "returned_candidate_errors": evaluated(20.0, 1.0, True),
                "candidate_pool_gt_oracle": evaluated(10.0, 0.5, True),
                "first_target_beam_rank": 1,
                "component_winner_errors": {"fusion": evaluated(20.0, 1.0, True)},
            },
        },
        {
            "evaluation_only_compatibility": {"fit_bucket": "MAP_B"},
            "evaluation": {
                "ranked_winner_errors": evaluated(200.0, 6.0, False),
                "returned_candidate_errors": evaluated(200.0, 6.0, False),
                "candidate_pool_gt_oracle": evaluated(15.0, 0.25, True),
                "first_target_beam_rank": 4,
                "component_winner_errors": {"fusion": evaluated(200.0, 6.0, False)},
            },
        },
    ]

    aggregate = module._aggregates(samples)

    assert aggregate["ranked_winner_successes"] == 1
    assert aggregate["selected_decisions"] == 2
    assert aggregate["abstentions"] == 0
    assert aggregate["returned_candidate_successes"] == 1
    assert aggregate["returned_candidate_false_accepts"] == 1
    assert aggregate["beam_target_successes"] == 2
    assert aggregate["component_winners"]["fusion"]["median_horizontal_m"] == 110.0
    assert aggregate["by_fit_bucket"]["MAP_A"]["returned_candidate_false_accepts"] == 0
    assert aggregate["by_fit_bucket"]["MAP_B"]["returned_candidate_false_accepts"] == 1


def test_evaluation_compatibility_is_compact_and_never_copies_reference_patch() -> None:
    record = module._evaluation_compatibility_record(_sample())
    encoded = json.dumps(record, sort_keys=True)

    assert record["fit_bucket"] == "MAP_B"
    assert record["metrics"]["median_deg"] == 0.2
    assert record["height"]["tier"] == "HEIGHT_A"
    assert record["used_by_estimator"] is False
    assert "terrain_inputs" not in encoded
    assert "center_local_m" not in encoded


def test_cli_requires_explicit_models_and_positive_subsample() -> None:
    with pytest.raises(SystemExit):
        module._parser().parse_args(["--atlas", "a.json", "--output", "out"])
    with pytest.raises(SystemExit):
        module._parser().parse_args(
            [
                "--atlas",
                "a.json",
                "--dexined-checkpoint",
                "edge.pth",
                "--depth-model-dir",
                "depth",
                "--subsample",
                "0",
                "--output",
                "out",
            ]
        )


def test_main_durably_freezes_estimator_before_truth_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_dir = tmp_path / "corpus" / "alpha"
    (sample_dir / "cyl").mkdir(parents=True)
    photo = sample_dir / "cyl" / "photo_crop.jpg"
    photo.write_bytes(b"source photo")
    atlas_dir = tmp_path / "atlas"
    atlas_dir.mkdir()
    results = {
        "schema": ATLAS_STUDY_SCHEMA,
        "config": {"terrain": {"extent_m": 40_000.0, "grid": 1335}},
        "samples": [_sample()],
    }
    results_bytes = module._json_bytes(results)
    atlas_path = atlas_dir / "results.json"
    atlas_path.write_bytes(results_bytes)
    empty_cache_sha = hashlib.sha256(module._json_bytes({})).hexdigest()
    run = {
        "schema": ATLAS_STUDY_SCHEMA,
        "status": "complete",
        "results_sha256": hashlib.sha256(results_bytes).hexdigest(),
        "inputs": {
            "files": [
                {
                    "path": "alpha/cyl/photo_crop.jpg",
                    "sha256": hashlib.sha256(b"source photo").hexdigest(),
                }
            ]
        },
        "terrain_cache": {"aggregate_sha256": empty_cache_sha},
    }
    (atlas_dir / "run.json").write_bytes(module._json_bytes(run))
    edge_path = tmp_path / "edge.pth"
    edge_path.write_bytes(b"edge")
    depth_dir = tmp_path / "depth"
    depth_dir.mkdir()
    (depth_dir / "config.json").write_text("{}")
    (depth_dir / "model.safetensors").write_bytes(b"depth")
    output = tmp_path / "result"

    monkeypatch.setattr(module, "_find_photo_sample_dirs", lambda: [sample_dir])
    monkeypatch.setattr(module, "default_terrain_cache_inventory", lambda: {})
    monkeypatch.setattr(module, "_implementation_record", lambda: {"aggregate_sha256": "i" * 64})
    monkeypatch.setattr(module, "_verify_stability", lambda **_kwargs: None)
    monkeypatch.setattr(module, "_load_dexined", lambda *_args: object())
    monkeypatch.setattr(module, "_load_depth_anything", lambda *_args: object())

    def run_estimators(specs, **_kwargs):
        module._assert_estimator_phase_inputs(specs)
        return ([{"name": "alpha", "verifier_archive": {"truth_free": True}}], {"alpha": object()})

    monkeypatch.setattr(module, "_run_estimators", run_estimators)
    events: list[tuple[str, Path]] = []
    original_write = module._write_once
    original_fsync = module._fsync_directory

    def write_once(path: Path, content: bytes) -> None:
        original_write(path, content)
        events.append(("write", path))

    def fsync_directory(path: Path) -> None:
        original_fsync(path)
        events.append(("fsync", path))

    monkeypatch.setattr(module, "_write_once", write_once)
    monkeypatch.setattr(module, "_fsync_directory", fsync_directory)

    def evaluate(samples, _archives):
        assert samples[0]["reference"]["position"]["up_m"] == 1000.0
        verifier_write = next(path for event, path in events if event == "write" and path.name == module.ESTIMATOR_FILE)
        assert ("fsync", verifier_write.parent) in events
        frozen = json.loads(verifier_write.read_text())
        assert frozen["numeric_evaluation_reference_used"] is False
        assert "compatibility" not in json.dumps(frozen, sort_keys=True)
        evaluated = {
            "errors": {"horizontal_position_m": 20.0, "yaw_deg": 1.0},
            "reaches_target": True,
            "original_estimator_rank": 1,
        }
        return [
            {
                "name": "alpha",
                "atlas_skyline_winner": None,
                "evaluation_only_compatibility": {"fit_bucket": "MAP_B"},
                "evaluation": {
                    "ranked_winner_errors": evaluated,
                    "returned_candidate_errors": None,
                    "candidate_pool_gt_oracle": evaluated,
                    "first_target_beam_rank": 1,
                    "component_winner_errors": {},
                },
            }
        ]

    monkeypatch.setattr(module, "_evaluate_frozen_archives", evaluate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench",
            "--atlas",
            str(atlas_path),
            "--dexined-checkpoint",
            str(edge_path),
            "--depth-model-dir",
            str(depth_dir),
            "--output",
            str(output),
        ],
    )

    module.main()

    assert (output / module.ESTIMATOR_FILE).is_file()
    assert json.loads((output / "run.json").read_text())["status"] == "complete"

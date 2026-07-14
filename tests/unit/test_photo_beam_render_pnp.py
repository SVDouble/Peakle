from __future__ import annotations

import inspect
from copy import deepcopy
from typing import Any

import pytest

from peakle.localize import photo_beam_render_pnp as module
from peakle.localize.photo_beam_render_pnp import build_photo_beam_render_seed_bridge


def _atlas_candidate(
    candidate_id: str,
    rank: int,
    *,
    east_m: float,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
) -> dict[str, Any]:
    return {
        "schema": "peakle_skyline_atlas_candidate_v2",
        "candidate_id": candidate_id,
        "estimator_rank": rank,
        "grid": {"row": 0, "column": rank - 1, "yaw_index": rank - 1},
        "pose": {
            "position": {"east_m": east_m, "north_m": -500.0 + rank, "up_m": 1500.0 + rank},
            "yaw_deg": yaw_deg,
            "pitch_deg": pitch_deg,
            "roll_deg": roll_deg,
        },
        "vertical_shift_px": -4.0,
        "vertical_slope_px_per_column": roll_deg,
        "estimator_score": float(rank),
        "ground_source": "terrain_map",
    }


def _photo_candidate(
    candidate_id: str,
    verifier_rank: int,
    *,
    beam_rank: int | None,
    east_m: float,
    yaw_deg: float,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> dict[str, Any]:
    return {
        "schema": "peakle_photo_geometry_verifier_candidate_v1",
        "candidate_id": candidate_id,
        "original_estimator_rank": verifier_rank,
        "verifier_rank": verifier_rank,
        "beam_rank": beam_rank,
        "atlas_candidate": _atlas_candidate(
            candidate_id,
            verifier_rank,
            east_m=east_m,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
        ),
        "terms": {
            "skyline_loss": 0.1,
            "symmetric_outline_loss": 0.1,
            "photo_to_dem_outline_loss": 0.1,
            "dem_to_photo_outline_loss": 0.1,
            "ordinal_depth_loss": 0.1,
            "terrain_overlap_loss": 0.1,
            "terrain_overlap_f1": 0.9,
            "common_depth_px": 200,
            "common_terrain_fraction": 0.8,
            "photo_terrain_px": 250,
            "candidate_terrain_px": 250,
            "ordinal_cells": 32,
            "candidate_outline_px": 40,
            "families": {
                family: {"candidate_px": 10, "dem_to_photo_loss": 0.1} for family in ("occlusion", "rib", "couloir")
            },
        },
        "fusion_score": verifier_rank / 10.0,
        "fold_fusion_scores": [0.1, 0.2],
    }


def _archive() -> dict[str, Any]:
    atlas_sha = "a" * 64
    candidates = [
        _photo_candidate("winner", 1, beam_rank=1, east_m=10.0, yaw_deg=20.0),
        _photo_candidate("outside-beam", 2, beam_rank=None, east_m=20.0, yaw_deg=30.0),
        _photo_candidate("second-beam", 3, beam_rank=2, east_m=30.0, yaw_deg=40.0),
        _photo_candidate(
            "useful-last",
            4,
            beam_rank=3,
            east_m=40.0,
            yaw_deg=50.0,
            pitch_deg=7.5,
            roll_deg=13.0,
        ),
    ]
    observation = {
        "track": "photo_auto",
        "source": "dexined_depth_anything",
        "candidate": "selected_photo_observation",
        "selection_uses_reference_truth": False,
        "evidence_generated_at_reference_pose": False,
        "source_atlas_sha256": atlas_sha,
    }
    record = {
        "schema": "peakle_photo_geometry_verifier_archive_v1",
        "experimental": True,
        "production_eligible": False,
        "photo_observable_evidence_only": True,
        "source_depth_reference_used": False,
        "numeric_evaluation_reference_used": False,
        "config": {
            "beam": {"size": 4},
            "decision": {"stability_folds": 2},
        },
        "source_atlas_sha256": atlas_sha,
        "query_geometry": {"projection": "cyltan", "width_px": 64, "height_px": 48},
        "native_patch_supplied": False,
        "terrain_surface": {"aggregate_sha256": "b" * 64},
        "evidence": {
            "source": "photo_rgb_only",
            "reference_depth_used": False,
            "numeric_reference_pose_used": False,
            "shape": [48, 64],
            "comparison_shape": [12, 16],
            "extraction_config_sha256": "c" * 64,
            "source_atlas_sha256": atlas_sha,
            "observation": observation,
            "observed_skyline_sha256": "d" * 64,
            "photo_rgb_sha256": "e" * 64,
            "edge_response_sha256": "f" * 64,
            "relative_depth_sha256": "1" * 64,
            "comparison_relative_depth_sha256": "2" * 64,
            "valid_mask_sha256": "3" * 64,
            "terrain_mask_sha256": "4" * 64,
            "ridge_weights_sha256": "5" * 64,
            "models": {
                "edge": {
                    "name": "edge",
                    "aggregate_sha256": "6" * 64,
                    "offline": True,
                    "input": "photo_rgb",
                },
                "depth": {
                    "name": "depth",
                    "aggregate_sha256": "7" * 64,
                    "offline": True,
                    "input": "photo_rgb",
                },
            },
            "quality": {"usable": True},
        },
        "candidate_pool": "complete_frozen_photo_atlas_pool",
        "candidate_count": len(candidates),
        "ranked_winner_candidate_id": "winner",
        "component_winners": {
            "fusion": "winner",
            "skyline": "winner",
            "symmetric_outline": "second-beam",
            "ordinal_depth": "useful-last",
            "terrain_overlap": "winner",
        },
        "fold_winner_candidate_ids": ["winner", "second-beam"],
        "prior_position_competitor_id": "outside-beam",
        "beam_policy": "greedy_position_yaw_basin_nms",
        "beam_candidate_ids": ["winner", "second-beam", "useful-last"],
        "decision": {
            "status": "selected",
            "returned_candidate_id": "winner",
            "ranked_winner_candidate_id": "winner",
            "rival_candidate_id": "second-beam",
            "rival_margin": 0.1,
            "stable_folds": 2,
            "reasons": [],
            "fallback": None,
            "calibrated": False,
        },
        "candidates": candidates,
    }
    _rehash(record)
    return record


def _rehash(record: dict[str, Any]) -> None:
    basis = dict(record)
    basis.pop("archive_sha256", None)
    record["archive_sha256"] = module._canonical_sha256(basis)


def test_preserves_every_ordered_beam_seed_including_useful_last() -> None:
    archive = _archive()

    bridge = build_photo_beam_render_seed_bridge(archive)

    assert bridge.ordered_candidate_ids == ("winner", "second-beam", "useful-last")
    assert tuple(seed.candidate_id for seed in bridge.render_seeds) == bridge.ordered_candidate_ids
    assert bridge.render_seeds[-1].candidate_id == "useful-last"
    assert len(bridge.render_seeds) == len(archive["beam_candidate_ids"])
    assert list(inspect.signature(build_photo_beam_render_seed_bridge).parameters) == ["archive_record"]
    record = bridge.to_record()
    assert record["beam"] == {
        "source_policy": "greedy_position_yaw_basin_nms",
        "order_preserved": True,
        "truncated": False,
        "reranked": False,
        "seed_count": 3,
        "ordered_candidate_ids": ["winner", "second-beam", "useful-last"],
    }
    assert record["source"]["photo_verifier_archive_sha256"] == archive["archive_sha256"]
    assert record["source"]["skyline_atlas_archive_sha256"] == archive["source_atlas_sha256"]
    assert record["truth_boundary"]["truth_or_evaluation_inputs"] == []


def test_output_is_detached_and_maps_pitch_while_discarding_roll() -> None:
    archive = _archive()
    bridge = build_photo_beam_render_seed_bridge(archive)
    original_archive_sha = archive["archive_sha256"]

    archive["beam_candidate_ids"].reverse()
    archive["candidates"][3]["atlas_candidate"]["pose"]["position"]["east_m"] = 9999.0
    first_copy = bridge.render_seeds
    first_copy[-1].render_seed.position.east_m = -9999.0
    second_copy = bridge.render_seeds

    useful = second_copy[-1]
    assert bridge.source_verifier_archive_sha256 == original_archive_sha
    assert bridge.ordered_candidate_ids == ("winner", "second-beam", "useful-last")
    assert useful.render_seed.position.east_m == pytest.approx(40.0)
    assert useful.render_seed.pitch_deg == pytest.approx(7.5)
    assert not hasattr(useful.render_seed, "roll_deg")
    useful_record = bridge.to_record()["seeds"][-1]
    assert useful_record["render_seed"]["query_pnp_initial_pitch_deg"] == pytest.approx(7.5)
    assert useful_record["render_seed"]["physical_render_roll_deg_in_round_zero_api"] == 0.0
    assert useful_record["discarded_atlas_roll_nuisance_deg"] == pytest.approx(13.0)
    assert bridge.to_record()["pose_semantics"]["atlas_roll"].startswith("discarded")


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "inconsistent_rank",
    ],
)
def test_rejects_rehashed_missing_duplicate_and_inconsistent_beam_references(mutation: str) -> None:
    archive = _archive()
    if mutation == "missing":
        archive["beam_candidate_ids"][-1] = "missing-candidate"
    elif mutation == "duplicate":
        archive["beam_candidate_ids"][-1] = "winner"
    else:
        archive["candidates"][3]["beam_rank"] = 2
    _rehash(archive)

    with pytest.raises(ValueError, match="beam IDs|beam ranks"):
        build_photo_beam_render_seed_bridge(archive)


def test_rehashed_injected_truth_is_rejected_by_frozen_archive_validator() -> None:
    archive = _archive()
    archive["candidates"][-1]["reference_truth"] = {
        "east_m": 40.0,
        "is_useful": True,
    }
    _rehash(archive)

    with pytest.raises(ValueError, match="photo verifier candidate fields"):
        build_photo_beam_render_seed_bridge(archive)


def test_bridge_is_deterministic_and_does_not_retain_mutable_records() -> None:
    archive = _archive()

    first = build_photo_beam_render_seed_bridge(archive)
    second = build_photo_beam_render_seed_bridge(deepcopy(archive))
    mutable_record = first.to_record()
    mutable_record["seeds"][0]["render_seed"]["position"]["east_m"] = 12345.0

    assert first == second
    assert first.to_record() == second.to_record()
    assert first.to_record()["seeds"][0]["render_seed"]["position"]["east_m"] == pytest.approx(10.0)

from __future__ import annotations

import inspect
import json
import math
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from peakle.domain.camera import CameraExtrinsics
from peakle.domain.coordinates import LocalPoint
from peakle.domain.terrain import TerrainMap
from peakle.localize import pfm_geometry_rerank as module
from peakle.localize.pfm_geometry_rerank import (
    PfmGeometryRerankConfig,
    build_pfm_geometry_rerank,
    evaluate_pfm_geometry_rerank,
)
from peakle.localize.skyline_atlas import ATLAS_ARCHIVE_SCHEMA
from peakle.localize.swissdem import Patch


def _candidate(candidate_id: str, rank: int, east_m: float, skyline_score: float) -> dict[str, Any]:
    return {
        "schema": "peakle_skyline_atlas_candidate_v2",
        "candidate_id": candidate_id,
        "estimator_rank": rank,
        "grid": {"row": 0, "column": rank - 1, "yaw_index": rank - 1},
        "pose": {
            "position": {"east_m": east_m, "north_m": 0.0, "up_m": 1002.5},
            "yaw_deg": 20.0 + east_m,
            "pitch_deg": 0.0,
            "roll_deg": 1.25,
        },
        "vertical_shift_px": -7.0,
        "vertical_slope_px_per_column": 0.0,
        "estimator_score": skyline_score,
        "ground_source": "terrain_map",
    }


def _atlas(*, native_patch: bool = False) -> dict[str, Any]:
    candidates = [
        _candidate("skyline-winner", 1, 10.0, 1.0),
        _candidate("geometry-winner", 2, 0.0, 5.0),
    ]
    record = {
        "schema": ATLAS_ARCHIVE_SCHEMA,
        "numeric_evaluation_reference_used": False,
        "supplied_prior_used": True,
        "config": {"residual_cap_px": 60.0},
        "query_geometry": {
            "projection": "cyltan",
            "width_px": 16,
            "height_px": 16,
            "horizontal_fov_deg": 60.0,
        },
        "native_patch_supplied": native_patch,
        "candidate_count": len(candidates),
        "candidate_pool": "spatially_diverse_yaw_shortlist",
        "selected_candidate_id": candidates[0]["candidate_id"],
        "candidates": candidates,
    }
    record["archive_sha256"] = module._canonical_sha256(record)
    return record


def _config() -> PfmGeometryRerankConfig:
    return PfmGeometryRerankConfig(
        subsample=2,
        min_source_family_px=4,
        min_common_depth_px=4,
    )


def _source_depth() -> np.ndarray:
    depth = np.full((16, 16), 1000.0)
    depth[:, 8:] = 2000.0
    return depth


def _terrain() -> TerrainMap:
    return cast(
        TerrainMap,
        SimpleNamespace(
            x_m=np.asarray([-5_000.0, 5_000.0]),
            y_m=np.asarray([-5_000.0, 5_000.0]),
        ),
    )


def _fake_depth_renderer(calls: list[dict[str, Any]]):
    def render(
        _terrain,
        _cam_z,
        _azimuths,
        width,
        height,
        _fov,
        vertical_shift,
        east,
        north,
        roll,
        *,
        sub,
        patch,
    ):
        calls.append(
            {
                "vertical_shift": vertical_shift,
                "east": east,
                "north": north,
                "roll": roll,
                "sub": sub,
                "patch": patch,
            }
        )
        shape = (len(range(0, height, sub)), len(range(0, width, sub)))
        depth = np.full(shape, 1000.0)
        if east == 0.0:
            depth[:, shape[1] // 2 :] = 2000.0
        hit = np.zeros(shape, dtype=int)
        elevation_pixels = np.zeros(shape)
        elevation_grid = np.zeros((shape[1], 2))
        distances = np.asarray([250.0, 5000.0])
        rows = np.arange(0, height, sub)
        return depth, hit, elevation_pixels, elevation_grid, distances, rows

    return render


def test_build_is_truth_free_deterministic_and_reranks_complete_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(module, "dem_depth_image", _fake_depth_renderer(calls))
    terrain = _terrain()

    archive = build_pfm_geometry_rerank(terrain, None, _source_depth(), _atlas(), config=_config())
    repeated = build_pfm_geometry_rerank(terrain, None, _source_depth(), _atlas(), config=_config())

    assert archive.selected.candidate_id == "geometry-winner"
    assert archive.component_winners["skyline"] == "skyline-winner"
    assert {candidate.candidate_id for candidate in archive.candidates} == {
        "skyline-winner",
        "geometry-winner",
    }
    assert archive.archive_sha256 == repeated.archive_sha256
    assert archive.to_record() == repeated.to_record()
    assert all(call["vertical_shift"] == -7.0 for call in calls)
    assert all(call["roll"] == 1.25 for call in calls)
    assert all(call["sub"] == 2 for call in calls)
    assert list(inspect.signature(build_pfm_geometry_rerank).parameters) == [
        "terrain",
        "native_patch",
        "source_depth",
        "atlas_archive",
        "config",
    ]
    encoded = json.dumps(archive.to_record(), sort_keys=True)
    assert '"numeric_evaluation_reference_used": false' in encoded
    assert "horizontal_position_m" not in encoded
    assert "candidate_pool_gt_oracle" not in encoded


def test_native_patch_is_required_to_match_atlas_and_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(module, "dem_depth_image", _fake_depth_renderer(calls))
    terrain = _terrain()
    patch = cast(Patch, object())

    with pytest.raises(ValueError, match="native patch availability"):
        build_pfm_geometry_rerank(terrain, None, _source_depth(), _atlas(native_patch=True), config=_config())

    build_pfm_geometry_rerank(terrain, patch, _source_depth(), _atlas(native_patch=True), config=_config())
    assert calls and all(call["patch"] is patch for call in calls)


def test_atlas_hash_tampering_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "dem_depth_image", _fake_depth_renderer([]))
    atlas = _atlas()
    atlas["candidates"][0]["estimator_score"] = 999.0

    with pytest.raises(ValueError, match="SHA-256"):
        build_pfm_geometry_rerank(_terrain(), None, _source_depth(), atlas, config=_config())


def test_atlas_projection_and_candidate_pool_contracts_are_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "dem_depth_image", _fake_depth_renderer([]))
    for mutate, message in (
        (lambda record: record["query_geometry"].update(projection="pinhole"), "cyltan"),
        (lambda record: record.update(candidate_pool="global_nms"), "candidate-pool"),
        (lambda record: record.update(candidate_count=99), "candidate count"),
    ):
        atlas = _atlas()
        mutate(atlas)
        atlas.pop("archive_sha256")
        atlas["archive_sha256"] = module._canonical_sha256(atlas)
        with pytest.raises(ValueError, match=message):
            build_pfm_geometry_rerank(_terrain(), None, _source_depth(), atlas, config=_config())


def test_horizontal_depth_is_converted_to_camera_ray_range(monkeypatch: pytest.MonkeyPatch) -> None:
    elevation_rad = math.radians(60.0)

    def render(*_args, **_kwargs):
        shape = (8, 8)
        return (
            np.full(shape, 500.0),
            np.zeros(shape, dtype=int),
            np.full(shape, elevation_rad),
            np.zeros((shape[1], 2)),
            np.asarray([250.0, 5000.0]),
            np.arange(0, 16, 2),
        )

    monkeypatch.setattr(module, "dem_depth_image", render)

    rendered = module._render_candidate(
        _terrain(),
        None,
        _atlas()["candidates"][0],
        16,
        16,
        60.0,
        _config(),
    )

    assert rendered.candidate_ray_depth == pytest.approx(np.full((8, 8), 1000.0))


def test_post_freeze_evaluation_can_disagree_with_rerank_winner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "dem_depth_image", _fake_depth_renderer([]))
    archive = build_pfm_geometry_rerank(
        _terrain(),
        None,
        _source_depth(),
        _atlas(),
        config=_config(),
    )
    truth = CameraExtrinsics(
        position=LocalPoint(east_m=10.0, north_m=0.0, up_m=1002.5),
        yaw_deg=30.0,
        pitch_deg=0.0,
        roll_deg=0.0,
    )

    evaluation = evaluate_pfm_geometry_rerank(archive, truth, top_ks=(1, 2))

    assert evaluation.winner_errors.candidate_id == "geometry-winner"
    assert evaluation.candidate_pool_gt_oracle.candidate_id == "skyline-winner"
    assert evaluation.candidate_pool_gt_oracle.normalized_joint_error == 0.0
    assert evaluation.component_winner_errors["fusion"].candidate_id == "geometry-winner"
    assert evaluation.component_winner_errors["skyline"].candidate_id == "skyline-winner"
    assert evaluation.top_k[0]["reaches_target"] is False
    assert evaluation.top_k[1]["reaches_target"] is True


def test_source_depth_resampling_never_blends_sky_and_depth() -> None:
    source = np.asarray(
        [
            [0.0, 1000.0, 2000.0],
            [0.0, 3000.0, 4000.0],
            [0.0, 5000.0, 6000.0],
        ]
    )

    sampled = module._resample_source_depth(
        source, width=5, height=5, subsample=2, max_range_m=5500.0, min_depth_m=250.0
    )

    assert sampled.shape == (3, 3)
    assert np.isnan(sampled[:, 0]).all()
    assert set(sampled[np.isfinite(sampled)]) <= {1000.0, 2000.0, 3000.0, 4000.0, 5000.0}


def test_config_rejects_unaccounted_fusion_weight() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        PfmGeometryRerankConfig(skyline_weight=0.5)


def test_outline_cap_is_explicitly_in_comparison_grid_pixels() -> None:
    record = PfmGeometryRerankConfig(subsample=4, outline_cap_comparison_px=10.0).to_record()

    assert record["outline_cap_comparison_px"] == 10.0
    assert record["comparison_grid_pixel_scale_in_source_pixels"] == 4
    assert "outline_cap_px" not in record

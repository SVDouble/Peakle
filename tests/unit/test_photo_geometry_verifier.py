from __future__ import annotations

import inspect
import json
from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from peakle.domain.camera import CameraExtrinsics
from peakle.domain.coordinates import GeoPoint, LocalPoint
from peakle.domain.terrain import TerrainMap, TerrainSpec
from peakle.localize import photo_geometry_verifier as module
from peakle.localize.atlas_geometry import RenderedCyltanCandidateDepth
from peakle.localize.photo_geometry_verifier import (
    PhotoGeometryEvidence,
    PhotoGeometryVerifierConfig,
    build_photo_geometry_verifier,
    evaluate_photo_geometry_verifier,
    extract_photo_geometry_evidence,
    validate_frozen_photo_geometry_verifier,
)
from peakle.localize.skyline_atlas import ATLAS_ARCHIVE_SCHEMA
from peakle.segmentation import Ridge, RidgeField


def _candidate(candidate_id: str, rank: int, east_m: float, yaw_deg: float, skyline_score: float) -> dict[str, Any]:
    return {
        "schema": "peakle_skyline_atlas_candidate_v2",
        "candidate_id": candidate_id,
        "estimator_rank": rank,
        "grid": {"row": 0, "column": rank - 1, "yaw_index": rank - 1},
        "pose": {
            "position": {"east_m": east_m, "north_m": 0.0, "up_m": 1002.5},
            "yaw_deg": yaw_deg,
            "pitch_deg": 0.0,
            "roll_deg": 1.25,
        },
        "vertical_shift_px": -7.0,
        "vertical_slope_px_per_column": 0.0,
        "estimator_score": skyline_score,
        "ground_source": "terrain_map",
    }


def _atlas(skyline_hash: str, *, native_patch: bool = False) -> dict[str, Any]:
    candidates = [
        _candidate("skyline-winner", 1, 0.0, 20.0, 1.0),
        _candidate("photo-winner", 2, 100.0, 30.0, 5.0),
        _candidate("photo-winner-neighbour", 3, 110.0, 31.0, 5.5),
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
            "observed_skyline_sha256": skyline_hash,
        },
        "coordinate_frame": {
            "type": "local_equirectangular_tangent_plane",
            "origin": {"latitude_deg": 46.0, "longitude_deg": 8.0, "elevation_m": 100.0},
        },
        "prior_position": {"east_m": 0.0, "north_m": 0.0, "up_m": 1002.5},
        "native_patch_supplied": native_patch,
        "candidate_count": len(candidates),
        "candidate_pool": "spatially_diverse_yaw_shortlist",
        "selected_candidate_id": candidates[0]["candidate_id"],
        "candidates": candidates,
    }
    record["archive_sha256"] = module._canonical_sha256(record)
    return record


def _terrain() -> TerrainMap:
    size = 32
    x_m = np.linspace(-5_000.0, 5_000.0, size)
    y_m = np.linspace(-5_000.0, 5_000.0, size)
    x_grid, y_grid = np.meshgrid(x_m, y_m)
    elevation = np.full((size, size), 100.0)
    return TerrainMap(
        spec=TerrainSpec(
            origin=GeoPoint(latitude_deg=46.0, longitude_deg=8.0, elevation_m=100.0),
            width_m=10_000.0,
            height_m=10_000.0,
            grid_width=size,
            grid_height=size,
            min_elevation_m=0.0,
            max_elevation_m=1000.0,
            seed=0,
        ),
        x_m=x_m,
        y_m=y_m,
        elevation_m=elevation,
        latitude_deg=46.0 + y_grid / 111_320.0,
        longitude_deg=8.0 + x_grid / 80_000.0,
    )


def _config(**changes: Any) -> PhotoGeometryVerifierConfig:
    defaults: dict[str, Any] = {
        "subsample": 2,
        "typed_min_component_px": 1,
        "min_candidate_outline_px": 4,
        "ordinal_grid_columns": 2,
        "ordinal_grid_rows": 2,
        "ordinal_min_samples_per_cell": 2,
        "ordinal_min_cells": 2,
        "min_common_depth_px": 4,
        "minimum_skyline_coverage": 0.1,
        "minimum_ridge_effective_samples": 2.0,
        "ridge_horizontal_bins": 2,
        "minimum_ridge_horizontal_bins": 1,
        "beam_size": 2,
        "skyline_weight": 0.0,
        "outline_weight": 1.0,
        "ordinal_depth_weight": 0.0,
        "terrain_overlap_weight": 0.0,
        "maximum_selected_score": 1.0,
        "minimum_rival_margin": 0.01,
    }
    defaults.update(changes)
    return PhotoGeometryVerifierConfig(**defaults)


def _evidence() -> PhotoGeometryEvidence:
    shape = (8, 8)
    ridge = np.zeros(shape)
    ridge[:, 4] = 1.0
    depth = np.tile(np.linspace(0.0, 1.0, shape[1]), (shape[0], 1))
    valid = np.ones(shape, dtype=bool)
    source_atlas_sha = _atlas("c" * 64)["archive_sha256"]
    return PhotoGeometryEvidence(
        image_width_px=16,
        image_height_px=16,
        comparison_width_px=8,
        comparison_height_px=8,
        extraction_config_sha256=module._canonical_sha256(_config().to_record()),
        source_atlas_sha256=source_atlas_sha,
        observation={
            "track": "photo_auto",
            "source": "fake",
            "candidate": "fake",
            "selection_uses_reference_truth": False,
            "evidence_generated_at_reference_pose": False,
            "source_atlas_sha256": source_atlas_sha,
        },
        observed_skyline_sha256="c" * 64,
        photo_rgb_sha256="1" * 64,
        edge_response_sha256="e" * 64,
        relative_depth_sha256="d" * 64,
        comparison_relative_depth_sha256=module._array_sha256(
            depth,
            "peakle_photo_comparison_relative_depth_v1",
        ),
        valid_mask_sha256=module._array_sha256(valid, "peakle_photo_valid_mask_v1"),
        terrain_mask_sha256=module._array_sha256(valid, "peakle_photo_terrain_mask_v1"),
        ridge_weights_sha256=module._array_sha256(ridge, "peakle_photo_ridge_weights_v1"),
        edge_model={
            "name": "fake-edge",
            "aggregate_sha256": "a" * 64,
            "offline": True,
            "input": "photo_rgb",
        },
        depth_model={
            "name": "fake-depth",
            "aggregate_sha256": "b" * 64,
            "offline": True,
            "input": "photo_rgb",
        },
        skyline_coverage=1.0,
        ridge_effective_samples=8.0,
        ridge_horizontal_bins=1,
        relative_depth_span_p90_p10=1.0,
        usable=True,
        rejection_reasons=(),
        valid_mask=valid,
        terrain_mask=valid.copy(),
        ridge_weights=ridge,
        relative_depth=depth,
    )


def _rendered(candidate: dict[str, Any]) -> RenderedCyltanCandidateDepth:
    depth = np.full((8, 8), 1000.0)
    if candidate["candidate_id"].startswith("photo-winner"):
        depth[:, 4:] = 2000.0
    return RenderedCyltanCandidateDepth(candidate, depth, 5000.0)


def _weak_evidence() -> PhotoGeometryEvidence:
    evidence = _evidence()
    ridge = np.zeros_like(evidence.ridge_weights)
    return replace(
        evidence,
        ridge_weights=ridge,
        ridge_weights_sha256=module._array_sha256(ridge, "peakle_photo_ridge_weights_v1"),
        ridge_effective_samples=0.0,
        ridge_horizontal_bins=0,
        usable=False,
        rejection_reasons=("insufficient_internal_ridge_support", "insufficient_internal_ridge_span"),
    )


def _frozen_verifier_record(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr(module, "render_cyltan_candidate_depth", lambda *args, **kwargs: _rendered(args[2]))
    evidence = _evidence()
    return build_photo_geometry_verifier(
        _terrain(),
        None,
        evidence,
        _atlas(evidence.observed_skyline_sha256),
        config=_config(),
    ).to_record()


def _rehash_verifier_record(record: dict[str, Any]) -> None:
    basis = dict(record)
    basis.pop("archive_sha256", None)
    record["archive_sha256"] = module._canonical_sha256(basis)


def _set_record_path(record: dict[str, Any], path: tuple[str | int, ...], value: Any) -> None:
    target: Any = record
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value


def test_extracts_photo_evidence_once_with_pinned_models(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"edge": 0, "depth": 0}

    class Edge:
        name = "edge"

        def detect(self, rgb):
            calls["edge"] += 1
            return np.tile(np.linspace(0.0, 1.0, rgb.shape[1]), (rgb.shape[0], 1))

    class Depth:
        name = "depth"

        def estimate(self, rgb):
            calls["depth"] += 1
            return np.tile(np.linspace(0.0, 1.0, rgb.shape[0])[:, None], (1, rgb.shape[1]))

    rows = np.full(16, 4.0)
    ridge_rows = np.full(16, 10.0)
    ridge = Ridge(rows=ridge_rows, confidence=np.ones(16), kind="ridge")
    field = RidgeField(
        skyline=Ridge(rows=rows, confidence=np.ones(16), kind="skyline"),
        ridges=[ridge],
        response=np.zeros((16, 16)),
    )
    monkeypatch.setattr(module, "extract_ridges", lambda *_args, **_kwargs: field)

    def provenance(name: str, digest: str) -> dict[str, Any]:
        return {
            "name": name,
            "aggregate_sha256": digest * 64,
            "offline": True,
            "input": "photo_rgb",
        }

    evidence = extract_photo_geometry_evidence(
        np.full((16, 16, 3), 100, dtype=np.uint8),
        rows,
        Edge(),
        Depth(),
        observation_provenance={
            "track": "photo_auto",
            "source": "color",
            "candidate": "color",
            "selection_uses_reference_truth": False,
            "evidence_generated_at_reference_pose": False,
            "source_atlas_sha256": "c" * 64,
        },
        edge_model_provenance=provenance("edge", "a"),
        depth_model_provenance=provenance("depth", "b"),
        config=_config(skyline_exclusion_source_px=1),
    )

    assert calls == {"edge": 1, "depth": 1}
    assert evidence.observed_skyline_sha256 == module._observed_skyline_sha256(rows)
    assert evidence.ridge_effective_samples > 2.0
    assert evidence.relative_depth_span_p90_p10 > 0.1
    assert evidence.usable is True
    assert evidence.to_record()["reference_depth_used"] is False


def test_validates_and_detaches_frozen_photo_verifier_record(monkeypatch: pytest.MonkeyPatch) -> None:
    record = _frozen_verifier_record(monkeypatch)

    validated = validate_frozen_photo_geometry_verifier(record)

    assert validated == record
    record["candidates"][0]["atlas_candidate"]["pose"]["yaw_deg"] = 999.0
    assert validated["candidates"][0]["atlas_candidate"]["pose"]["yaw_deg"] != 999.0

    tampered = deepcopy(validated)
    tampered["candidates"][0]["fusion_score"] = 0.99
    with pytest.raises(ValueError, match="archive SHA-256"):
        validate_frozen_photo_geometry_verifier(tampered)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("numeric_evaluation_reference_used",), True, "truth-boundary"),
        (("evidence", "source_atlas_sha256"), "0" * 64, "different source atlas"),
        (("evidence", "observation", "track"), "pfm_oracle", "photo_auto"),
        (("candidates", 0, "reference_truth"), {"east_m": 1.0}, "photo verifier candidate fields"),
        (("candidates", 0, "atlas_candidate", "candidate_id"), "unknown", "original atlas ID"),
        (("candidates", 0, "beam_rank"), 2, "beam ranks"),
        (("component_winners", "skyline"), "unknown", "component-winner"),
        (("fold_winner_candidate_ids", 0), "unknown", "fold-winner"),
        (("prior_position_competitor_id",), "unknown", "prior-position"),
        (("decision", "rival_candidate_id"), "unknown", "rival reference"),
        (("decision", "returned_candidate_id"), None, "selected.*inconsistent"),
    ],
)
def test_rejects_rehashed_truth_rank_and_reference_mutations(
    monkeypatch: pytest.MonkeyPatch,
    path: tuple[str | int, ...],
    value: Any,
    match: str,
) -> None:
    record = _frozen_verifier_record(monkeypatch)
    _set_record_path(record, path, value)
    _rehash_verifier_record(record)

    with pytest.raises(ValueError, match=match):
        validate_frozen_photo_geometry_verifier(record)


def test_rejects_rehashed_noncontiguous_original_ranks_and_beam_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _frozen_verifier_record(monkeypatch)
    record["candidates"][0]["original_estimator_rank"] = 9
    record["candidates"][0]["atlas_candidate"]["estimator_rank"] = 9
    _rehash_verifier_record(record)
    with pytest.raises(ValueError, match="complete contiguous atlas ranks"):
        validate_frozen_photo_geometry_verifier(record)

    record = _frozen_verifier_record(monkeypatch)
    record["beam_candidate_ids"].reverse()
    for candidate in record["candidates"]:
        candidate_id = candidate["candidate_id"]
        candidate["beam_rank"] = (
            record["beam_candidate_ids"].index(candidate_id) + 1
            if candidate_id in record["beam_candidate_ids"]
            else None
        )
    _rehash_verifier_record(record)
    with pytest.raises(ValueError, match="beam IDs, order, or size"):
        validate_frozen_photo_geometry_verifier(record)


def test_pfm_label_cannot_cross_the_photo_observation_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    class Edge:
        name = "edge"

        def detect(self, rgb):
            return np.zeros(rgb.shape[:2])

    class Depth:
        name = "depth"

        def estimate(self, rgb):
            return np.zeros(rgb.shape[:2])

    monkeypatch.setattr(module, "extract_ridges", lambda *_args, **_kwargs: pytest.fail("must reject first"))

    def model(name: str, digest: str) -> dict[str, Any]:
        return {
            "name": name,
            "aggregate_sha256": digest * 64,
            "offline": True,
            "input": "photo_rgb",
        }

    with pytest.raises(ValueError, match="photo_auto"):
        extract_photo_geometry_evidence(
            np.full((16, 16, 3), 100, dtype=np.uint8),
            np.full(16, 4.0),
            Edge(),
            Depth(),
            observation_provenance={
                "track": "pfm_oracle",
                "source": "source_depth_pfm",
                "candidate": "pfm",
                "selection_uses_reference_truth": False,
                "evidence_generated_at_reference_pose": True,
                "source_atlas_sha256": "c" * 64,
            },
            edge_model_provenance=model("edge", "a"),
            depth_model_provenance=model("depth", "b"),
            config=_config(),
        )


def test_build_is_truth_free_deterministic_and_scores_complete_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    rendered_ids: list[str] = []

    def render(_terrain, _patch, candidate, *_args, **_kwargs):
        rendered_ids.append(candidate["candidate_id"])
        return _rendered(candidate)

    monkeypatch.setattr(module, "render_cyltan_candidate_depth", render)
    evidence = _evidence()
    atlas = _atlas(evidence.observed_skyline_sha256)

    archive = build_photo_geometry_verifier(_terrain(), None, evidence, atlas, config=_config())
    repeated = build_photo_geometry_verifier(_terrain(), None, evidence, atlas, config=_config())

    assert archive.ranked_winner.candidate_id == "photo-winner"
    assert archive.prior_position_competitor_id == "skyline-winner"
    assert len(archive.candidates) == 3
    assert set(rendered_ids) == {"skyline-winner", "photo-winner", "photo-winner-neighbour"}
    assert archive.beam_candidate_ids == ("photo-winner", "skyline-winner")
    assert archive.decision.status == "selected"
    assert archive.archive_sha256 == repeated.archive_sha256
    assert archive.to_record() == repeated.to_record()
    assert list(inspect.signature(build_photo_geometry_verifier).parameters) == [
        "terrain",
        "native_patch",
        "evidence",
        "atlas_archive",
        "config",
    ]
    encoded = json.dumps(archive.to_record(), sort_keys=True)
    assert '"numeric_evaluation_reference_used": false' in encoded
    assert '"source_depth_reference_used": false' in encoded
    assert "horizontal_position_m" not in encoded
    assert "candidate_pool_gt_oracle" not in encoded


def test_atlas_and_photo_skyline_hashes_are_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "render_cyltan_candidate_depth", lambda *args, **kwargs: _rendered(args[2]))
    evidence = _evidence()
    atlas = _atlas("x" * 64)
    with pytest.raises(ValueError, match="skyline hash"):
        build_photo_geometry_verifier(_terrain(), None, evidence, atlas, config=_config())

    atlas = _atlas(evidence.observed_skyline_sha256)
    atlas["candidates"][0]["estimator_score"] = 999.0
    with pytest.raises(ValueError, match="SHA-256"):
        build_photo_geometry_verifier(_terrain(), None, evidence, atlas, config=_config())


def test_evidence_array_mutation_is_rejected_before_candidate_render(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        module,
        "render_cyltan_candidate_depth",
        lambda *_args, **_kwargs: pytest.fail("candidate rendering must not start"),
    )
    evidence = _evidence()
    tampered_weights = evidence.ridge_weights.copy()
    tampered_weights[0, 0] = 1.0
    tampered = replace(evidence, ridge_weights=tampered_weights)

    with pytest.raises(ValueError, match="ridge weights differs"):
        build_photo_geometry_verifier(
            _terrain(),
            None,
            tampered,
            _atlas(evidence.observed_skyline_sha256),
            config=_config(),
        )


def test_observation_provenance_mutation_is_rejected_before_candidate_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        module,
        "render_cyltan_candidate_depth",
        lambda *_args, **_kwargs: pytest.fail("candidate rendering must not start"),
    )
    evidence = _evidence()
    evidence.observation["track"] = "pfm_oracle"

    with pytest.raises(ValueError, match="photo_auto"):
        build_photo_geometry_verifier(
            _terrain(),
            None,
            evidence,
            _atlas(evidence.observed_skyline_sha256),
            config=_config(),
        )


def test_input_mutation_is_detached_and_archive_mutation_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "render_cyltan_candidate_depth", lambda *args, **kwargs: _rendered(args[2]))
    evidence = _evidence()
    atlas = _atlas(evidence.observed_skyline_sha256)
    archive = build_photo_geometry_verifier(_terrain(), None, evidence, atlas, config=_config())
    original_east = archive.candidates[0].atlas_candidate["pose"]["position"]["east_m"]

    atlas["candidates"][1]["pose"]["position"]["east_m"] = 9999.0
    assert archive.candidates[0].atlas_candidate["pose"]["position"]["east_m"] == original_east

    archive.candidates[0].atlas_candidate["pose"]["position"]["east_m"] = 7777.0
    truth = CameraExtrinsics(
        position=LocalPoint(east_m=100.0, north_m=0.0, up_m=1002.5),
        yaw_deg=30.0,
        pitch_deg=0.0,
        roll_deg=0.0,
    )
    with pytest.raises(ValueError, match="archive SHA-256"):
        evaluate_photo_geometry_verifier(archive, truth)


def test_empty_candidate_outlines_are_penalized() -> None:
    weights = np.zeros((8, 8))
    weights[:, 4] = 1.0
    empty = np.zeros((8, 8), dtype=bool)

    assert module._symmetric_weighted_outline_terms(weights, empty, 3.0, 1) == (1.0, 1.0)


def test_camera_ray_depth_is_not_clipped_to_horizontal_terrain_extent(monkeypatch: pytest.MonkeyPatch) -> None:
    def render(_terrain, _patch, candidate, *_args, **_kwargs):
        depth = np.full((8, 8), 20_000.0)
        if candidate["candidate_id"].startswith("photo-winner"):
            depth[:, 4:] = 40_000.0
        return RenderedCyltanCandidateDepth(candidate, depth, 14_000.0)

    monkeypatch.setattr(module, "render_cyltan_candidate_depth", render)
    evidence = _evidence()
    archive = build_photo_geometry_verifier(
        _terrain(),
        None,
        evidence,
        _atlas(evidence.observed_skyline_sha256),
        config=_config(),
    )

    assert archive.ranked_winner.candidate_id == "photo-winner"
    assert archive.ranked_winner.terms.candidate_terrain_px == 64


def test_low_shared_terrain_support_forces_abstention(monkeypatch: pytest.MonkeyPatch) -> None:
    def render(_terrain, _patch, candidate, *_args, **_kwargs):
        depth = _rendered(candidate).candidate_ray_depth
        if candidate["candidate_id"] == "skyline-winner":
            depth = np.where(np.indices(depth.shape)[1] == 0, depth, np.nan)
        return RenderedCyltanCandidateDepth(candidate, depth, 5000.0)

    monkeypatch.setattr(module, "render_cyltan_candidate_depth", render)
    config = _config(
        skyline_weight=0.99,
        outline_weight=0.01,
        ordinal_depth_weight=0.0,
        minimum_winner_terrain_overlap_f1=0.5,
        minimum_winner_common_terrain_fraction=0.25,
    )
    evidence = replace(
        _evidence(),
        extraction_config_sha256=module._canonical_sha256(config.to_record()),
    )
    archive = build_photo_geometry_verifier(
        _terrain(),
        None,
        evidence,
        _atlas(evidence.observed_skyline_sha256),
        config=config,
    )

    assert archive.ranked_winner.candidate_id == "skyline-winner"
    assert archive.decision.status == "abstained"
    assert "insufficient_winner_terrain_overlap" in archive.decision.reasons
    assert "insufficient_winner_common_terrain_support" in archive.decision.reasons


def test_ordinal_depth_is_scale_invariant_and_directional() -> None:
    photo = np.repeat(np.arange(8, dtype=float)[None, :], 8, axis=0) / 7.0
    matching = np.exp(2.0 + photo)
    reversed_depth = np.exp(3.0 - photo)
    common = np.ones((8, 8), dtype=bool)
    config = _config(
        ordinal_grid_columns=4,
        ordinal_grid_rows=2,
        ordinal_min_samples_per_cell=2,
        ordinal_min_cells=4,
    )

    matching_loss, matching_cells = module._ordinal_depth_loss(photo, matching, common, config)
    reversed_loss, reversed_cells = module._ordinal_depth_loss(photo, reversed_depth, common, config)

    assert matching_cells == reversed_cells == 8
    assert matching_loss == pytest.approx(0.0)
    assert reversed_loss == pytest.approx(1.0)


def test_stability_fold_score_excludes_full_image_skyline_anchor() -> None:
    evidence = _evidence()
    atlas = _atlas(evidence.observed_skyline_sha256)
    first, second = atlas["candidates"][:2]
    depth = _rendered(second).candidate_ray_depth
    config = _config(
        skyline_weight=0.2,
        outline_weight=0.8,
        ordinal_depth_weight=0.0,
    )

    _, first_full = module._score_candidate(evidence, depth, first, atlas, config)
    _, second_full = module._score_candidate(evidence, depth, second, atlas, config)
    _, first_fold = module._score_candidate(
        evidence,
        depth,
        first,
        atlas,
        config,
        include_skyline=False,
    )
    _, second_fold = module._score_candidate(
        evidence,
        depth,
        second,
        atlas,
        config,
        include_skyline=False,
    )

    assert first_full < second_full
    assert first_fold == pytest.approx(second_fold)


def test_weak_photo_evidence_forces_abstention(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "render_cyltan_candidate_depth", lambda *args, **kwargs: _rendered(args[2]))
    evidence = _weak_evidence()
    archive = build_photo_geometry_verifier(
        _terrain(),
        None,
        evidence,
        _atlas(evidence.observed_skyline_sha256),
        config=_config(),
    )

    assert archive.decision.status == "abstained"
    assert archive.decision.returned_candidate_id is None
    assert archive.decision.reasons[:2] == (
        "insufficient_internal_ridge_support",
        "insufficient_internal_ridge_span",
    )


def test_post_freeze_evaluation_reports_ranking_beam_and_abstention(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "render_cyltan_candidate_depth", lambda *args, **kwargs: _rendered(args[2]))
    evidence = _weak_evidence()
    archive = build_photo_geometry_verifier(
        _terrain(),
        None,
        evidence,
        _atlas(evidence.observed_skyline_sha256),
        config=_config(),
    )
    truth = CameraExtrinsics(
        position=LocalPoint(east_m=100.0, north_m=0.0, up_m=1002.5),
        yaw_deg=30.0,
        pitch_deg=0.0,
        roll_deg=0.0,
    )

    evaluation = evaluate_photo_geometry_verifier(archive, truth, top_ks=(1, 2, 3))

    assert evaluation.ranked_winner_errors.reaches_target is False
    assert evaluation.returned_candidate_errors is None
    assert evaluation.first_target_verifier_rank == 2
    assert evaluation.first_target_beam_rank == 2
    assert evaluation.candidate_pool_gt_oracle.candidate_id == "photo-winner"
    assert evaluation.top_k[0]["reaches_target"] is False


def test_config_rejects_unaccounted_weight_and_invalid_fold_gate() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        PhotoGeometryVerifierConfig(skyline_weight=0.5)
    with pytest.raises(ValueError, match="stable folds"):
        PhotoGeometryVerifierConfig(stability_folds=3, minimum_stable_folds=4)
    with pytest.raises(ValueError, match="support fractions"):
        PhotoGeometryVerifierConfig(minimum_winner_terrain_overlap_f1=1.1)
    with pytest.raises(ValueError, match="non-skyline cue"):
        PhotoGeometryVerifierConfig(
            skyline_weight=1.0,
            outline_weight=0.0,
            ordinal_depth_weight=0.0,
            terrain_overlap_weight=0.0,
        )

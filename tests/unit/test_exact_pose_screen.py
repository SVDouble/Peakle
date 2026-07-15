from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics
from peakle.domain.coordinates import LocalPoint
from peakle.localize.correspondence import MatchSet
from peakle.research import exact_pose_screen as module
from peakle.research.webgl_contract import QueryArtifactFile
from peakle.scripts import bench_exact_pose_screen as cli


@dataclass(frozen=True)
class _FrozenQuery:
    files: MappingProxyType[str, bytes]
    manifest: tuple = ()


class _Matcher:
    def __init__(self, matcher_id: str, events: list[str]) -> None:
        self.matcher_id = matcher_id
        self.events = events

    def identity(self) -> dict:
        return {"id": self.matcher_id}

    def match_many(self, _query, renders) -> list[MatchSet]:
        self.events.append(f"match:{self.matcher_id}:{len(renders)}")
        return [_matches(4) for _ in renders]

    def match(self, _query, _render) -> MatchSet:
        self.events.append("identity")
        x, y = np.meshgrid(np.linspace(1, 318, 4), np.linspace(1, 178, 4))
        xy = np.column_stack((x.ravel(), y.ravel()))
        return MatchSet(xy, xy, np.ones(16))


def _matches(count: int) -> MatchSet:
    xy = np.column_stack((np.arange(count, dtype=float) + 2.0, np.arange(count, dtype=float) + 2.0))
    return MatchSet(xy, xy, np.ones(count))


def _truth() -> CameraExtrinsics:
    return CameraExtrinsics(
        position=LocalPoint(east_m=0.0, north_m=0.0, up_m=1000.0),
        yaw_deg=0.0,
        pitch_deg=0.0,
        roll_deg=0.0,
    )


def _terrain(seed: int):
    return SimpleNamespace(
        spec=SimpleNamespace(seed=seed, width_m=14_000.0, height_m=10_000.0, grid_width=97, grid_height=73),
        elevation_at=lambda _east, _north: 997.5,
    )


def _grade(passed: bool = False) -> dict:
    return {
        "raw_count": 4,
        "selected_count": 4,
        "query_lift_valid_count": 4,
        "query_lift_valid_fraction": 1.0,
        "render_lift_valid_count": 4,
        "render_lift_valid_fraction": 1.0,
        "both_lifts_valid_count": 4,
        "both_lifts_valid_fraction": 1.0,
        "correct_count": 4 if passed else 0,
        "precision": 1.0 if passed else 0.0,
        "occupied_4x4_cells": 8 if passed else 0,
        "coverage_fraction": 0.5 if passed else 0.0,
        "x_span_fraction": 0.6 if passed else 0.0,
        "y_span_fraction": 0.3 if passed else 0.0,
        "confidence": {},
        "gate": {"passed": passed},
    }


def test_registered_protocol_rejects_mislabeled_scenes_and_intrinsics() -> None:
    scenes = [
        {
            "scene_id": f"rugged-s{seed}-v{view + 1:02d}",
            "terrain_seed": seed,
            "view_index": view,
            "terrain": _terrain(seed),
            "truth": _truth(),
        }
        for seed in (31, 47)
        for view in (0, 1)
    ]
    intrinsics = CameraIntrinsics.from_horizontal_fov(320, 180, 55.0)
    module._validate_registered_protocol(scenes, intrinsics)

    with pytest.raises(ValueError, match="registered terrain contract"):
        module._validate_registered_protocol([{**scenes[0], "scene_id": "mislabeled"}, *scenes[1:]], intrinsics)
    with pytest.raises(ValueError, match="registered image"):
        module._validate_registered_protocol(scenes, CameraIntrinsics.from_horizontal_fov(320, 180, 60.0))


def test_screen_freezes_every_query_before_matcher_factories_and_truth_grading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    scenes = [
        {
            "scene_id": f"rugged-s{seed}-v{view:02d}",
            "terrain_seed": seed,
            "view_index": view - 1,
            "terrain": _terrain(seed),
            "truth": _truth(),
        }
        for seed in (31, 47)
        for view in (1, 2)
    ]
    query = SimpleNamespace(
        rgb=np.zeros((180, 320, 3), dtype=np.uint8),
        provenance=SimpleNamespace(model_dump=lambda **_kwargs: {"renderer": "fake"}),
    )

    def capture(*_args):
        events.append("capture")
        return query

    def freeze(_query, artifact_id):
        events.append("freeze")
        manifest = (
            QueryArtifactFile(
                filename=f"{artifact_id}.scene.json",
                role="sealed_truth_scene_json",
                media_type="application/json",
                dtype=None,
                shape=None,
                sha256="0" * 64,
                bytes=5,
            ),
        )
        return _FrozenQuery(MappingProxyType({f"{artifact_id}.fake": b"query"}), manifest)

    def factory(matcher_id: str):
        def build():
            events.append(f"factory:{matcher_id}")
            return _Matcher(matcher_id, events)

        return build

    class Renderer:
        def render(self, _terrain, _intrinsics, extrinsics, *, modality, terrain_stride):
            assert terrain_stride == 1
            return SimpleNamespace(
                rgb=np.zeros((180, 320, 3), dtype=np.uint8),
                provenance={"modality": modality},
                extrinsics=extrinsics,
            )

    monkeypatch.setattr(module, "freeze_webgl_query_artifact", freeze)
    monkeypatch.setattr(
        module,
        "load_frozen_webgl_query_rgb",
        lambda *_args: events.append("rgb-load") or np.zeros((180, 320, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        module,
        "load_frozen_webgl_query_artifact",
        lambda *_args, **_kwargs: events.append("truth-load") or object(),
    )
    monkeypatch.setattr(
        module,
        "cross_render_calibration",
        lambda *_args: {"mask_iou": 1.0, "shared_depth_count": 100, "p95_abs_log_depth": 0.0, "passed": True},
    )
    monkeypatch.setattr(module, "grade_frozen_exact_pose_correspondences", lambda *_args: _grade())
    results, files = module.run_exact_pose_screen(
        scenes,
        CameraIntrinsics.from_horizontal_fov(320, 180, 55.0),
        {matcher_id: factory(matcher_id) for matcher_id in ("sift", "roma_outdoor", "minima_roma")},
        capture,
        identity_matcher=_Matcher("sift", events),
        terrain_renderer=Renderer(),
    )

    last_freeze = max(index for index, value in enumerate(events) if value == "freeze")
    first_factory = min(index for index, value in enumerate(events) if value.startswith("factory:"))
    first_rgb = min(index for index, value in enumerate(events) if value == "rgb-load")
    last_rgb = max(index for index, value in enumerate(events) if value == "rgb-load")
    last_match = max(index for index, value in enumerate(events) if value.startswith("match:"))
    first_truth = min(index for index, value in enumerate(events) if value == "truth-load")
    assert last_freeze < first_rgb <= last_rgb < first_factory
    assert last_match < first_truth
    assert [value for value in events if value.endswith(":6")] == [
        f"match:{matcher}:6" for _query in range(2) for matcher in ("sift", "roma_outdoor", "minima_roma")
    ]
    assert len(results["case_evaluations"]) == 54
    assert len(results["identical_query_calibrations"]) == 4
    assert len(files) == 62
    assert results["observations"][0]["query_files"][0]["role"] == "sealed_truth_scene_json"
    assert all("query_render" not in observation for observation in results["observations"])
    assert module.ExactPoseScreenResult.model_validate(results).schema_id == module.SCHEMA
    with pytest.raises(ValidationError, match="extra_forbidden"):
        module.ExactPoseScreenResult.model_validate({**results, "undeclared": True})


def test_pnp_receives_all_selected_render_lift_valid_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    frozen = module.freeze_match_artifact(_matches(5), "matches")
    run = {
        "match_run_id": "matches",
        "kind": "positive",
        "query_id": "rugged-s1-v01",
        "render_id": "rugged-s1-v01",
        "matcher_id": "roma_outdoor",
        "modality": "hillshade",
        "manifest": frozen.manifest,
        "artifact_files": frozen.files,
    }
    render = SimpleNamespace(extrinsics=_truth())
    prepared = {
        "rugged-s1-v01": {
            "scene": {"scene_id": "rugged-s1-v01", "terrain": object(), "truth": _truth()},
            "renders": {"hillshade": render},
        }
    }
    captured: dict = {}
    valid = np.array([True, False, True, False, True])
    monkeypatch.setattr(
        module,
        "lift_render_pixels",
        lambda *_args: SimpleNamespace(valid=valid, world_xyz_m=np.arange(15, dtype=float).reshape(5, 3)),
    )

    def fit(world, image, confidence, *_args, **kwargs):
        captured.update(world=world, image=image, confidence=confidence, kwargs=kwargs)
        return SimpleNamespace(status="abstained", solved=False, extrinsics=None, diagnostics={})

    monkeypatch.setattr(module, "fit_pose_ransac", fit)
    results = module._run_pnp(
        [
            {
                "pair_id": "roma_outdoor:hillshade",
                "matcher_id": "roma_outdoor",
                "modality": "hillshade",
                "correspondence_survived": True,
            }
        ],
        [run],
        prepared,
        CameraIntrinsics.from_horizontal_fov(32, 18, 55.0),
    )

    assert captured["world"].shape == (3, 3)
    assert captured["image"].shape == (3, 2)
    assert captured["kwargs"] == {
        "prior": None,
        "use_position_prior": False,
        "use_orientation_prior": False,
        "config": module.PNP_CONFIG,
    }
    assert results[0]["input_selected_render_lift_valid"] == 3


def test_advancement_is_same_modality_sift_relative_and_deterministic() -> None:
    pairs = []
    for matcher in ("sift", "roma_outdoor", "minima_roma"):
        for modality in module.MODALITIES:
            learned = matcher != "sift" and modality == "hillshade"
            pairs.append(
                {
                    "pair_id": f"{matcher}:{modality}",
                    "matcher_id": matcher,
                    "modality": modality,
                    "correspondence_survived": learned,
                    "positive_pass_count": 4 if learned else 0,
                    "median_coverage": 0.5 if learned else 0.25,
                    "median_precision": 0.9,
                    "median_correct_count": 30.0 if learned else 20.0,
                }
            )
    pnp = []
    for matcher in ("roma_outdoor", "minima_roma"):
        pair_id = f"{matcher}:hillshade"
        pnp.extend(
            {
                "pair_id": pair_id,
                "kind": "positive",
                "query_id": f"rugged-s{seed}-v{view:02d}",
                "passed": True,
                "solved": True,
            }
            for seed in (1, 2)
            for view in (1, 2)
        )
        pnp.extend(
            {
                "pair_id": pair_id,
                "kind": "cross_seed_negative",
                "query_id": f"rugged-s{seed}-v01",
                "passed": True,
                "solved": False,
            }
            for seed in (1, 2)
        )

    result = module._advance_pairs(pairs, pnp)

    assert result["surviving_pairs"] == ["roma_outdoor:hillshade", "minima_roma:hillshade"]
    assert result["advanced_pairs"] == ["minima_roma:hillshade", "roma_outdoor:hillshade"]


def test_cli_refuses_existing_output_before_reading_provenance(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    monkeypatch.setattr(cli, "whole_worktree_provenance", lambda *_args: pytest.fail("must not inspect source"))
    monkeypatch.setattr("sys.argv", ["bench", "--output", str(output)])

    with pytest.raises(SystemExit, match="refusing to overwrite"):
        cli.main()


def test_artifact_merge_rejects_filename_collisions() -> None:
    target = {"artifact.bin": b"first"}

    with pytest.raises(ValueError, match="artifact filename collision"):
        module._merge_files(target, {"artifact.bin": b"replacement"})

    assert target == {"artifact.bin": b"first"}

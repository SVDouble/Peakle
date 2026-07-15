"""Registered exact-pose representation-by-matcher screen."""

from __future__ import annotations

import math
import resource
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, replace
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from peakle.domain.angles import angle_delta_deg
from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics, CameraModel
from peakle.domain.terrain import TerrainMap
from peakle.localize.correspondence import DenseMatcher, MatchSet, SiftMatcher, match_image_fan
from peakle.localize.pnp import PoseRansacConfig, fit_pose_ransac
from peakle.rendering.terrain_view import TerrainViewRenderer, lift_render_pixels
from peakle.research.exact_pose_correspondence import (
    EXACT_POSE_CASE_GATE,
    cross_render_calibration,
    freeze_match_artifact,
    grade_frozen_exact_pose_correspondences,
    identical_query_calibration,
    load_frozen_match_artifact,
)
from peakle.research.webgl_contract import (
    freeze_webgl_query_artifact,
    load_frozen_webgl_query_artifact,
    load_frozen_webgl_query_rgb,
)

SCHEMA = "peakle_exact_pose_representation_screen_v1"
MODALITIES = ("hillshade", "relative_depth", "camera_normal")
PNP_CONFIG = replace(
    PoseRansacConfig(), prior_weight_px=0.0, ground_weight_px=0.0, clearance_constraint_policy="free_up"
)
QueryCapture = Callable[[TerrainMap, CameraIntrinsics, CameraExtrinsics], Any]
MatcherFactory = Callable[[], DenseMatcher]


class ExactPoseScreenResult(BaseModel):
    """Strict JSON envelope for the registered screen artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False, populate_by_name=True)

    schema_id: Literal["peakle_exact_pose_representation_screen_v1"] = Field(alias="schema")
    run_id: str | None = None
    contract: dict[str, Any]
    config: dict[str, Any]
    observations: list[dict[str, Any]]
    cross_renderer_calibrations: list[dict[str, Any]]
    identical_query_calibrations: list[dict[str, Any]]
    match_runs: list[dict[str, Any]]
    case_evaluations: list[dict[str, Any]]
    pair_decisions: list[dict[str, Any]]
    pnp_evaluations: list[dict[str, Any]]
    aggregates: dict[str, Any]
    runtime_s: float


RunSpec = tuple[str, str, str, str, str]


def run_exact_pose_screen(
    scenes: list[dict[str, Any]],
    intrinsics: CameraIntrinsics,
    matcher_factories: Mapping[str, MatcherFactory],
    query_capture: QueryCapture,
    *,
    identity_matcher: DenseMatcher | None = None,
    terrain_renderer: TerrainViewRenderer | None = None,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Capture/freeze queries, freeze all matches, then open truth for grading."""

    started = time.perf_counter()
    _validate_registered_protocol(scenes, intrinsics)
    prepared, files = _prepare_scenes(scenes, intrinsics, query_capture, terrain_renderer)
    matchers, runs, identity_runs = _freeze_all_matches(
        prepared,
        matcher_factories,
        identity_matcher,
        files,
    )
    loaded_queries = {
        scene_id: load_frozen_webgl_query_artifact(item["query"].manifest, item["query"].files)
        for scene_id, item in prepared.items()
    }
    calibrations: list[dict[str, Any]] = []
    for scene_id, item in prepared.items():
        calibration = cross_render_calibration(loaded_queries[scene_id], item["renders"]["hillshade"])
        calibrations.append({"observation_id": scene_id, **calibration})
    identity_grades = [
        {
            "query_id": run["query_id"],
            **identical_query_calibration(
                run["manifest"], run["artifact_files"], width_px=intrinsics.width_px, height_px=intrinsics.height_px
            ),
        }
        for run in identity_runs
    ]
    evaluations: list[dict[str, Any]] = []
    for run in runs:
        grade = grade_frozen_exact_pose_correspondences(
            run["manifest"],
            run["artifact_files"],
            loaded_queries[run["query_id"]],
            intrinsics,
            _scene(prepared[run["query_id"]]["scene"])[2],
            prepared[run["render_id"]]["renders"][run["modality"]],
        )
        evaluations.append({**_run_record(run), "grade": grade})
    harness_passed = all(item["passed"] for item in calibrations + identity_grades)
    pairs = _aggregate_pairs(evaluations, scenes, harness_passed)
    pnp = _run_pnp(pairs, runs, prepared, intrinsics)
    decisions = _advance_pairs(pairs, pnp)
    results = ExactPoseScreenResult(
        schema=SCHEMA,
        contract=_contract(),
        config=_configuration(intrinsics, matchers),
        observations=[_observation(scene_id, item) for scene_id, item in prepared.items()],
        cross_renderer_calibrations=calibrations,
        identical_query_calibrations=identity_grades,
        match_runs=[_run_record(run) for run in [*runs, *identity_runs]],
        case_evaluations=evaluations,
        pair_decisions=pairs,
        pnp_evaluations=pnp,
        aggregates={"harness_passed": harness_passed, **decisions},
        runtime_s=round(time.perf_counter() - started, 6),
    ).model_dump(mode="json", by_alias=True)
    return results, files


def _prepare_scenes(
    scenes: list[dict[str, Any]],
    intrinsics: CameraIntrinsics,
    query_capture: QueryCapture,
    terrain_renderer: TerrainViewRenderer | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, bytes]]:
    """Freeze every query before exposing even its RGB to the matcher phase."""

    files: dict[str, bytes] = {}
    prepared: dict[str, dict[str, Any]] = {}
    for scene in scenes:
        scene_id, terrain, truth = _scene(scene)
        if scene_id in prepared:
            raise ValueError(f"duplicate exact-pose scene id: {scene_id}")
        query = query_capture(terrain, intrinsics, truth)
        frozen = freeze_webgl_query_artifact(query, f"query-{scene_id}")
        _merge_files(files, frozen.files)
        provenance = query.provenance.model_dump(mode="json", by_alias=True)
        prepared[scene_id] = {"scene": scene, "query": frozen, "query_provenance": provenance}
        del query
    for item in prepared.values():
        item["query_rgb"] = load_frozen_webgl_query_rgb(item["query"].manifest, item["query"].files)
    renderer = terrain_renderer or TerrainViewRenderer()
    for item in prepared.values():
        _, terrain, truth = _scene(item["scene"])
        item["renders"] = {
            modality: renderer.render(terrain, intrinsics, truth, modality=modality, terrain_stride=1)
            for modality in MODALITIES
        }
    return prepared, files


def _freeze_all_matches(
    prepared: Mapping[str, dict[str, Any]],
    matcher_factories: Mapping[str, MatcherFactory],
    identity_matcher: DenseMatcher | None,
    files: dict[str, bytes],
) -> tuple[dict[str, DenseMatcher], list[dict[str, Any]], list[dict[str, Any]]]:
    """Construct matchers after query freeze and freeze every output before grading."""

    matchers = {matcher_id: factory() for matcher_id, factory in matcher_factories.items()}
    if set(matchers) != {"sift", "roma_outdoor", "minima_roma"}:
        raise ValueError("registered matchers must be exactly sift, roma_outdoor, and minima_roma")
    runs: list[dict[str, Any]] = []
    identity_runs: list[dict[str, Any]] = []
    sift_identity = identity_matcher or SiftMatcher()
    for query_id, item in prepared.items():
        query_rgb = np.asarray(item["query_rgb"], dtype=np.uint8)
        specs = _fan_specs(query_id, prepared)
        render_images = [prepared[render_id]["renders"][modality].rgb for _, render_id, modality in specs]
        for matcher_id, matcher in matchers.items():
            fan_started = time.perf_counter()
            matches = match_image_fan(matcher, query_rgb, render_images)
            fan_runtime = time.perf_counter() - fan_started
            for match, (kind, render_id, modality) in zip(matches, specs, strict=True):
                runs.append(
                    _freeze_run(
                        match,
                        files,
                        len(runs),
                        (kind, query_id, render_id, matcher_id, modality),
                        fan_runtime,
                        sift_memory=matcher_id == "sift",
                    )
                )
        identity_started = time.perf_counter()
        identity_runs.append(
            _freeze_run(
                sift_identity.match(query_rgb, query_rgb),
                files,
                1000 + len(identity_runs),
                ("identical_query_calibration", query_id, query_id, "sift", "query_rgb"),
                time.perf_counter() - identity_started,
                sift_memory=True,
            )
        )
    return matchers, runs, identity_runs


def _fan_specs(query_id: str, prepared: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    specs = [("positive", query_id, modality) for modality in MODALITIES]
    if query_id.endswith("-v01"):
        seed = prepared[query_id]["scene"]["terrain_seed"]
        other = next(
            key for key, item in prepared.items() if key.endswith("-v01") and item["scene"]["terrain_seed"] != seed
        )
        specs.extend(("cross_seed_negative", other, modality) for modality in MODALITIES)
    return specs


def _freeze_run(
    matches: MatchSet,
    files: dict[str, bytes],
    index: int,
    spec: RunSpec,
    runtime_s: float,
    *,
    sift_memory: bool = False,
) -> dict[str, Any]:
    kind, query_id, render_id, matcher_id, modality = spec
    frozen = freeze_match_artifact(matches, f"match-{index:04d}")
    _merge_files(files, frozen.files)
    return {
        "match_run_id": f"match-{index:04d}",
        "kind": kind,
        "query_id": query_id,
        "render_id": render_id,
        "matcher_id": matcher_id,
        "modality": modality,
        "manifest": frozen.manifest,
        "artifact_files": frozen.files,
        "raw_count": matches.count,
        "selected_count": int(np.asarray(matches.selected).sum()),
        "runtime": {
            "producer_pair_s": matches.diagnostics.get("runtime_s"),
            "shared_fan_wall_s": round(runtime_s, 6),
            "shared_fan_wall_semantics": "whole grouped matcher invocation; do not sum across pair records",
        },
        "memory": (
            {
                "ru_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "semantics": "Linux cumulative process high-water RSS after in-process SIFT; not per-call delta",
            }
            if sift_memory
            else matches.diagnostics.get("memory")
        ),
        "diagnostics": matches.diagnostics,
        "provenance": matches.provenance,
    }


def _run_record(run: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: ([asdict(record) for record in value] if key == "manifest" else value)
        for key, value in run.items()
        if key != "artifact_files"
    }


def _aggregate_pairs(
    evaluations: list[dict[str, Any]], scenes: list[dict[str, Any]], harness: bool
) -> list[dict[str, Any]]:
    seeds = {scene["scene_id"]: scene["terrain_seed"] for scene in scenes}
    pairs = []
    for matcher in ("sift", "roma_outdoor", "minima_roma"):
        for modality in MODALITIES:
            cases = [item for item in evaluations if (item["matcher_id"], item["modality"]) == (matcher, modality)]
            positives = [item for item in cases if item["kind"] == "positive"]
            negatives = [item for item in cases if item["kind"] == "cross_seed_negative"]
            passed = [item for item in positives if item["grade"]["gate"]["passed"]]
            pair_passed = bool(
                harness
                and len(passed) >= 3
                and len({seeds[item["query_id"]] for item in passed}) == 2
                and not any(item["grade"]["gate"]["passed"] for item in negatives)
            )
            pairs.append(
                {
                    "pair_id": f"{matcher}:{modality}",
                    "matcher_id": matcher,
                    "modality": modality,
                    "positive_pass_count": len(passed),
                    "negative_pass_count": sum(item["grade"]["gate"]["passed"] for item in negatives),
                    "median_coverage": float(np.median([item["grade"]["coverage_fraction"] for item in positives])),
                    "median_precision": float(np.median([item["grade"]["precision"] for item in positives])),
                    "median_correct_count": float(np.median([item["grade"]["correct_count"] for item in positives])),
                    "correspondence_survived": pair_passed,
                }
            )
    return pairs


def _run_pnp(
    pairs: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    prepared: Mapping[str, Any],
    intrinsics: CameraIntrinsics,
) -> list[dict[str, Any]]:
    results = []
    camera = CameraModel.from_intrinsics(intrinsics)
    for pair in (item for item in pairs if item["correspondence_survived"]):
        for run in runs:
            if (run["matcher_id"], run["modality"]) != (pair["matcher_id"], pair["modality"]):
                continue
            matches = load_frozen_match_artifact(run["manifest"], run["artifact_files"])
            render = prepared[run["render_id"]]["renders"][run["modality"]]
            lifted = lift_render_pixels(render, matches.render_xy_px)
            use = np.asarray(matches.selected) & lifted.valid
            fit = fit_pose_ransac(
                lifted.world_xyz_m[use],
                matches.query_xy_px[use],
                matches.confidence[use],
                camera,
                render.extrinsics,
                prior=None,
                use_position_prior=False,
                use_orientation_prior=False,
                config=PNP_CONFIG,
            )
            truth = _scene(prepared[run["query_id"]]["scene"])[2]
            horizontal = yaw = None
            if fit.solved and fit.extrinsics is not None:
                horizontal = math.hypot(
                    fit.extrinsics.position.east_m - truth.position.east_m,
                    fit.extrinsics.position.north_m - truth.position.north_m,
                )
                yaw = angle_delta_deg(fit.extrinsics.yaw_deg, truth.yaw_deg)
            passed = (
                fit.solved and horizontal is not None and yaw is not None and horizontal <= 50.0 and yaw <= 1.0
                if run["kind"] == "positive"
                else not fit.solved
            )
            results.append(
                {
                    **{key: run[key] for key in ("match_run_id", "kind", "query_id", "render_id")},
                    "pair_id": pair["pair_id"],
                    "input_selected_render_lift_valid": int(use.sum()),
                    "status": fit.status,
                    "solved": fit.solved,
                    "initializer_extrinsics": render.extrinsics.model_dump(mode="json"),
                    "fitted_extrinsics": fit.extrinsics.model_dump(mode="json") if fit.extrinsics else None,
                    "horizontal_error_m": horizontal,
                    "yaw_error_deg": yaw,
                    "passed": bool(passed),
                    "diagnostics": fit.diagnostics,
                }
            )
    return results


def _advance_pairs(pairs: list[dict[str, Any]], pnp: list[dict[str, Any]]) -> dict[str, Any]:
    for pair in pairs:
        cases = [item for item in pnp if item["pair_id"] == pair["pair_id"]]
        positives = [item for item in cases if item["kind"] == "positive"]
        negatives = [item for item in cases if item["kind"] != "positive"]
        pair["pnp_positive_pass_count"] = sum(item["passed"] for item in positives)
        pair["pnp_survived"] = bool(
            pair["correspondence_survived"]
            and pair["pnp_positive_pass_count"] >= 3
            and len({item["query_id"].split("-")[1] for item in positives if item["passed"]}) == 2
            and not any(item["solved"] for item in negatives)
        )
    sift = {item["modality"]: item for item in pairs if item["matcher_id"] == "sift"}
    eligible = [
        item
        for item in pairs
        if item["matcher_id"] != "sift"
        and item["pnp_survived"]
        and item["median_coverage"] > sift[item["modality"]]["median_coverage"]
        and item["median_precision"] >= sift[item["modality"]]["median_precision"] - 0.05
    ]
    eligible.sort(
        key=lambda item: (
            -item["positive_pass_count"],
            -item["median_coverage"],
            -item["median_precision"],
            -item["median_correct_count"],
            item["pair_id"],
        )
    )
    return {
        "surviving_pairs": [item["pair_id"] for item in pairs if item["pnp_survived"]],
        "advanced_pairs": [item["pair_id"] for item in eligible[:2]],
    }


def _contract() -> dict[str, Any]:
    return {
        "truth_class": "independent_rasterizer_same_scene_model",
        "matcher_inputs": ["query_rgb", "candidate_rgb"],
        "all_queries_frozen_before_matcher_construction": True,
        "all_matches_frozen_before_truth_geometry_grading": True,
        "pnp_truth_filtering": False,
        "oracle_setup": "exact estimator terrain, pose initializer, and FOV; representation screen only",
    }


def _configuration(intrinsics: CameraIntrinsics, matchers: Mapping[str, DenseMatcher]) -> dict[str, Any]:
    return {
        "seeds": [31, 47],
        "views_per_seed": 2,
        "image": {**intrinsics.model_dump(mode="json"), "horizontal_fov_deg": intrinsics.horizontal_fov_deg()},
        "terrain": {
            "width_m": 14_000.0,
            "height_m": 10_000.0,
            "grid_width": 97,
            "grid_height": 73,
            "render_stride": 1,
            "eye_height_m": 2.5,
        },
        "modalities": list(MODALITIES),
        "matchers": {key: value.identity() for key, value in matchers.items()},
        "case_construction": {
            "positive": "four exact-pose same-seed query/render pairs",
            "negative": "rugged-s31-v01 to rugged-s47-v01 and reverse",
        },
        "case_gate": asdict(EXACT_POSE_CASE_GATE),
        "geographic_correctness": {"reprojection_px": 5.0, "world_m": "max(25, 0.01 * query_range)"},
        "cross_renderer_gate": {"mask_iou_min": 0.99, "p95_abs_log_depth_max": 0.01},
        "identical_query_gate": {"selected_min": 12, "within_1px_fraction_min": 0.95, "occupied_min": 4},
        "pair_gate": {"positive_passes": 3, "both_seeds": True, "negative_passes": 0},
        "pnp": PNP_CONFIG.__dict__,
        "pnp_inputs": {
            "prior": None,
            "position_prior": False,
            "orientation_prior": False,
            "initializer": "exact candidate render extrinsics",
        },
        "pnp_gate": {"positive_passes": 3, "both_seeds": True, "horizontal_m": 50.0, "yaw_deg": 1.0},
        "advancement": {
            "max_pairs": 2,
            "precision_loss_max": 0.05,
            "coverage": "strictly greater than same-modality SIFT positive median occupied_4x4/16",
        },
    }


def _observation(scene_id: str, item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "observation_id": scene_id,
        "terrain_seed": item["scene"]["terrain_seed"],
        "view_index": item["scene"]["view_index"],
        "query_files": [record.model_dump(mode="json", by_alias=True) for record in item["query"].manifest],
        "query_provenance": item["query_provenance"],
        "renders": {modality: bundle.provenance for modality, bundle in item["renders"].items()},
    }


def _scene(scene: Mapping[str, Any]) -> tuple[str, TerrainMap, CameraExtrinsics]:
    return scene["scene_id"], scene["terrain"], scene["truth"]


def _validate_registered_protocol(scenes: list[dict[str, Any]], intrinsics: CameraIntrinsics) -> None:
    expected = {(seed, view): f"rugged-s{seed}-v{view + 1:02d}" for seed in (31, 47) for view in (0, 1)}
    scene_keys = {(scene["terrain_seed"], scene["view_index"]) for scene in scenes}
    if intrinsics != CameraIntrinsics.from_horizontal_fov(320, 180, 55.0) or scene_keys != set(expected):
        raise ValueError("exact-pose screen inputs differ from the registered image or scene count")
    for scene in scenes:
        scene_id, terrain, truth = _scene(scene)
        seed, view, spec = scene["terrain_seed"], scene["view_index"], terrain.spec
        terrain_contract = (spec.seed, spec.width_m, spec.height_m, spec.grid_width, spec.grid_height)
        clearance = truth.position.up_m - terrain.elevation_at(truth.position.east_m, truth.position.north_m)
        if (
            expected[(seed, view)] != scene_id
            or terrain_contract != (seed, 14_000.0, 10_000.0, 97, 73)
            or not math.isclose(clearance, 2.5, abs_tol=1e-6)
        ):
            raise ValueError(f"exact-pose scene differs from the registered terrain contract: {scene_id}")


def _merge_files(target: dict[str, bytes], incoming: Mapping[str, bytes]) -> None:
    if overlap := set(target) & set(incoming):
        raise ValueError(f"artifact filename collision: {', '.join(sorted(overlap))}")
    target.update(incoming)

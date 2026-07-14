"""Truth-separated custom-pinhole, shared-renderer stage upper-bound.

The real-data benchmark answers whether Peakle works on photographs.  This
module answers a narrower question: *which isolated score fails when the answer
is known exactly?*  Synthetic query evidence is rendered from a private
reference pose, then a custom pinhole candidate archive is built with the same
mesh renderer.  The query evidence and controlled priors are reference-derived,
but numeric truth fields are not passed directly to candidate scoring.  Only
:func:`evaluate_synthetic_candidate_archive` receives the truth object.  It does not
exercise the production cylindrical/raycast atlas, learned render matching,
PnP, holdout validation, or continuous refinement.

The benchmark deliberately keeps separate measurements for:

* photo-side skyline extraction;
* proposal-pool recall before any score is inspected;
* skyline-only ranking;
* ideal, reference-rendered depth and typed-outline ranking ceilings; and
* ambiguity detection / abstention on a rotationally symmetric control.

Reference-rendered depth is an oracle analysis track.  It is not a production
input and must never be presented as PFM/photo evidence or a deployable result.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt, gaussian_filter

from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics
from peakle.domain.coordinates import LocalPoint
from peakle.domain.terrain import TerrainMap
from peakle.localize.typed_outlines import TypedOutlines, extract_typed_outlines
from peakle.rendering.rasterizer import SyntheticRenderer
from peakle.rendering.terrain_view import terrain_fingerprint

SYNTHETIC_BENCHMARK_SCHEMA = "peakle_synthetic_pose_pipeline_benchmark_v2"
SYNTHETIC_ARCHIVE_SCHEMA = "peakle_synthetic_candidate_archive_v2"
SYNTHETIC_EVALUATION_SCHEMA = "peakle_synthetic_candidate_evaluation_v2"

SKYLINE_METHOD = "skyline"
ORACLE_METRIC_DEPTH_METHOD = "oracle_metric_depth"
ORACLE_RELATIVE_DEPTH_METHOD = "oracle_relative_depth"
ORACLE_TYPED_DEPTH_METHOD = "oracle_typed_depth_outlines"
ORACLE_GEOMETRY_FUSION_METHOD = "oracle_geometry_fusion"
RANKING_METHODS = (
    SKYLINE_METHOD,
    ORACLE_METRIC_DEPTH_METHOD,
    ORACLE_RELATIVE_DEPTH_METHOD,
    ORACLE_TYPED_DEPTH_METHOD,
    ORACLE_GEOMETRY_FUSION_METHOD,
)
TRACK_RANKING_METHODS = (SKYLINE_METHOD, ORACLE_GEOMETRY_FUSION_METHOD)
TRACK_INVARIANT_ORACLE_METHODS = (
    ORACLE_METRIC_DEPTH_METHOD,
    ORACLE_RELATIVE_DEPTH_METHOD,
    ORACLE_TYPED_DEPTH_METHOD,
)
ORACLE_ONLY_METHODS = frozenset(
    {
        ORACLE_METRIC_DEPTH_METHOD,
        ORACLE_RELATIVE_DEPTH_METHOD,
        ORACLE_TYPED_DEPTH_METHOD,
        ORACLE_GEOMETRY_FUSION_METHOD,
    }
)


@dataclass(frozen=True, slots=True)
class SyntheticSearchConfig:
    """Frozen candidate-generation and scoring contract."""

    position_spacing_m: float = 100.0
    position_radius_steps: int = 2
    yaw_spacing_deg: float = 5.0
    yaw_radius_steps: int = 3
    eye_height_m: float = 2.5
    render_stride: int = 2
    depth_log_cap: float = math.log(2.0)
    outline_distance_cap_px: float = 8.0
    skyline_residual_cap_px: float = 40.0
    target_position_m: float = 50.0
    target_yaw_deg: float = 5.0
    ambiguity_score_delta: float = 0.0025
    ambiguity_position_separation_m: float = 50.0
    ambiguity_yaw_separation_deg: float = 8.0
    minimum_extraction_coverage: float = 0.50
    minimum_extraction_agreement: float = 0.40

    def __post_init__(self) -> None:
        positive = {
            "position_spacing_m": self.position_spacing_m,
            "yaw_spacing_deg": self.yaw_spacing_deg,
            "eye_height_m": self.eye_height_m,
            "depth_log_cap": self.depth_log_cap,
            "outline_distance_cap_px": self.outline_distance_cap_px,
            "skyline_residual_cap_px": self.skyline_residual_cap_px,
            "target_position_m": self.target_position_m,
            "target_yaw_deg": self.target_yaw_deg,
            "ambiguity_position_separation_m": self.ambiguity_position_separation_m,
            "ambiguity_yaw_separation_deg": self.ambiguity_yaw_separation_deg,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in (
            ("position_radius_steps", self.position_radius_steps),
            ("yaw_radius_steps", self.yaw_radius_steps),
            ("render_stride", self.render_stride),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.ambiguity_score_delta) or self.ambiguity_score_delta < 0.0:
            raise ValueError("ambiguity_score_delta must be finite and non-negative")
        for name, value in (
            ("minimum_extraction_coverage", self.minimum_extraction_coverage),
            ("minimum_extraction_agreement", self.minimum_extraction_agreement),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    def to_record(self) -> dict[str, Any]:
        return {
            "position_spacing_m": self.position_spacing_m,
            "position_radius_steps": self.position_radius_steps,
            "yaw_spacing_deg": self.yaw_spacing_deg,
            "yaw_radius_steps": self.yaw_radius_steps,
            "eye_height_m": self.eye_height_m,
            "render_stride": self.render_stride,
            "scores": {
                "depth_log_cap": self.depth_log_cap,
                "outline_distance_cap_px": self.outline_distance_cap_px,
                "skyline_residual_cap_px": self.skyline_residual_cap_px,
                "geometry_fusion_weights": {
                    "skyline": 0.15,
                    "relative_depth": 0.55,
                    "typed_depth_outlines": 0.20,
                    "depth_overlap": 0.10,
                },
            },
            "success_target": {
                "horizontal_position_m_lte": self.target_position_m,
                "absolute_yaw_deg_lte": self.target_yaw_deg,
            },
            "abstention": {
                "score_delta_lte": self.ambiguity_score_delta,
                "competitive_position_separation_m_gte": self.ambiguity_position_separation_m,
                "competitive_yaw_separation_deg_gte": self.ambiguity_yaw_separation_deg,
                "minimum_extraction_coverage": self.minimum_extraction_coverage,
                "minimum_extraction_agreement": self.minimum_extraction_agreement,
            },
        }


@dataclass(frozen=True, slots=True)
class SyntheticPriorRegime:
    """A prior perturbation expressed in search-grid steps."""

    name: str
    east_steps: int
    north_steps: int
    yaw_steps: int

    def to_record(self, config: SyntheticSearchConfig) -> dict[str, Any]:
        return {
            "name": self.name,
            "east_offset_m": self.east_steps * config.position_spacing_m,
            "north_offset_m": self.north_steps * config.position_spacing_m,
            "yaw_offset_deg": self.yaw_steps * config.yaw_spacing_deg,
        }


DEFAULT_PRIOR_REGIMES: dict[str, SyntheticPriorRegime] = {
    "exact": SyntheticPriorRegime("exact", 0, 0, 0),
    "moderate": SyntheticPriorRegime("moderate", 1, -1, 2),
    "wide": SyntheticPriorRegime("wide", 2, -1, 3),
}


def controlled_prior(
    terrain: TerrainMap,
    truth: CameraExtrinsics,
    regime: SyntheticPriorRegime,
    config: SyntheticSearchConfig,
) -> CameraExtrinsics:
    """Construct a deterministic prior whose offset is fully declared."""

    east_m = truth.position.east_m + regime.east_steps * config.position_spacing_m
    north_m = truth.position.north_m + regime.north_steps * config.position_spacing_m
    _validate_inside_terrain(terrain, east_m, north_m, "controlled prior")
    return CameraExtrinsics(
        position=LocalPoint(
            east_m=east_m,
            north_m=north_m,
            up_m=terrain.elevation_at(east_m, north_m) + config.eye_height_m,
        ),
        yaw_deg=_wrap_yaw(truth.yaw_deg + regime.yaw_steps * config.yaw_spacing_deg),
        pitch_deg=truth.pitch_deg,
        roll_deg=0.0,
    )


def build_synthetic_candidate_archive(
    terrain: TerrainMap,
    intrinsics: CameraIntrinsics,
    prior: CameraExtrinsics,
    observed_skylines: Mapping[str, NDArray[np.float64]],
    reference_rendered_depth_oracle: NDArray[np.float64],
    *,
    config: SyntheticSearchConfig | None = None,
) -> dict[str, Any]:
    """Build a candidate archive without accepting numeric pose truth.

    ``reference_rendered_depth_oracle`` is generated from the hidden synthetic
    reference pose.  Its scores are analysis-only ceilings and are named as such
    in every persisted field.
    """

    settings = config or SyntheticSearchConfig()
    profiles = _validated_profiles(observed_skylines, intrinsics)
    source_depth = _validated_depth(reference_rendered_depth_oracle, intrinsics)
    source_typed = extract_typed_outlines(source_depth, min_px=_typed_min_px(intrinsics.width_px))
    renderer = SyntheticRenderer()
    candidates: list[dict[str, Any]] = []
    position_offsets = range(-settings.position_radius_steps, settings.position_radius_steps + 1)
    yaw_offsets = range(-settings.yaw_radius_steps, settings.yaw_radius_steps + 1)

    for north_index, north_step in enumerate(position_offsets):
        north_m = prior.position.north_m + north_step * settings.position_spacing_m
        for east_index, east_step in enumerate(position_offsets):
            east_m = prior.position.east_m + east_step * settings.position_spacing_m
            if not _inside_terrain(terrain, east_m, north_m):
                continue
            up_m = terrain.elevation_at(east_m, north_m) + settings.eye_height_m
            for yaw_index, yaw_step in enumerate(yaw_offsets):
                candidate = CameraExtrinsics(
                    position=LocalPoint(east_m=east_m, north_m=north_m, up_m=up_m),
                    yaw_deg=_wrap_yaw(prior.yaw_deg + yaw_step * settings.yaw_spacing_deg),
                    pitch_deg=prior.pitch_deg,
                    roll_deg=0.0,
                )
                geometry = renderer.geometry(
                    terrain,
                    intrinsics,
                    candidate,
                    stride=settings.render_stride,
                )
                candidate_depth = np.asarray(geometry.forward_depth_m, dtype=np.float64)
                candidate_typed = extract_typed_outlines(
                    candidate_depth,
                    min_px=_typed_min_px(intrinsics.width_px),
                )
                depth_terms = _depth_score_terms(source_depth, candidate_depth, settings)
                typed_loss = _typed_outline_loss(source_typed, candidate_typed, settings.outline_distance_cap_px)
                skyline_scores = {
                    name: _skyline_loss(profile, geometry.skyline_profile, settings.skyline_residual_cap_px)
                    for name, profile in profiles.items()
                }
                fusion_scores = {
                    name: _round_score(
                        0.15 * skyline_score
                        + 0.55 * depth_terms["relative_log_depth_loss"]
                        + 0.20 * typed_loss
                        + 0.10 * depth_terms["overlap_loss"]
                    )
                    for name, skyline_score in skyline_scores.items()
                }
                candidates.append(
                    {
                        "candidate_id": f"n{north_index:02d}-e{east_index:02d}-y{yaw_index:02d}",
                        "grid": {
                            "east_step_from_prior": east_step,
                            "north_step_from_prior": north_step,
                            "yaw_step_from_prior": yaw_step,
                        },
                        "pose": candidate.model_dump(mode="json"),
                        "scores": {
                            "skyline": {name: _round_score(value) for name, value in skyline_scores.items()},
                            "oracle_metric_depth": depth_terms["metric_log_depth_loss"],
                            "oracle_relative_depth": depth_terms["relative_log_depth_loss"],
                            "oracle_typed_depth_outlines": _round_score(typed_loss),
                            "oracle_depth_overlap_loss": depth_terms["overlap_loss"],
                            "oracle_geometry_fusion": fusion_scores,
                        },
                        "oracle_depth_diagnostics": depth_terms,
                    }
                )
    if not candidates:
        raise RuntimeError("synthetic candidate search produced no in-bounds hypotheses")

    archive: dict[str, Any] = {
        "schema": SYNTHETIC_ARCHIVE_SCHEMA,
        "numeric_pose_reference_fields_used_directly_in_scoring": False,
        "reference_pose_derived_query_evidence_used": True,
        "supplied_prior_may_be_reference_derived": True,
        "supplied_prior_used": True,
        "config": settings.to_record(),
        "estimator_terrain": terrain_fingerprint(terrain),
        "query_geometry": {
            "projection": "pinhole",
            "intrinsics": intrinsics.model_dump(mode="json"),
        },
        "prior": prior.model_dump(mode="json"),
        "observation_tracks": {
            name: {
                "skyline_sha256": _array_sha256(profile),
                "finite_columns": int(np.isfinite(profile).sum()),
            }
            for name, profile in profiles.items()
        },
        "reference_rendered_depth_oracle": {
            "analysis_only": True,
            "production_eligible": False,
            "generated_from_exact_reference_pose": True,
            "numeric_pose_values_supplied_to_archive_builder": False,
            "sha256": _array_sha256(source_depth),
            "typed_outline_counts": source_typed.counts(),
        },
        "candidate_pool": {
            "generation": "square_position_grid_times_bounded_yaw_grid_around_supplied_prior",
            "numeric_truth_used_directly": False,
            "candidate_count": len(candidates),
        },
        "ranking_methods": _ranking_method_contract(),
        "candidates": candidates,
    }
    archive["archive_sha256"] = _canonical_sha256(archive)
    return archive


def evaluate_synthetic_candidate_archive(
    archive: Mapping[str, Any],
    truth: CameraExtrinsics,
    observation_quality: Mapping[str, Mapping[str, Any]],
    expected_observation_actions: Mapping[str, str],
    *,
    expected_pose_ambiguity: bool,
) -> dict[str, Any]:
    """Evaluate a frozen archive against synthetic truth after ranking."""

    record = _validated_archive(archive)
    settings = record["config"]
    target_position = float(settings["success_target"]["horizontal_position_m_lte"])
    target_yaw = float(settings["success_target"]["absolute_yaw_deg_lte"])
    candidates = list(record["candidates"])
    evaluated = [
        {
            "candidate": candidate,
            "errors": _pose_errors(candidate["pose"], truth),
        }
        for candidate in candidates
    ]
    for item in evaluated:
        errors = item["errors"]
        item["reaches_target"] = bool(
            errors["horizontal_position_m"] <= target_position and errors["yaw_deg"] <= target_yaw
        )
        item["normalized_joint_error"] = _round_score(
            math.hypot(errors["horizontal_position_m"] / target_position, errors["yaw_deg"] / target_yaw)
        )

    pool_oracle = min(
        evaluated,
        key=lambda item: (
            item["normalized_joint_error"],
            item["errors"]["position_3d_m"],
            item["candidate"]["candidate_id"],
        ),
    )
    target_count = sum(bool(item["reaches_target"]) for item in evaluated)
    proposal = {
        "evaluation_order": "before_any_candidate_ranking",
        "candidate_count": len(evaluated),
        "target_candidate_count": target_count,
        "full_pool_recall": 1.0 if target_count else 0.0,
        "reaches_target": bool(target_count),
        "best_reachable_candidate": _evaluated_candidate_record(pool_oracle),
    }

    tracks: dict[str, Any] = {}
    for track in record["observation_tracks"]:
        quality = dict(observation_quality.get(track, {}))
        expected_track_action = expected_observation_actions.get(track)
        if expected_track_action not in {"select", "abstain"}:
            raise ValueError(f"expected observation action for {track!r} must be 'select' or 'abstain'")
        method_results: dict[str, Any] = {}
        for method in TRACK_RANKING_METHODS:
            expected_abstain = expected_pose_ambiguity or (
                method == SKYLINE_METHOD and expected_track_action == "abstain"
            )
            method_results[method] = _evaluate_ranking_method(
                evaluated,
                method,
                track,
                quality,
                settings["abstention"],
                expected_abstain=expected_abstain,
            )
        tracks[track] = {
            "observation_quality": quality,
            "expected_observation_action": expected_track_action,
            "expected_observation_action_source": "predeclared_case_design",
            "methods": method_results,
        }

    score_track = next(iter(record["observation_tracks"]))
    oracle_depth_methods = {
        method: _evaluate_ranking_method(
            evaluated,
            method,
            score_track,
            {},
            settings["abstention"],
            expected_abstain=expected_pose_ambiguity,
        )
        for method in TRACK_INVARIANT_ORACLE_METHODS
    }

    return {
        "schema": SYNTHETIC_EVALUATION_SCHEMA,
        "archive_sha256": record["archive_sha256"],
        "reference_data_used": True,
        "numeric_truth_fields_used_by_archive_builder": False,
        "reference_derived_evidence_used_by_archive_builder": True,
        "expected_pose_ambiguity": expected_pose_ambiguity,
        "truth": truth.model_dump(mode="json"),
        "proposal": proposal,
        "tracks": tracks,
        "oracle_depth_methods": {
            "track_invariant": True,
            "reported_once_per_candidate_archive": True,
            "methods": oracle_depth_methods,
        },
    }


def extraction_quality(
    observed: NDArray[np.float64],
    oracle: NDArray[np.float64],
    *,
    extractor_name: str,
    coverage: float,
    agreement: float,
    config: SyntheticSearchConfig,
) -> dict[str, Any]:
    """Post-hoc extraction metrics plus a truth-free acceptance decision."""

    observed_profile = np.asarray(observed, dtype=np.float64)
    oracle_profile = np.asarray(oracle, dtype=np.float64)
    if observed_profile.shape != oracle_profile.shape:
        raise ValueError("observed and oracle skyline profiles must share a shape")
    common = np.isfinite(observed_profile) & np.isfinite(oracle_profile)
    residual = np.abs(observed_profile[common] - oracle_profile[common])
    accepted = coverage >= config.minimum_extraction_coverage and agreement >= config.minimum_extraction_agreement
    return {
        "extractor": extractor_name,
        "coverage": _round_score(coverage),
        "agreement": _round_score(agreement),
        "accepted": bool(accepted),
        "acceptance_uses_truth": False,
        "evaluation": {
            "reference_mask_used": True,
            "common_columns": int(common.sum()),
            "mae_px": _round_score(float(np.mean(residual))) if residual.size else None,
            "p95_px": _round_score(float(np.percentile(residual, 95.0))) if residual.size else None,
        },
    }


def haze_image(rgb: NDArray[np.uint8], seed: int) -> NDArray[np.uint8]:
    """Create a deterministic low-contrast/cloud nuisance without pose truth."""

    image = np.asarray(rgb, dtype=np.float64) / 255.0
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("synthetic haze input must be RGB")
    height, width = image.shape[:2]
    blurred = gaussian_filter(image, sigma=(1.6, 1.6, 0.0))
    haze_colour = np.asarray([0.76, 0.79, 0.81], dtype=np.float64)
    result = 0.38 * blurred + 0.62 * haze_colour
    yy, xx = np.indices((height, width), dtype=np.float64)
    rng = np.random.default_rng(seed)
    cloud = np.zeros((height, width), dtype=np.float64)
    for _ in range(7):
        center_x = rng.uniform(0.0, max(width - 1, 1))
        center_y = rng.uniform(0.08 * height, 0.62 * height)
        radius_x = rng.uniform(0.06 * width, 0.18 * width)
        radius_y = rng.uniform(0.04 * height, 0.12 * height)
        cloud = np.maximum(
            cloud,
            np.exp(-0.5 * (((xx - center_x) / radius_x) ** 2 + ((yy - center_y) / radius_y) ** 2)),
        )
    result = result * (1.0 - 0.20 * cloud[..., None]) + 0.96 * 0.20 * cloud[..., None]
    return np.clip(np.rint(result * 255.0), 0.0, 255.0).astype(np.uint8)


def aggregate_synthetic_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate proposal, extraction, ranking, and abstention independently."""

    result = _aggregate_case_set(cases)
    variants = sorted({str(case.get("estimator_terrain", {}).get("variant", "unspecified")) for case in cases})
    result["estimator_terrain_strata"] = {
        variant: _aggregate_case_set(
            [case for case in cases if str(case.get("estimator_terrain", {}).get("variant", "unspecified")) == variant]
        )
        for variant in variants
    }
    return result


def _aggregate_case_set(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one already-selected case stratum without recursive strata."""

    proposal_rows = [case["evaluation"]["proposal"] for case in cases]
    extraction: dict[str, list[dict[str, Any]]] = {}
    methods: dict[tuple[str, str], list[dict[str, Any]]] = {}
    unique_observations: dict[str, dict[str, Any]] = {}
    for case in cases:
        observation_id = str(case.get("observation_id", case["case_id"]))
        unique_observations.setdefault(observation_id, case)
        for track, track_record in case["evaluation"]["tracks"].items():
            for method, method_record in track_record["methods"].items():
                methods.setdefault((track, method), []).append(method_record)
        for method, method_record in case["evaluation"]["oracle_depth_methods"]["methods"].items():
            methods.setdefault(("reference_depth_oracle", method), []).append(method_record)
    for case in unique_observations.values():
        for track, track_record in case["evaluation"]["tracks"].items():
            extraction.setdefault(track, []).append(track_record["observation_quality"])
    return {
        "proposal_recall": {
            "cases": len(proposal_rows),
            "reached": sum(bool(row["reaches_target"]) for row in proposal_rows),
            "rate": _rate(sum(bool(row["reaches_target"]) for row in proposal_rows), len(proposal_rows)),
        },
        "observation_extraction": [
            {
                "track": track,
                "unique_observations": len(rows),
                "accepted": sum(bool(row.get("accepted")) for row in rows),
                "acceptance_rate": _rate(sum(bool(row.get("accepted")) for row in rows), len(rows)),
                "median_mae_px": _median_optional(row.get("evaluation", {}).get("mae_px") for row in rows),
            }
            for track, rows in sorted(extraction.items())
        ],
        "candidate_ranking": [
            {
                "track": track,
                "method": method,
                "evidence_role": rows[0]["evidence_role"],
                "production_eligible": rows[0]["production_eligible"],
                "cases": len(rows),
                "raw_top1_target_hits": sum(bool(row["raw_top1_reaches_target"]) for row in rows),
                "raw_top1_target_hit_rate": _rate(
                    sum(bool(row["raw_top1_reaches_target"]) for row in rows),
                    len(rows),
                ),
                "selections": sum(bool(row["selected"]) for row in rows),
                "selection_coverage": _rate(sum(bool(row["selected"]) for row in rows), len(rows)),
                "selected_pose_target_hits": sum(bool(row["selected_pose_reaches_target"]) for row in rows),
                "selected_pose_target_hit_rate": _rate(
                    sum(bool(row["selected_pose_reaches_target"]) for row in rows),
                    len(rows),
                ),
                "false_accepts": sum(bool(row["false_accept"]) for row in rows),
                "accepted_selection_accuracy": _rate(
                    sum(bool(row["selected_pose_reaches_target"]) for row in rows),
                    sum(bool(row["selected"]) for row in rows),
                ),
                "correct_decisions": sum(bool(row["decision_correct"]) for row in rows),
                "decision_accuracy": _rate(sum(bool(row["decision_correct"]) for row in rows), len(rows)),
                "abstentions": sum(bool(row["decision"]["abstained"]) for row in rows),
                "median_first_target_rank": _median_optional(row["first_target_rank"] for row in rows),
                "median_winner_position_error_m": _median_optional(
                    row["winner"]["errors"]["horizontal_position_m"] for row in rows
                ),
                "median_winner_yaw_error_deg": _median_optional(row["winner"]["errors"]["yaw_deg"] for row in rows),
            }
            for (track, method), rows in sorted(methods.items())
        ],
    }


def canonical_json_bytes(value: Any) -> bytes:
    """Encode strict, stable JSON for artifact hashes."""

    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _ranking_method_contract() -> dict[str, Any]:
    return {
        SKYLINE_METHOD: {
            "evidence": "custom pinhole shared-renderer query skyline only",
            "analysis_only": True,
            "production_eligible": False,
        },
        ORACLE_METRIC_DEPTH_METHOD: {
            "evidence": "exact metric depth rendered from synthetic reference pose",
            "analysis_only": True,
            "production_eligible": False,
        },
        ORACLE_RELATIVE_DEPTH_METHOD: {
            "evidence": "median-scale-aligned log depth rendered from synthetic reference pose",
            "analysis_only": True,
            "production_eligible": False,
        },
        ORACLE_TYPED_DEPTH_METHOD: {
            "evidence": "typed outlines extracted from reference-rendered exact depth",
            "analysis_only": True,
            "production_eligible": False,
        },
        ORACLE_GEOMETRY_FUSION_METHOD: {
            "evidence": "fixed fusion containing reference-rendered exact depth",
            "analysis_only": True,
            "production_eligible": False,
        },
    }


def _validated_profiles(
    profiles: Mapping[str, NDArray[np.float64]],
    intrinsics: CameraIntrinsics,
) -> dict[str, NDArray[np.float64]]:
    if not profiles:
        raise ValueError("at least one observed skyline track is required")
    result: dict[str, NDArray[np.float64]] = {}
    for name, profile in profiles.items():
        if not name:
            raise ValueError("observed skyline track names cannot be empty")
        rows = np.asarray(profile, dtype=np.float64)
        if rows.shape != (intrinsics.width_px,):
            raise ValueError(f"observed skyline {name!r} must have shape {(intrinsics.width_px,)}")
        if np.any(np.isinf(rows)):
            raise ValueError(f"observed skyline {name!r} contains infinity")
        result[name] = rows
    return result


def _validated_depth(depth: NDArray[np.float64], intrinsics: CameraIntrinsics) -> NDArray[np.float64]:
    source = np.asarray(depth, dtype=np.float64)
    expected = (intrinsics.height_px, intrinsics.width_px)
    if source.shape != expected:
        raise ValueError(f"reference-rendered depth must have shape {expected}")
    finite = np.isfinite(source)
    if np.any(finite & (source <= 0.0)) or np.any(np.isinf(source)):
        raise ValueError("reference-rendered depth must contain positive finite terrain depth or NaN sky")
    return source


def _depth_score_terms(
    source: NDArray[np.float64],
    candidate: NDArray[np.float64],
    config: SyntheticSearchConfig,
) -> dict[str, Any]:
    source_valid = np.isfinite(source) & (source > 0.0)
    candidate_valid = np.isfinite(candidate) & (candidate > 0.0)
    intersection = source_valid & candidate_valid
    source_count = int(source_valid.sum())
    candidate_count = int(candidate_valid.sum())
    common_count = int(intersection.sum())
    overlap_f1 = 2.0 * common_count / max(source_count + candidate_count, 1)
    overlap_loss = 1.0 - overlap_f1
    if common_count:
        source_log = np.log(source[intersection])
        candidate_log = np.log(candidate[intersection])
        residual = candidate_log - source_log
        metric_loss = float(np.mean(np.minimum(np.abs(residual), config.depth_log_cap)) / config.depth_log_cap)
        scale_shift = float(np.median(residual))
        relative_loss = float(
            np.mean(np.minimum(np.abs(residual - scale_shift), config.depth_log_cap)) / config.depth_log_cap
        )
    else:
        metric_loss = 1.0
        relative_loss = 1.0
        scale_shift = None
    return {
        "metric_log_depth_loss": _round_score(0.85 * metric_loss + 0.15 * overlap_loss),
        "relative_log_depth_loss": _round_score(0.85 * relative_loss + 0.15 * overlap_loss),
        "overlap_loss": _round_score(overlap_loss),
        "overlap_f1": _round_score(overlap_f1),
        "common_depth_pixels": common_count,
        "source_depth_pixels": source_count,
        "candidate_depth_pixels": candidate_count,
        "candidate_to_source_log_scale_shift": _round_score(scale_shift) if scale_shift is not None else None,
    }


def _typed_outline_loss(source: TypedOutlines, candidate: TypedOutlines, cap_px: float) -> float:
    source_union = source.occlusion | source.rib | source.couloir
    candidate_union = candidate.occlusion | candidate.rib | candidate.couloir
    union_loss = _symmetric_mask_chamfer(source_union, candidate_union, cap_px)
    family_losses: list[float] = []
    for source_family, candidate_family in (
        (source.occlusion, candidate.occlusion),
        (source.rib, candidate.rib),
        (source.couloir, candidate.couloir),
    ):
        if np.any(source_family) or np.any(candidate_family):
            family_losses.append(_symmetric_mask_chamfer(source_family, candidate_family, cap_px))
    family_loss = float(np.mean(family_losses)) if family_losses else 0.0
    return 0.5 * union_loss + 0.5 * family_loss


def _symmetric_mask_chamfer(source: NDArray[np.bool_], candidate: NDArray[np.bool_], cap_px: float) -> float:
    source_mask = np.asarray(source, dtype=np.bool_)
    candidate_mask = np.asarray(candidate, dtype=np.bool_)
    if source_mask.shape != candidate_mask.shape:
        raise ValueError("typed outline masks must share a shape")
    source_count = int(source_mask.sum())
    candidate_count = int(candidate_mask.sum())
    if not source_count and not candidate_count:
        return 0.0
    if not source_count or not candidate_count:
        return 1.0
    source_distance = distance_transform_edt(~source_mask)
    candidate_distance = distance_transform_edt(~candidate_mask)
    source_to_candidate = np.minimum(candidate_distance[source_mask], cap_px) / cap_px
    candidate_to_source = np.minimum(source_distance[candidate_mask], cap_px) / cap_px
    return float(0.5 * (np.mean(source_to_candidate) + np.mean(candidate_to_source)))


def _skyline_loss(observed: NDArray[np.float64], predicted: NDArray[np.float64], cap_px: float) -> float:
    observed_rows = np.asarray(observed, dtype=np.float64)
    predicted_rows = np.asarray(predicted, dtype=np.float64)
    valid_observed = np.isfinite(observed_rows)
    common = valid_observed & np.isfinite(predicted_rows)
    if not np.any(valid_observed) or not np.any(common):
        return 1.0
    residual = np.minimum(np.abs(observed_rows[common] - predicted_rows[common]), cap_px) / cap_px
    coverage = common.sum() / valid_observed.sum()
    return float(0.90 * np.mean(residual) + 0.10 * (1.0 - coverage))


def _method_score(candidate: Mapping[str, Any], method: str, track: str) -> float:
    scores = candidate["scores"]
    if method == SKYLINE_METHOD:
        return float(scores["skyline"][track])
    if method == ORACLE_GEOMETRY_FUSION_METHOD:
        return float(scores["oracle_geometry_fusion"][track])
    return float(scores[method])


def _evaluate_ranking_method(
    evaluated: list[dict[str, Any]],
    method: str,
    track: str,
    observation_quality: Mapping[str, Any],
    abstention_contract: Mapping[str, Any],
    *,
    expected_abstain: bool,
) -> dict[str, Any]:
    ranked = sorted(
        evaluated,
        key=lambda item: (
            _method_score(item["candidate"], method, track),
            item["candidate"]["candidate_id"],
        ),
    )
    first_target_rank = next(
        (rank for rank, item in enumerate(ranked, start=1) if item["reaches_target"]),
        None,
    )
    decision = _abstention_decision(
        ranked,
        method,
        track,
        observation_quality,
        abstention_contract,
    )
    winner = ranked[0]
    raw_top1_reaches_target = bool(winner["reaches_target"])
    selected = not decision["abstained"]
    selected_pose_reaches_target = raw_top1_reaches_target and selected
    decision_correct = decision["abstained"] if expected_abstain else selected_pose_reaches_target
    return {
        "evidence_role": (
            "oracle_only_reference_pose_generated"
            if method in ORACLE_ONLY_METHODS
            else "custom_pinhole_query_skyline_stage_harness"
        ),
        "production_eligible": False,
        "winner": _evaluated_candidate_record(winner),
        "winner_score": _round_score(_method_score(winner["candidate"], method, track)),
        "first_target_rank": first_target_rank,
        "top_k_recall": {
            str(k): bool(first_target_rank is not None and first_target_rank <= k) for k in (1, 5, 10, 25, 50, 100)
        },
        "raw_top1_reaches_target": raw_top1_reaches_target,
        "selected": selected,
        "selected_pose_reaches_target": selected_pose_reaches_target,
        "false_accept": selected and (expected_abstain or not raw_top1_reaches_target),
        "decision": decision,
        "expected_action": "abstain" if expected_abstain else "select",
        "expected_action_source": "predeclared_case_design",
        "decision_correct": bool(decision_correct),
    }


def _abstention_decision(
    ranked: list[dict[str, Any]],
    method: str,
    track: str,
    observation_quality: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    if method == SKYLINE_METHOD and not bool(observation_quality.get("accepted", False)):
        reasons.append("observation_quality_rejected")
    best_score = _method_score(ranked[0]["candidate"], method, track)
    delta = float(contract["score_delta_lte"])
    competitive = [item for item in ranked if _method_score(item["candidate"], method, track) <= best_score + delta]
    max_position_separation = 0.0
    max_yaw_separation = 0.0
    for left_index, left in enumerate(competitive):
        left_pose = left["candidate"]["pose"]
        for right in competitive[left_index + 1 :]:
            right_pose = right["candidate"]["pose"]
            dx = float(left_pose["position"]["east_m"]) - float(right_pose["position"]["east_m"])
            dy = float(left_pose["position"]["north_m"]) - float(right_pose["position"]["north_m"])
            max_position_separation = max(max_position_separation, math.hypot(dx, dy))
            max_yaw_separation = max(
                max_yaw_separation,
                _angle_error(float(left_pose["yaw_deg"]), float(right_pose["yaw_deg"])),
            )
    pose_ambiguous = len(competitive) > 1 and (
        max_position_separation >= float(contract["competitive_position_separation_m_gte"])
        or max_yaw_separation >= float(contract["competitive_yaw_separation_deg_gte"])
    )
    if pose_ambiguous:
        reasons.append("competitive_pose_modes")
    return {
        "abstained": bool(reasons),
        "reasons": reasons,
        "competitive_candidate_count": len(competitive),
        "competitive_max_position_separation_m": _round_score(max_position_separation),
        "competitive_max_yaw_separation_deg": _round_score(max_yaw_separation),
        "best_score": _round_score(best_score),
    }


def _pose_errors(pose_record: Mapping[str, Any], truth: CameraExtrinsics) -> dict[str, float]:
    position = pose_record["position"]
    dx = float(position["east_m"]) - truth.position.east_m
    dy = float(position["north_m"]) - truth.position.north_m
    dz = float(position["up_m"]) - truth.position.up_m
    return {
        "horizontal_position_m": _round_score(math.hypot(dx, dy)),
        "vertical_m": _round_score(abs(dz)),
        "position_3d_m": _round_score(math.sqrt(dx * dx + dy * dy + dz * dz)),
        "yaw_deg": _round_score(_angle_error(float(pose_record["yaw_deg"]), truth.yaw_deg)),
        "pitch_deg": _round_score(abs(float(pose_record["pitch_deg"]) - truth.pitch_deg)),
    }


def _evaluated_candidate_record(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": item["candidate"]["candidate_id"],
        "pose": item["candidate"]["pose"],
        "errors": item["errors"],
        "normalized_joint_error": item["normalized_joint_error"],
        "reaches_target": item["reaches_target"],
    }


def _validated_archive(archive: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(archive)
    if record.get("schema") != SYNTHETIC_ARCHIVE_SCHEMA:
        raise ValueError("unsupported synthetic candidate archive schema")
    supplied = record.get("archive_sha256")
    unhashed = dict(record)
    unhashed.pop("archive_sha256", None)
    if supplied != _canonical_sha256(unhashed):
        raise ValueError("synthetic candidate archive SHA-256 does not match its content")
    if record.get("numeric_pose_reference_fields_used_directly_in_scoring") is not False:
        raise ValueError("synthetic candidate archive scoring must not receive numeric truth fields directly")
    return record


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _array_sha256(array: NDArray[np.generic]) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode())
    digest.update(values.dtype.str.encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _typed_min_px(width: int) -> int:
    return max(3, width // 32)


def _inside_terrain(terrain: TerrainMap, east_m: float, north_m: float) -> bool:
    return bool(
        float(terrain.x_m[0]) <= east_m <= float(terrain.x_m[-1])
        and float(terrain.y_m[0]) <= north_m <= float(terrain.y_m[-1])
    )


def _validate_inside_terrain(terrain: TerrainMap, east_m: float, north_m: float, label: str) -> None:
    if not _inside_terrain(terrain, east_m, north_m):
        raise ValueError(f"{label} lies outside the synthetic terrain")


def _wrap_yaw(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _angle_error(left: float, right: float) -> float:
    return abs(_wrap_yaw(left - right))


def _round_score(value: float) -> float:
    return round(float(value), 8)


def _rate(numerator: int, denominator: int) -> float | None:
    return _round_score(numerator / denominator) if denominator else None


def _median_optional(values) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return _round_score(float(np.median(finite))) if finite else None

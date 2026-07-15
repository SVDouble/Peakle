"""Run the synthetic stage ceiling with shared or independent query rendering."""

from __future__ import annotations

import argparse
import hashlib
import math
import platform
import subprocess
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from peakle.config import AppSettings, load_settings
from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics
from peakle.domain.coordinates import EARTH_RADIUS_M, LocalPoint
from peakle.domain.terrain import TerrainMap
from peakle.io.artifacts import fsync_directory as _fsync_directory
from peakle.io.artifacts import publish_directory_once as _publish_directory_once
from peakle.io.artifacts import write_once_bytes as _write_once
from peakle.localize.paths import BASE
from peakle.localize.synthetic_pipeline_bench import (
    DEFAULT_PRIOR_REGIMES,
    SYNTHETIC_BENCHMARK_SCHEMA,
    SyntheticPriorRegime,
    SyntheticSearchConfig,
    aggregate_synthetic_cases,
    build_synthetic_candidate_archive,
    canonical_json_bytes,
    controlled_prior,
    evaluate_synthetic_candidate_archive,
)
from peakle.rendering.terrain_view import terrain_fingerprint
from peakle.research.synthetic_estimator import ImmutableArtifact, run_synthetic_estimator
from peakle.research.synthetic_query import (
    build_synthetic_query_observations,
    observation_track_contract,
    renderer_contract,
)
from peakle.scene.state import place_cameras
from peakle.terrain.generator import TerrainGenerator
from peakle.terrain.peak_detection import PeakDetector


def main() -> None:
    args = _parser().parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing artifact directory: {output}")
    seeds = _csv_ints(args.seeds, "terrain seed")
    prior_regimes = _prior_regimes(args.prior_regimes)
    estimator_terrain_variants = _csv_choices(
        args.estimator_terrain_variants,
        ("exact", "coarse"),
        "estimator terrain variant",
    )
    if args.coarse_factor < 2:
        raise SystemExit("--coarse-factor must be at least 2")
    config = SyntheticSearchConfig(
        position_spacing_m=args.position_spacing_m,
        position_radius_steps=args.position_radius_steps,
        yaw_spacing_deg=args.yaw_spacing_deg,
        yaw_radius_steps=args.yaw_radius_steps,
        eye_height_m=args.eye_height_m,
        render_stride=args.render_stride,
        ambiguity_score_delta=args.ambiguity_score_delta,
    )
    intrinsics = CameraIntrinsics.from_horizontal_fov(
        args.image_width,
        args.image_height,
        args.horizontal_fov_deg,
    )
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    settings = load_settings()
    cases: list[dict[str, Any]] = []
    observation_records: dict[str, dict[str, Any]] = {}
    artifact_files: dict[str, bytes] = {}
    query_artifact_manifest: list[dict[str, Any]] = []
    estimator_artifact_manifest: dict[str, dict[str, Any]] = {}
    scenes = _rugged_scenes(
        seeds,
        views_per_scene=args.views_per_scene,
        terrain_width_m=args.terrain_width_m,
        terrain_height_m=args.terrain_height_m,
        terrain_grid_width=args.terrain_grid_width,
        terrain_grid_height=args.terrain_grid_height,
        eye_height_m=config.eye_height_m,
        settings=settings,
    )
    if not args.no_radial_control:
        scenes.append(
            _radial_control_scene(
                terrain_width_m=args.terrain_width_m,
                terrain_height_m=args.terrain_height_m,
                terrain_grid_width=args.terrain_grid_width,
                terrain_grid_height=args.terrain_grid_height,
                eye_height_m=config.eye_height_m,
                settings=settings,
            )
        )
    total = len(scenes) * len(prior_regimes) * len(estimator_terrain_variants)
    print(
        f"Synthetic {args.query_renderer} query stage ceiling: {len(scenes)} scene/view(s), "
        f"{len(prior_regimes)} prior regime(s), {len(estimator_terrain_variants)} estimator terrain variant(s), "
        f"{total} candidate archives",
        flush=True,
    )
    for scene_index, scene in enumerate(scenes, start=1):
        observations = build_synthetic_query_observations(
            scene,
            intrinsics,
            config,
            query_renderer=args.query_renderer,
            chromium_path=args.chromium_path,
        )
        for filename, content in observations.artifact_files.items():
            if filename in artifact_files:
                raise RuntimeError(f"duplicate frozen query artifact filename: {filename}")
            artifact_files[filename] = content
        query_artifact_manifest.extend(record.model_dump(mode="json") for record in observations.artifact_manifest)
        observation_records[str(scene["scene_id"])] = {
            "observation_id": scene["scene_id"],
            "query_renderer": args.query_renderer,
            "query_provenance": observations.query_provenance,
            "tracks": observations.metadata,
            "query_artifacts": [record.model_dump(mode="json") for record in observations.artifact_manifest],
        }
        semantic_record = next(
            (record for record in observations.artifact_manifest if record.role == "semantic_u8"),
            None,
        )
        geometry_record = next(
            (record for record in observations.artifact_manifest if record.role == "normal_xyz_depth_f32le"),
            None,
        )
        for variant in estimator_terrain_variants:
            estimator_terrain, terrain_record = _estimator_terrain(
                scene["terrain"],
                scene["truth"],
                variant=variant,
                coarse_factor=args.coarse_factor,
                query_renderer=args.query_renderer,
            )
            for regime in prior_regimes:
                case_started = time.perf_counter()
                case_id = f"{scene['scene_id']}--{variant}--{regime.name}"
                prior = controlled_prior(estimator_terrain, scene["truth"], regime, config)
                estimator_transaction: dict[str, Any] | None = None
                if args.query_renderer == "webgl":
                    if semantic_record is None or geometry_record is None:
                        raise RuntimeError("WebGL estimator requires frozen semantic and geometry query artifacts")
                    estimator_run = run_synthetic_estimator(
                        estimator_terrain,
                        intrinsics,
                        prior,
                        config,
                        semantic_u8=artifact_files[semantic_record.filename],
                        geometry_f32le=artifact_files[geometry_record.filename],
                        terrain_filename=f"estimator-terrain-{scene['scene_id']}-{variant}.json",
                        semantic_filename=semantic_record.filename,
                        geometry_filename=geometry_record.filename,
                        request_filename=f"estimator-request-{case_id}.json",
                        output_filename=f"candidate-archive-{case_id}.json",
                    )
                    archive = estimator_run.archive
                    for role, artifact in (
                        ("estimator_terrain", estimator_run.terrain_artifact),
                        ("estimator_request", estimator_run.request_artifact),
                        ("frozen_candidate_archive", estimator_run.archive_artifact),
                    ):
                        _register_estimator_artifact(
                            artifact_files,
                            estimator_artifact_manifest,
                            artifact,
                            role=role,
                            case_id=case_id,
                        )
                    estimator_transaction = {
                        "execution": "isolated_python_subprocess",
                        "truth_fields_in_request": False,
                        "request": estimator_run.request_artifact.ref.model_dump(mode="json"),
                        "estimator_terrain": estimator_run.terrain_artifact.ref.model_dump(mode="json"),
                        "candidate_archive": estimator_run.archive_artifact.ref.model_dump(mode="json"),
                    }
                else:
                    archive = build_synthetic_candidate_archive(
                        estimator_terrain,
                        intrinsics,
                        prior,
                        observations.profiles,
                        observations.reference_depth_oracle,
                        config=config,
                    )
                evaluation = evaluate_synthetic_candidate_archive(
                    archive,
                    scene["truth"],
                    observations.quality,
                    observations.expected_actions,
                    expected_pose_ambiguity=bool(scene["expected_pose_ambiguity"]),
                )
                cases.append(
                    {
                        "case_id": case_id,
                        "observation_id": scene["scene_id"],
                        "scene": {
                            "scene_id": scene["scene_id"],
                            "kind": scene["kind"],
                            "terrain_seed": scene["terrain_seed"],
                            "view_index": scene["view_index"],
                            "expected_pose_ambiguity": scene["expected_pose_ambiguity"],
                            "authoritative_terrain": terrain_fingerprint(scene["terrain"]),
                        },
                        "estimator_terrain": terrain_record,
                        "prior_regime": regime.to_record(config),
                        "observation_tracks": observations.metadata,
                        "archive": archive,
                        "estimator_transaction": estimator_transaction,
                        "evaluation": evaluation,
                        "runtime_s": round(time.perf_counter() - case_started, 4),
                    }
                )
                print(
                    f"[{scene_index}/{len(scenes)}] {scene['scene_id']} / {variant} / {regime.name}: "
                    f"{cases[-1]['runtime_s']:.1f}s",
                    flush=True,
                )
    finished_at = datetime.now(UTC)
    results = {
        "schema": SYNTHETIC_BENCHMARK_SCHEMA,
        "run_id": output.name,
        "contract": {
            "truth_class": "independent_synthetic" if args.query_renderer == "webgl" else "diagnostic_oracle",
            "independence_class": (
                "independent_rasterizer_same_scene_model"
                if args.query_renderer == "webgl"
                else "shared_python_renderer_inverse_crime"
            ),
            "purpose": ("custom pinhole upper-bound stage harness; never a replacement for the real-photo benchmark"),
            "production_atlas_scope": {
                "production_pipeline_coverage": False,
                "build_skyline_atlas_called": False,
                "cyltan_raycast_path_tested": False,
                "claim": "does not directly validate the production skyline-atlas implementation",
                "untested_stages": [
                    "production cyltan query geometry",
                    "HorizonProfile/raycast skyline-atlas generation",
                    "photo-to-render learned correspondence matching",
                    "render-match lifting and PnP",
                    "candidate holdout validation",
                    "continuous local pose refinement",
                ],
            },
            "truth_separation": (
                "query RGB/mask/depth and controlled priors are reference-derived; candidate scoring receives "
                "those declared inputs but not numeric truth fields, and archives are frozen before post-hoc "
                "pose-error evaluation"
            ),
            "proposal_recall_evaluated_before_ranking": True,
            "reference_rendered_depth": {
                "role": "oracle_only_reference_pose_generated",
                "production_eligible": False,
                "must_not_be_called_photo_or_pfm_evidence": True,
            },
            "renderer_family": renderer_contract(args.query_renderer),
            "terrain_mismatch_axis": (
                "coarse deterministically subsamples the authoritative grid before candidate rendering"
            ),
            "expected_ambiguity_control": "radially symmetric terrain must cause pose-mode abstention",
        },
        "config": {
            "query_renderer": args.query_renderer,
            "search": config.to_record(),
            "terrain_seeds": list(seeds),
            "views_per_rugged_scene": args.views_per_scene,
            "radial_control_included": not args.no_radial_control,
            "prior_regimes": [regime.to_record(config) for regime in prior_regimes],
            "estimator_terrain_variants": list(estimator_terrain_variants),
            "coarse_factor": args.coarse_factor,
            "image": intrinsics.model_dump(mode="json"),
            "terrain": {
                "width_m": args.terrain_width_m,
                "height_m": args.terrain_height_m,
                "grid_width": args.terrain_grid_width,
                "grid_height": args.terrain_grid_height,
            },
            "observation_tracks": observation_track_contract(args.query_renderer),
            "scene_generation_settings": settings.model_dump(mode="json"),
        },
        "observations": observation_records,
        "query_artifacts": {
            "frozen_before_estimator": args.query_renderer == "webgl",
            "file_count": len(query_artifact_manifest),
            "files": query_artifact_manifest,
        },
        "estimator_artifacts": {
            "subprocess_boundary_used": args.query_renderer == "webgl",
            "file_count": len(estimator_artifact_manifest),
            "files": list(estimator_artifact_manifest.values()),
        },
        "cases": cases,
        "aggregates": aggregate_synthetic_cases(cases),
        "runtime_s": round(time.perf_counter() - started, 4),
    }
    summary = _summary_markdown(results)
    _commit_artifact(
        output,
        results,
        summary,
        started_at=started_at,
        finished_at=finished_at,
        query_renderer=args.query_renderer,
        extra_files=artifact_files,
    )
    print(f"Wrote {output}", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(_default_output_dir()))
    parser.add_argument("--seeds", default="7,23")
    parser.add_argument("--views-per-scene", type=int, default=2)
    parser.add_argument("--prior-regimes", default="exact,wide")
    parser.add_argument("--estimator-terrain-variants", default="exact,coarse")
    parser.add_argument("--coarse-factor", type=int, default=2)
    parser.add_argument("--query-renderer", choices=("shared-python", "webgl"), default="shared-python")
    parser.add_argument("--chromium-path", help="optional Chromium executable for --query-renderer webgl")
    parser.add_argument("--no-radial-control", action="store_true")
    parser.add_argument("--image-width", type=int, default=160)
    parser.add_argument("--image-height", type=int, default=90)
    parser.add_argument("--horizontal-fov-deg", type=float, default=55.0)
    parser.add_argument("--terrain-width-m", type=float, default=14_000.0)
    parser.add_argument("--terrain-height-m", type=float, default=10_000.0)
    parser.add_argument("--terrain-grid-width", type=int, default=97)
    parser.add_argument("--terrain-grid-height", type=int, default=73)
    parser.add_argument("--position-spacing-m", type=float, default=100.0)
    parser.add_argument("--position-radius-steps", type=int, default=2)
    parser.add_argument("--yaw-spacing-deg", type=float, default=5.0)
    parser.add_argument("--yaw-radius-steps", type=int, default=3)
    parser.add_argument("--eye-height-m", type=float, default=2.5)
    parser.add_argument("--render-stride", type=int, default=2)
    parser.add_argument("--ambiguity-score-delta", type=float, default=0.0025)
    return parser


def _rugged_scenes(
    seeds: tuple[int, ...],
    *,
    views_per_scene: int,
    terrain_width_m: float,
    terrain_height_m: float,
    terrain_grid_width: int,
    terrain_grid_height: int,
    eye_height_m: float,
    settings: AppSettings | None = None,
) -> list[dict[str, Any]]:
    if views_per_scene < 1:
        raise SystemExit("--views-per-scene must be positive")
    settings = settings or load_settings()
    scenes: list[dict[str, Any]] = []
    for seed in seeds:
        spec = settings.terrain.model_copy(
            update={
                "seed": seed,
                "width_m": terrain_width_m,
                "height_m": terrain_height_m,
                "grid_width": terrain_grid_width,
                "grid_height": terrain_grid_height,
            }
        )
        terrain = TerrainGenerator(spec).generate()
        peaks = PeakDetector(settings.peak_detection).detect(terrain)
        if not peaks:
            raise RuntimeError(f"rugged synthetic terrain seed {seed} produced no peaks")
        cameras = place_cameras(
            terrain,
            peaks,
            settings.camera.model_copy(
                update={
                    "view_count": views_per_scene,
                    "overlook_height_m": eye_height_m,
                }
            ),
        )
        for index, truth in enumerate(cameras):
            scenes.append(
                {
                    "scene_id": f"rugged-s{seed}-v{index + 1:02d}",
                    "kind": "rugged_generated",
                    "terrain_seed": seed,
                    "view_index": index,
                    "terrain": terrain,
                    "truth": truth,
                    "expected_pose_ambiguity": False,
                }
            )
    return scenes


def _radial_control_scene(
    *,
    terrain_width_m: float,
    terrain_height_m: float,
    terrain_grid_width: int,
    terrain_grid_height: int,
    eye_height_m: float,
    settings: AppSettings | None = None,
) -> dict[str, Any]:
    settings = settings or load_settings()
    spec = settings.terrain.model_copy(
        update={
            "seed": 0,
            "width_m": terrain_width_m,
            "height_m": terrain_height_m,
            "grid_width": terrain_grid_width,
            "grid_height": terrain_grid_height,
            "min_elevation_m": 1000.0,
            "max_elevation_m": 2200.0,
        }
    )
    x_m = np.linspace(-terrain_width_m / 2.0, terrain_width_m / 2.0, terrain_grid_width)
    y_m = np.linspace(-terrain_height_m / 2.0, terrain_height_m / 2.0, terrain_grid_height)
    xx, yy = np.meshgrid(x_m, y_m)
    radius = np.hypot(xx, yy)
    elevation = 1000.0 + np.clip((radius - 700.0) * 0.28, 0.0, 1200.0)
    lat = spec.origin.latitude_deg + np.degrees(yy / EARTH_RADIUS_M)
    lon = spec.origin.longitude_deg + np.degrees(
        xx / (EARTH_RADIUS_M * math.cos(math.radians(spec.origin.latitude_deg)))
    )
    terrain = TerrainMap(
        spec=spec,
        x_m=x_m.astype(np.float64),
        y_m=y_m.astype(np.float64),
        elevation_m=elevation.astype(np.float64),
        latitude_deg=lat.astype(np.float64),
        longitude_deg=lon.astype(np.float64),
    )
    truth = CameraExtrinsics(
        position=LocalPoint(east_m=0.0, north_m=0.0, up_m=terrain.elevation_at(0.0, 0.0) + eye_height_m),
        yaw_deg=0.0,
        pitch_deg=4.0,
        roll_deg=0.0,
    )
    return {
        "scene_id": "radial-ambiguity-control",
        "kind": "radially_symmetric_negative_control",
        "terrain_seed": None,
        "view_index": 0,
        "terrain": terrain,
        "truth": truth,
        "expected_pose_ambiguity": True,
    }


def _estimator_terrain(
    authoritative: TerrainMap,
    truth: CameraExtrinsics,
    *,
    variant: str,
    coarse_factor: int,
    query_renderer: str = "shared-python",
) -> tuple[TerrainMap, dict[str, Any]]:
    """Build one declared estimator surface without changing query evidence."""

    if variant == "exact":
        estimator = authoritative
        factor = 1
    elif variant == "coarse":
        rows = _stride_indices(authoritative.y_m.size, coarse_factor)
        columns = _stride_indices(authoritative.x_m.size, coarse_factor)
        spec = authoritative.spec.model_copy(
            update={
                "grid_width": int(columns.size),
                "grid_height": int(rows.size),
            }
        )
        estimator = TerrainMap(
            spec=spec,
            x_m=np.asarray(authoritative.x_m[columns], dtype=np.float64),
            y_m=np.asarray(authoritative.y_m[rows], dtype=np.float64),
            elevation_m=np.asarray(authoritative.elevation_m[np.ix_(rows, columns)], dtype=np.float64),
            latitude_deg=np.asarray(authoritative.latitude_deg[np.ix_(rows, columns)], dtype=np.float64),
            longitude_deg=np.asarray(authoritative.longitude_deg[np.ix_(rows, columns)], dtype=np.float64),
        )
        factor = coarse_factor
    else:
        raise ValueError(f"unsupported estimator terrain variant {variant!r}")
    authoritative_ground = authoritative.elevation_at(truth.position.east_m, truth.position.north_m)
    estimator_ground = estimator.elevation_at(truth.position.east_m, truth.position.north_m)
    return estimator, {
        "variant": variant,
        "derivation": (
            "authoritative terrain unchanged"
            if variant == "exact"
            else "take every Nth authoritative row/column and retain final boundaries; no interpolation"
        ),
        "downsample_factor": factor,
        "shared_renderer_family": "SyntheticRenderer" if query_renderer == "shared-python" else None,
        "query_renderer_family": ("SyntheticRenderer" if query_renderer == "shared-python" else "chromium_webgl2_raw"),
        "candidate_renderer_family": "SyntheticRenderer",
        "independent_renderer_used": query_renderer == "webgl",
        "authoritative": terrain_fingerprint(authoritative),
        "estimator": terrain_fingerprint(estimator),
        "ground_elevation_delta_at_truth_xy_m": round(estimator_ground - authoritative_ground, 8),
    }


def _stride_indices(size: int, factor: int) -> NDArray[np.int64]:
    indices = np.arange(0, size, factor, dtype=np.int64)
    if indices[-1] != size - 1:
        indices = np.append(indices, size - 1)
    return indices


def _summary_markdown(results: dict[str, Any]) -> str:
    aggregate = results["aggregates"]
    proposal = aggregate["proposal_recall"]
    query_renderer = results.get("config", {}).get("query_renderer", "shared-python")
    lines = [
        f"# Synthetic {query_renderer} query stage ceiling",
        "",
        "This is a stage-isolation benchmark, not a real-photo leaderboard. Reference-rendered depth is oracle-only.",
        "",
        f"Full proposal-pool recall before ranking: {proposal['reached']}/{proposal['cases']} ({proposal['rate']}).",
        "",
        "| track | method | role | n | selected pose reaches target* | decision accuracy | abstain | "
        "raw top-1 | selection coverage | median target rank | median numerical-truth rank† | "
        "median truth regret | median alias margin‡ | median pos m | median yaw deg |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate["candidate_ranking"]:
        lines.append(
            "| {track} | {method} | {role} | {cases} | {success} | {decision} | {abstentions} | "
            "{raw_top1} | {coverage} | {rank} | {truth_rank} | {regret} | {alias_margin} | "
            "{position} | {yaw} |".format(
                track=row["track"],
                method=row["method"],
                role=row["evidence_role"],
                cases=row["cases"],
                success=row["selected_pose_target_hit_rate"],
                decision=row["decision_accuracy"],
                abstentions=row["abstentions"],
                raw_top1=row["raw_top1_target_hit_rate"],
                coverage=row["selection_coverage"],
                rank=row["median_first_target_rank"],
                truth_rank=row["median_numerical_truth_candidate_rank"],
                regret=row["median_numerical_truth_score_regret_to_winner"],
                alias_margin=row["median_geographic_alias_margin"],
                position=row["median_winner_position_error_m"],
                yaw=row["median_winner_yaw_error_deg"],
            )
        )
    lines.extend(
        [
            "",
            "\\* A target hit is an identity check, not benchmark success; expected-ambiguity cases must abstain. "
            "Decision accuracy is the contract metric. † The numerical-truth candidate is selected post-freeze by "
            "pose error and can be off-grid; rank ties are candidate-ID ordered, so read rank with regret. "
            "‡ Alias score minus numerical-truth score: positive favors truth, negative favors a distant alias.",
        ]
    )
    lines.extend(
        [
            "",
            "## Paired against the unchanged input prior",
            "",
            "Negative deltas improve on the prior. Decision deltas are zero when the method abstains "
            "and leaves the prior unchanged.",
            "",
            "| track | method | n | raw joint wins | raw joint regressions | raw Δpos m | raw Δyaw deg | "
            "decision Δpos m | decision Δyaw deg |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate["candidate_ranking"]:
        lines.append(
            "| {track} | {method} | {cases} | {wins} | {regressions} | {raw_position} | {raw_yaw} | "
            "{decision_position} | {decision_yaw} |".format(
                track=row["track"],
                method=row["method"],
                cases=row["cases"],
                wins=row["winner_prior_improvements"],
                regressions=row["winner_prior_regressions"],
                raw_position=row["median_winner_position_error_delta_vs_prior_m"],
                raw_yaw=row["median_winner_yaw_error_delta_vs_prior_deg"],
                decision_position=row["median_decision_position_error_delta_vs_prior_m"],
                decision_yaw=row["median_decision_yaw_error_delta_vs_prior_deg"],
            )
        )
    lines.extend(
        [
            "",
            "## Estimator-terrain strata",
            "",
            "| terrain | track | method | n | raw top-1 | decision accuracy | false accepts | median pos m |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    focus = {
        ("oracle_mask", "skyline"),
        ("rgb_color", "skyline"),
        ("reference_depth_oracle", "oracle_relative_depth"),
    }
    for variant, stratum in aggregate["estimator_terrain_strata"].items():
        for row in stratum["candidate_ranking"]:
            if (row["track"], row["method"]) not in focus:
                continue
            lines.append(
                "| {variant} | {track} | {method} | {cases} | {raw_top1} | {decision} | {false_accepts} | "
                "{position} |".format(
                    variant=variant,
                    track=row["track"],
                    method=row["method"],
                    cases=row["cases"],
                    raw_top1=row["raw_top1_target_hit_rate"],
                    decision=row["decision_accuracy"],
                    false_accepts=row["false_accepts"],
                    position=row["median_winner_position_error_m"],
                )
            )
    return "\n".join(lines) + "\n"


def _register_estimator_artifact(
    files: dict[str, bytes],
    manifest: dict[str, dict[str, Any]],
    artifact: ImmutableArtifact,
    *,
    role: str,
    case_id: str,
) -> None:
    filename = artifact.ref.filename
    existing_content = files.get(filename)
    if existing_content is not None and existing_content != artifact.content:
        raise RuntimeError(f"conflicting immutable estimator artifact: {filename}")
    files[filename] = artifact.content
    existing_record = manifest.get(filename)
    if existing_record is None:
        manifest[filename] = {
            **artifact.ref.model_dump(mode="json"),
            "role": role,
            "case_ids": [case_id],
        }
    else:
        if existing_record["role"] != role or existing_record["sha256"] != artifact.ref.sha256:
            raise RuntimeError(f"conflicting estimator artifact manifest: {filename}")
        existing_record["case_ids"].append(case_id)


def _commit_artifact(
    output: Path,
    results: dict[str, Any],
    summary: str,
    *,
    started_at: datetime,
    finished_at: datetime,
    query_renderer: str = "shared-python",
    extra_files: Mapping[str, bytes] | None = None,
) -> None:
    result_bytes = canonical_json_bytes(results)
    summary_bytes = summary.encode()
    run = {
        "schema": SYNTHETIC_BENCHMARK_SCHEMA,
        "run_id": output.name,
        "status": "complete",
        "created_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "case_count": len(results["cases"]),
        "results_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "code": _code_provenance(query_renderer),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pydantic": distribution_version("pydantic"),
            "scipy": distribution_version("scipy"),
        },
    }
    files = {
        "results.json": result_bytes,
        "summary.md": summary_bytes,
        "run.json": canonical_json_bytes(run),
    }
    for name, content in (extra_files or {}).items():
        if name in files:
            raise ValueError(f"extra artifact file collides with the benchmark transaction: {name}")
        files[name] = content
    _publish_directory_once(
        output,
        files,
        write_bytes=_write_once,
        sync_directory=_fsync_directory,
    )


def _code_provenance(query_renderer: str = "shared-python") -> dict[str, Any]:
    paths = [
        BASE / "pyproject.toml",
        BASE / "uv.lock",
        BASE / "src/peakle/config.py",
        BASE / "src/peakle/default_settings.yaml",
        BASE / "src/peakle/domain/camera.py",
        BASE / "src/peakle/domain/coordinates.py",
        BASE / "src/peakle/domain/projection.py",
        BASE / "src/peakle/domain/terrain.py",
        BASE / "src/peakle/localize/synthetic_pipeline_bench.py",
        Path(__file__),
        BASE / "src/peakle/localize/extract.py",
        BASE / "src/peakle/localize/typed_outlines.py",
        BASE / "src/peakle/rendering/rasterizer.py",
        BASE / "src/peakle/rendering/pinhole.py",
        BASE / "src/peakle/rendering/skyline.py",
        BASE / "src/peakle/research/synthetic_query.py",
        BASE / "src/peakle/research/webgl_contract.py",
        BASE / "src/peakle/scene/state.py",
        BASE / "src/peakle/terrain/generator.py",
        BASE / "src/peakle/terrain/peak_detection.py",
    ]
    if query_renderer == "webgl":
        paths.extend(
            (
                BASE / "src/peakle/research/webgl_query.py",
                BASE / "src/peakle/research/webgl_query.html",
                BASE / "src/peakle/research/synthetic_estimator.py",
            )
        )
    implementation_subset = []
    for path in paths:
        implementation_subset.append(
            {
                "path": str(path.relative_to(BASE)),
                "sha256": _file_sha256(path),
            }
        )
    status = _git("status", "--porcelain", "--untracked-files=all")
    diff = _git("diff", "--binary", "HEAD")
    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_tree_sha": _git("rev-parse", "HEAD^{tree}"),
        "dirty": bool(status) if status is not None else None,
        "scope": "whole_worktree",
        "worktree_status": status,
        "worktree_status_sha256": hashlib.sha256((status or "").encode()).hexdigest(),
        "worktree_diff_sha256": hashlib.sha256((diff or "").encode()).hexdigest(),
        "implementation_subset_role": "causal_files_for_human_review",
        "implementation_subset": implementation_subset,
    }


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ("git", *args),
            cwd=BASE,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except OSError, subprocess.CalledProcessError:
        return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_ints(value: str, label: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise SystemExit(f"invalid {label} list: {value}") from exc
    if not result:
        raise SystemExit(f"at least one {label} is required")
    return result


def _csv_choices(value: str, allowed: tuple[str, ...], label: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(result) - set(allowed))
    if unknown:
        raise SystemExit(f"unknown {label}(s): {', '.join(unknown)}; expected {', '.join(allowed)}")
    if not result:
        raise SystemExit(f"at least one {label} is required")
    return result


def _prior_regimes(value: str) -> tuple[SyntheticPriorRegime, ...]:
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(names) - set(DEFAULT_PRIOR_REGIMES))
    if unknown:
        raise SystemExit(f"unknown prior regime(s): {', '.join(unknown)}; expected {', '.join(DEFAULT_PRIOR_REGIMES)}")
    if not names:
        raise SystemExit("at least one prior regime is required")
    return tuple(DEFAULT_PRIOR_REGIMES[name] for name in names)


def _default_output_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return BASE / f"local/output/{stamp}-synthetic-pose-bench"


if __name__ == "__main__":
    main()

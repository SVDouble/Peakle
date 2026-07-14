"""Run the custom-pinhole shared-renderer synthetic stage upper-bound."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import platform
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from peakle.config import load_settings
from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics
from peakle.domain.coordinates import EARTH_RADIUS_M, LocalPoint
from peakle.domain.terrain import TerrainMap
from peakle.localize.extract import best_skyline_candidate, extract_candidates
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
    extraction_quality,
    haze_image,
)
from peakle.rendering.rasterizer import SyntheticRenderer
from peakle.rendering.terrain_view import terrain_fingerprint
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
    cases: list[dict[str, Any]] = []
    scenes = _rugged_scenes(
        seeds,
        views_per_scene=args.views_per_scene,
        terrain_width_m=args.terrain_width_m,
        terrain_height_m=args.terrain_height_m,
        terrain_grid_width=args.terrain_grid_width,
        terrain_grid_height=args.terrain_grid_height,
        eye_height_m=config.eye_height_m,
    )
    if not args.no_radial_control:
        scenes.append(
            _radial_control_scene(
                terrain_width_m=args.terrain_width_m,
                terrain_height_m=args.terrain_height_m,
                terrain_grid_width=args.terrain_grid_width,
                terrain_grid_height=args.terrain_grid_height,
                eye_height_m=config.eye_height_m,
            )
        )
    total = len(scenes) * len(prior_regimes) * len(estimator_terrain_variants)
    print(
        f"Custom-pinhole shared-renderer stage upper-bound: {len(scenes)} scene/view(s), "
        f"{len(prior_regimes)} prior regime(s), {len(estimator_terrain_variants)} estimator terrain variant(s), "
        f"{total} candidate archives",
        flush=True,
    )
    for scene_index, scene in enumerate(scenes, start=1):
        observations = _observations(scene, intrinsics, config)
        for variant in estimator_terrain_variants:
            estimator_terrain, terrain_record = _estimator_terrain(
                scene["terrain"],
                scene["truth"],
                variant=variant,
                coarse_factor=args.coarse_factor,
            )
            for regime in prior_regimes:
                case_started = time.perf_counter()
                prior = controlled_prior(estimator_terrain, scene["truth"], regime, config)
                archive = build_synthetic_candidate_archive(
                    estimator_terrain,
                    intrinsics,
                    prior,
                    observations["profiles"],
                    observations["reference_depth_oracle"],
                    config=config,
                )
                evaluation = evaluate_synthetic_candidate_archive(
                    archive,
                    scene["truth"],
                    observations["quality"],
                    observations["expected_actions"],
                    expected_pose_ambiguity=bool(scene["expected_pose_ambiguity"]),
                )
                cases.append(
                    {
                        "case_id": f"{scene['scene_id']}--{variant}--{regime.name}",
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
                        "observation_tracks": observations["metadata"],
                        "archive": archive,
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
            "renderer_family": (
                "authoritative observations and estimator candidates use SyntheticRenderer; "
                "this benchmark does not claim independent-renderer validation"
            ),
            "terrain_mismatch_axis": (
                "coarse deterministically subsamples the authoritative grid before candidate rendering"
            ),
            "expected_ambiguity_control": "radially symmetric terrain must cause pose-mode abstention",
        },
        "config": {
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
            "observation_tracks": {
                "oracle_mask": "exact rendered terrain mask; diagnostic extraction ceiling",
                "rgb_color": "current deterministic colour-mask extractor on the synthetic RGB render",
                "rgb_haze": "same extractor after deterministic low-contrast/cloud corruption",
            },
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
) -> list[dict[str, Any]]:
    if views_per_scene < 1:
        raise SystemExit("--views-per-scene must be positive")
    settings = load_settings()
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
) -> dict[str, Any]:
    settings = load_settings()
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
        "shared_renderer_family": "SyntheticRenderer",
        "independent_renderer_used": False,
        "authoritative": terrain_fingerprint(authoritative),
        "estimator": terrain_fingerprint(estimator),
        "ground_elevation_delta_at_truth_xy_m": round(estimator_ground - authoritative_ground, 8),
    }


def _stride_indices(size: int, factor: int) -> NDArray[np.int64]:
    indices = np.arange(0, size, factor, dtype=np.int64)
    if indices[-1] != size - 1:
        indices = np.append(indices, size - 1)
    return indices


def _observations(
    scene: dict[str, Any],
    intrinsics: CameraIntrinsics,
    config: SyntheticSearchConfig,
) -> dict[str, Any]:
    renderer = SyntheticRenderer()
    render = renderer.render(scene["terrain"], intrinsics, scene["truth"], stride=config.render_stride)
    geometry = renderer.geometry(scene["terrain"], intrinsics, scene["truth"], stride=config.render_stride)
    oracle = np.asarray(render.skyline_profile, dtype=np.float64)
    rgb = np.asarray(render.image, dtype=np.uint8)
    profiles: dict[str, NDArray[np.float64]] = {"oracle_mask": oracle}
    quality: dict[str, dict[str, Any]] = {
        "oracle_mask": extraction_quality(
            oracle,
            oracle,
            extractor_name="exact_terrain_mask",
            coverage=1.0,
            agreement=1.0,
            config=config,
        )
    }
    metadata: dict[str, Any] = {
        "oracle_mask": {
            "source": "exact reference-pose terrain mask",
            "analysis_only": True,
            "production_eligible": False,
            "quality": quality["oracle_mask"],
            "expected_action": "select",
            "expected_action_source": "predeclared_case_design",
        }
    }
    expected_actions = {
        "oracle_mask": "select",
        "rgb_color": "select",
        "rgb_haze": "abstain",
    }
    for track, track_rgb in (
        ("rgb_color", rgb),
        ("rgb_haze", haze_image(rgb, seed=_stable_seed(scene["scene_id"]))),
    ):
        selected = best_skyline_candidate(extract_candidates(track_rgb, backend="color"))
        if selected is None:
            extractor_name = "none"
            extracted = np.full(intrinsics.width_px, np.nan, dtype=np.float64)
            coverage = 0.0
            agreement = 0.0
        else:
            extractor_name, candidate = selected
            extracted = np.asarray(candidate.rows, dtype=np.float64)
            coverage = candidate.coverage
            agreement = candidate.agreement
        profiles[track] = extracted
        quality[track] = extraction_quality(
            extracted,
            oracle,
            extractor_name=extractor_name,
            coverage=coverage,
            agreement=agreement,
            config=config,
        )
        metadata[track] = {
            "source": "synthetic RGB render" if track == "rgb_color" else "deterministically hazed RGB render",
            "analysis_only": True,
            "production_eligible": False,
            "production_analogue": "current deterministic colour-mask skyline extractor",
            "quality": quality[track],
            "expected_action": expected_actions[track],
            "expected_action_source": "predeclared_case_design",
        }
    return {
        "profiles": profiles,
        "quality": quality,
        "metadata": metadata,
        "expected_actions": expected_actions,
        "reference_depth_oracle": np.asarray(geometry.forward_depth_m, dtype=np.float64),
    }


def _summary_markdown(results: dict[str, Any]) -> str:
    aggregate = results["aggregates"]
    proposal = aggregate["proposal_recall"]
    lines = [
        "# Synthetic custom-pinhole shared-renderer stage upper-bound",
        "",
        "This is a stage-isolation benchmark, not a real-photo leaderboard. Reference-rendered depth is oracle-only.",
        "",
        f"Full proposal-pool recall before ranking: {proposal['reached']}/{proposal['cases']} ({proposal['rate']}).",
        "",
        "| track | method | role | n | selected pose reaches target* | decision accuracy | abstain | "
        "raw top-1 | selection coverage | median target rank | median pos m | median yaw deg |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate["candidate_ranking"]:
        lines.append(
            "| {track} | {method} | {role} | {cases} | {success} | {decision} | {abstentions} | "
            "{raw_top1} | {coverage} | {rank} | {position} | {yaw} |".format(
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
                position=row["median_winner_position_error_m"],
                yaw=row["median_winner_yaw_error_deg"],
            )
        )
    lines.extend(
        [
            "",
            "\\* A target hit is an identity check, not benchmark success; expected-ambiguity cases must abstain. "
            "Decision accuracy is the contract metric.",
        ]
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


def _commit_artifact(
    output: Path,
    results: dict[str, Any],
    summary: str,
    *,
    started_at: datetime,
    finished_at: datetime,
) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to replace completed artifact directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging-{os.getpid()}")
    staging.mkdir(exist_ok=False)
    try:
        result_bytes = canonical_json_bytes(results)
        summary_bytes = summary.encode()
        _write_once(staging / "results.json", result_bytes)
        _write_once(staging / "summary.md", summary_bytes)
        run = {
            "schema": SYNTHETIC_BENCHMARK_SCHEMA,
            "run_id": output.name,
            "status": "complete",
            "created_at": started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "case_count": len(results["cases"]),
            "results_sha256": hashlib.sha256(result_bytes).hexdigest(),
            "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
            "code": _code_provenance(),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
            },
        }
        _write_once(staging / "run.json", canonical_json_bytes(run))
        _fsync_directory(staging)
        os.replace(staging, output)
        _fsync_directory(output.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        _fsync_directory(output.parent)
        raise


def _code_provenance() -> dict[str, Any]:
    paths = (
        BASE / "src/peakle/localize/synthetic_pipeline_bench.py",
        Path(__file__),
        BASE / "src/peakle/localize/extract.py",
        BASE / "src/peakle/localize/typed_outlines.py",
        BASE / "src/peakle/rendering/rasterizer.py",
        BASE / "src/peakle/terrain/generator.py",
    )
    implementation = []
    for path in paths:
        implementation.append(
            {
                "path": str(path.relative_to(BASE)),
                "sha256": _file_sha256(path),
            }
        )
    relative_paths = tuple(str(path.relative_to(BASE)) for path in paths)
    status = _git("status", "--porcelain", "--", *relative_paths)
    diff = _git("diff", "--binary", "HEAD", "--", *relative_paths)
    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "scope": "listed_implementation_paths_only",
        "implementation_status": status,
        "implementation_status_sha256": hashlib.sha256((status or "").encode()).hexdigest(),
        "implementation_diff_sha256": hashlib.sha256((diff or "").encode()).hexdigest(),
        "implementation": implementation,
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


def _write_once(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _stable_seed(value: str) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:16], 16)


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

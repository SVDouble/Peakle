"""Truth-side observation providers for the existing synthetic stage harness."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics
from peakle.domain.terrain import TerrainMap
from peakle.localize.extract import best_skyline_candidate, extract_candidates
from peakle.localize.synthetic_pipeline_bench import SyntheticSearchConfig, extraction_quality, haze_image
from peakle.rendering.rasterizer import SyntheticRenderer
from peakle.research.webgl_contract import QueryArtifactFile, freeze_webgl_query_artifact
from peakle.research.webgl_query import render_terrain_webgl_query


@dataclass(frozen=True, slots=True)
class SyntheticQueryObservations:
    profiles: Mapping[str, NDArray[np.float64]]
    quality: Mapping[str, Mapping[str, Any]]
    metadata: Mapping[str, Mapping[str, Any]]
    expected_actions: Mapping[str, str]
    reference_depth_oracle: NDArray[np.float64]
    artifact_files: Mapping[str, bytes]
    artifact_manifest: tuple[QueryArtifactFile, ...]
    query_provenance: Mapping[str, Any]


def build_synthetic_query_observations(
    scene: Mapping[str, object],
    intrinsics: CameraIntrinsics,
    config: SyntheticSearchConfig,
    *,
    query_renderer: str = "shared-python",
    chromium_path: str | None = None,
) -> SyntheticQueryObservations:
    """Render and freeze query observations before candidate generation."""

    scene_id, terrain, truth = _scene_fields(scene)
    if query_renderer == "shared-python":
        renderer = SyntheticRenderer()
        render = renderer.render(terrain, intrinsics, truth, stride=config.render_stride)
        geometry = renderer.geometry(terrain, intrinsics, truth, stride=config.render_stride)
        oracle = np.asarray(render.skyline_profile, dtype=np.float64)
        rgb = np.asarray(render.image, dtype=np.uint8)
        reference_depth = np.asarray(geometry.forward_depth_m, dtype=np.float64)
        query_source = "exact reference-pose terrain mask"
        query_provenance: dict[str, Any] = {
            "renderer_family": "SyntheticRenderer",
            "independent_rasterizer": False,
        }
        artifact_files: Mapping[str, bytes] = {}
        artifact_manifest: tuple[QueryArtifactFile, ...] = ()
    elif query_renderer == "webgl":
        query = render_terrain_webgl_query(terrain, intrinsics, truth, chromium_path=chromium_path)
        oracle = np.asarray(query.skyline_profile, dtype=np.float64)
        rgb = np.asarray(query.rgb, dtype=np.uint8)
        reference_depth = np.asarray(query.forward_depth_m, dtype=np.float64)
        query_source = "sealed independent Chromium/WebGL semantic terrain mask"
        query_provenance = query.provenance.model_dump(mode="json", by_alias=True)
        frozen = freeze_webgl_query_artifact(query, f"query-{scene_id}")
        artifact_files = frozen.files
        artifact_manifest = frozen.manifest
    else:
        raise ValueError(f"unsupported query renderer {query_renderer!r}")

    oracle_coverage = float(np.isfinite(oracle).mean())
    profiles: dict[str, NDArray[np.float64]] = {"oracle_mask": oracle}
    quality: dict[str, dict[str, Any]] = {
        "oracle_mask": extraction_quality(
            oracle,
            oracle,
            extractor_name="exact_terrain_mask",
            coverage=oracle_coverage,
            agreement=1.0,
            config=config,
        )
    }
    metadata: dict[str, dict[str, Any]] = {
        "oracle_mask": {
            "source": query_source,
            "sealed_truth_side_query_provenance_ref": scene_id,
            "provenance_supplied_to_candidate_builder": False,
            "analysis_only": True,
            "production_eligible": False,
            "quality": quality["oracle_mask"],
            "expected_action": "select",
            "expected_action_source": "predeclared_case_design",
        }
    }
    expected_actions = {"oracle_mask": "select"}
    if query_renderer == "shared-python":
        expected_actions.update({"rgb_color": "select", "rgb_haze": "abstain"})
        for track, track_rgb in (
            ("rgb_color", rgb),
            ("rgb_haze", haze_image(rgb, seed=_stable_seed(scene_id))),
        ):
            _add_rgb_track(track, track_rgb, intrinsics, config, oracle, profiles, quality, metadata, expected_actions)
    return SyntheticQueryObservations(
        profiles=profiles,
        quality=quality,
        metadata=metadata,
        expected_actions=expected_actions,
        reference_depth_oracle=reference_depth,
        artifact_files=artifact_files,
        artifact_manifest=artifact_manifest,
        query_provenance=query_provenance,
    )


def renderer_contract(query_renderer: str) -> str:
    if query_renderer == "webgl":
        return (
            "authoritative observations use an isolated raw WebGL2 rasterizer in headless Chromium; "
            "estimator candidates use SyntheticRenderer. This is independent rasterization over a shared "
            "pinhole/heightfield scene model, not independent physical synthetic truth"
        )
    return (
        "authoritative observations and estimator candidates use SyntheticRenderer; "
        "this benchmark does not claim independent-renderer validation"
    )


def observation_track_contract(query_renderer: str) -> dict[str, str]:
    tracks = {"oracle_mask": "exact rendered semantic terrain mask; diagnostic extraction ceiling"}
    if query_renderer == "shared-python":
        tracks.update(
            {
                "rgb_color": "current deterministic colour-mask extractor on the synthetic RGB render",
                "rgb_haze": "same extractor after deterministic low-contrast/cloud corruption",
            }
        )
    return tracks


def _scene_fields(scene: Mapping[str, object]) -> tuple[str, TerrainMap, CameraExtrinsics]:
    scene_id = scene.get("scene_id")
    terrain = scene.get("terrain")
    truth = scene.get("truth")
    if not isinstance(scene_id, str) or not scene_id:
        raise ValueError("synthetic query scene needs a non-empty scene_id")
    if not isinstance(terrain, TerrainMap) or not isinstance(truth, CameraExtrinsics):
        raise TypeError("synthetic query scene needs validated terrain and camera truth")
    return scene_id, terrain, truth


def _add_rgb_track(
    track: str,
    rgb: NDArray[np.uint8],
    intrinsics: CameraIntrinsics,
    config: SyntheticSearchConfig,
    oracle: NDArray[np.float64],
    profiles: dict[str, NDArray[np.float64]],
    quality: dict[str, dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
    expected_actions: Mapping[str, str],
) -> None:
    selected = best_skyline_candidate(extract_candidates(rgb, backend="color"))
    if selected is None:
        extractor_name = "none"
        extracted = np.full(intrinsics.width_px, np.nan, dtype=np.float64)
        coverage = agreement = 0.0
    else:
        extractor_name, candidate = selected
        extracted = np.asarray(candidate.rows, dtype=np.float64)
        coverage, agreement = candidate.coverage, candidate.agreement
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


def _stable_seed(value: str) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:16], 16)

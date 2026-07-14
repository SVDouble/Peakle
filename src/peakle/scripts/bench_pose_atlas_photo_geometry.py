"""Verify a frozen ``photo_auto`` pose atlas with photo-observable geometry.

This benchmark is intentionally offline and write-once.  It reconstructs the
exact automatic skyline that produced the source atlas, extracts DexiNed edges
and Depth-Anything relative depth from the source RGB, then scores every frozen
atlas candidate.  The model files, photo, terrain cache, source atlas, and
selected implementation files are content-addressed and checked again before
publication.

Numeric pose truth is not retained by the estimator phase.  ``verifier.json``
is written, file-fsynced, hash-checked, and directory-fsynced before the source
artifact is re-read for post-hoc numeric evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import socket
import statistics
import subprocess
import tempfile
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from peakle.depth import DepthEstimator
from peakle.domain.camera import CameraExtrinsics
from peakle.domain.coordinates import GeoPoint, LocalPoint
from peakle.domain.pose import PosePrior
from peakle.edges import EdgeDetector, LearnedEdges
from peakle.io.artifacts import fsync_directory as _fsync_directory
from peakle.io.artifacts import write_once_bytes as _write_once
from peakle.localize.atlas_dashboard import ATLAS_STUDY_SCHEMA
from peakle.localize.atlas_dashboard import canonical_json_bytes as _json_bytes
from peakle.localize.extract import best_skyline_candidate, extract_candidates
from peakle.localize.paths import BASE, COP_TILES_DIR, GEOPOSE_DIR
from peakle.localize.photo_geometry_verifier import (
    PHOTO_VERIFIER_ARCHIVE_SCHEMA,
    PhotoGeometryVerifierArchive,
    PhotoGeometryVerifierConfig,
    build_photo_geometry_verifier,
    evaluate_photo_geometry_verifier,
    extract_photo_geometry_evidence,
)
from peakle.localize.strategy_bench import (
    PriorScenario,
    default_terrain_cache_inventory,
    file_sha256,
    provision_estimator_terrain,
)
from peakle.terrain.copernicus import load_copernicus_terrain

PHOTO_GEOMETRY_STUDY_SCHEMA = "peakle_pose_atlas_photo_geometry_study_v1"
ESTIMATOR_FILE = "verifier.json"
PHOTO_TRACK = "photo_auto"


def main() -> None:
    args = _parser().parse_args()
    atlas_path = Path(args.atlas)
    output_dir = Path(args.output)
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing artifact directory: {output_dir}")

    source_bytes = atlas_path.read_bytes()
    source_results_sha256 = hashlib.sha256(source_bytes).hexdigest()
    source_results = _json_object(source_bytes, atlas_path)
    source_run_path = atlas_path.with_name("run.json")
    source_run_bytes = source_run_path.read_bytes()
    source_run_sha256 = hashlib.sha256(source_run_bytes).hexdigest()
    source_run = _json_object(source_run_bytes, source_run_path)
    _validate_source_artifact(source_results, source_run, source_bytes)
    selected = _selected_records(source_results, args.samples)
    estimator_specs = _estimator_specs(selected)
    terrain_config = _estimator_terrain_config(source_results)
    expected_photo_hashes = _expected_photo_hashes(source_run, estimator_specs)
    expected_cache_sha = _expected_cache_sha(source_run)

    # The full sample records contain numeric truth.  Drop them before loading a
    # model or calling estimator code; they are re-read only after the durable
    # estimator freeze below.
    del selected, source_results, source_bytes, source_run, source_run_bytes

    sample_dirs = {path.name: path for path in _find_photo_sample_dirs()}
    missing = [str(spec["name"]) for spec in estimator_specs if str(spec["name"]) not in sample_dirs]
    if missing:
        raise SystemExit(f"GeoPose inputs are unavailable for: {', '.join(missing)}")
    _validate_photo_hashes(estimator_specs, expected_photo_hashes, sample_dirs)

    cache_before = _compact_cache_inventory(default_terrain_cache_inventory())
    if cache_before["aggregate_sha256"] != expected_cache_sha:
        raise SystemExit("terrain cache differs from the source atlas run; rerun the atlas before verification")
    edge_model_before = _file_model_provenance("dexined", Path(args.dexined_checkpoint), kind="checkpoint")
    depth_model_before = _directory_model_provenance("depth-anything", Path(args.depth_model_dir))
    implementation = _implementation_record()
    config = PhotoGeometryVerifierConfig(subsample=args.subsample)
    started_at = datetime.now(UTC)

    print(
        f"Photo geometry verifier: {len(estimator_specs)} sample(s), track {PHOTO_TRACK}, subsample {config.subsample}",
        flush=True,
    )
    with _offline_inference_environment():
        edge_detector = _load_dexined(Path(args.dexined_checkpoint), args.device)
        depth_estimator = _load_depth_anything(Path(args.depth_model_dir), args.device)
        estimator_samples, archives = _run_estimators(
            estimator_specs,
            terrain_config=terrain_config,
            sample_dirs=sample_dirs,
            expected_photo_hashes=expected_photo_hashes,
            edge_detector=edge_detector,
            depth_estimator=depth_estimator,
            edge_model_provenance=edge_model_before,
            depth_model_provenance=depth_model_before,
            config=config,
        )

    _verify_stability(
        atlas_path=atlas_path,
        source_results_sha256=source_results_sha256,
        source_run_path=source_run_path,
        source_run_sha256=source_run_sha256,
        estimator_specs=estimator_specs,
        expected_photo_hashes=expected_photo_hashes,
        sample_dirs=sample_dirs,
        cache_before=cache_before,
        edge_model_before=edge_model_before,
        edge_checkpoint=Path(args.dexined_checkpoint),
        depth_model_before=depth_model_before,
        depth_model_dir=Path(args.depth_model_dir),
        implementation_before=implementation,
    )
    estimator_study = {
        "schema": PHOTO_GEOMETRY_STUDY_SCHEMA,
        "experimental": True,
        "production_eligible": False,
        "photo_observable_evidence_only": True,
        "source_depth_reference_used": False,
        "numeric_evaluation_reference_used": False,
        "source_atlas": {
            "path": str(atlas_path),
            "results_sha256": source_results_sha256,
            "run_sha256": source_run_sha256,
            "schema": ATLAS_STUDY_SCHEMA,
            "candidate_track": PHOTO_TRACK,
        },
        "config": config.to_record(),
        "models": {"edge": edge_model_before, "depth": depth_model_before},
        "offline_contract": _offline_contract_record(),
        "implementation": implementation,
        "terrain_cache": cache_before,
        "samples": estimator_samples,
    }
    estimator_bytes = _json_bytes(estimator_study)
    estimator_sha = hashlib.sha256(estimator_bytes).hexdigest()

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    committed = False
    try:
        _write_once(staging_dir / ESTIMATOR_FILE, estimator_bytes)
        persisted = (staging_dir / ESTIMATOR_FILE).read_bytes()
        if hashlib.sha256(persisted).hexdigest() != estimator_sha:
            raise RuntimeError("persisted estimator archive failed its post-write digest check")
        _fsync_directory(staging_dir)

        # Only now is the truth-bearing source sample record re-read.  Evaluation
        # cannot alter the already durable, content-addressed estimator archive.
        evaluation_source_bytes = atlas_path.read_bytes()
        if hashlib.sha256(evaluation_source_bytes).hexdigest() != source_results_sha256:
            raise RuntimeError("source atlas changed before numeric evaluation")
        evaluation_source = _json_object(evaluation_source_bytes, atlas_path)
        evaluation_samples = _selected_records(evaluation_source, args.samples)
        evaluated = _evaluate_frozen_archives(evaluation_samples, archives)
        results = {
            "schema": PHOTO_GEOMETRY_STUDY_SCHEMA,
            "experimental": True,
            "production_eligible": False,
            "photo_observable_evidence_only": True,
            "source_atlas": {
                "results_sha256": source_results_sha256,
                "run_sha256": source_run_sha256,
                "candidate_track": PHOTO_TRACK,
            },
            "estimator_archive": {
                "file": ESTIMATOR_FILE,
                "sha256": estimator_sha,
                "frozen_before_numeric_evaluation": True,
            },
            "samples": evaluated,
            "aggregates": _aggregates(evaluated),
        }
        results_bytes = _json_bytes(results)
        summary_bytes = _summary_markdown(results).encode()
        _verify_stability(
            atlas_path=atlas_path,
            source_results_sha256=source_results_sha256,
            source_run_path=source_run_path,
            source_run_sha256=source_run_sha256,
            estimator_specs=estimator_specs,
            expected_photo_hashes=expected_photo_hashes,
            sample_dirs=sample_dirs,
            cache_before=cache_before,
            edge_model_before=edge_model_before,
            edge_checkpoint=Path(args.dexined_checkpoint),
            depth_model_before=depth_model_before,
            depth_model_dir=Path(args.depth_model_dir),
            implementation_before=implementation,
        )
        finished_at = datetime.now(UTC)
        run = {
            "schema": PHOTO_GEOMETRY_STUDY_SCHEMA,
            "run_id": output_dir.name,
            "status": "complete",
            "created_at": started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "wall_runtime_s": round((finished_at - started_at).total_seconds(), 3),
            "source_atlas_results_sha256": source_results_sha256,
            "source_atlas_run_sha256": source_run_sha256,
            "estimator_archive_sha256": estimator_sha,
            "results_sha256": hashlib.sha256(results_bytes).hexdigest(),
            "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
            "truth_separation": {
                "photo_rgb_used_by_estimator": True,
                "source_depth_pfm_used_by_estimator": False,
                "numeric_reference_pose_used_by_estimator": False,
                "truth_records_reloaded_after_estimator_freeze": True,
                "estimator_archive_frozen_before_numeric_evaluation": True,
            },
            "provenance_stability": {
                "captured_before_model_load": True,
                "rechecked_before_estimator_freeze": True,
                "rechecked_before_artifact_commit": True,
                "changed_sections": [],
            },
            "environment": _environment_record(),
            "limitations": [
                "the verifier is experimental and its select/abstain thresholds are not calibrated",
                "Depth-Anything contributes scale-invariant ordinal depth only",
                "DexiNed internal ridges are not semantic terrain labels",
                "the complete frozen photo-atlas shortlist is scored; no candidates are added",
                "the benchmark grades horizontal position and yaw; pitch and roll remain ungraded",
            ],
        }
        _write_once(staging_dir / "results.json", results_bytes)
        _write_once(staging_dir / "summary.md", summary_bytes)
        _write_once(staging_dir / "run.json", _json_bytes(run))
        _fsync_directory(staging_dir)
        if output_dir.exists():
            raise RuntimeError(f"output directory appeared during execution: {output_dir}")
        os.rename(staging_dir, output_dir)
        _fsync_directory(output_dir.parent)
        committed = True
    finally:
        if not committed:
            shutil.rmtree(staging_dir, ignore_errors=True)
    print(f"Committed {output_dir} (estimator archive sha256 {estimator_sha[:12]}…).", flush=True)


def _run_estimators(
    estimator_specs: list[dict[str, Any]],
    *,
    terrain_config: dict[str, Any],
    sample_dirs: dict[str, Path],
    expected_photo_hashes: dict[str, str],
    edge_detector: EdgeDetector,
    depth_estimator: DepthEstimator,
    edge_model_provenance: dict[str, Any],
    depth_model_provenance: dict[str, Any],
    config: PhotoGeometryVerifierConfig,
) -> tuple[list[dict[str, Any]], dict[str, PhotoGeometryVerifierArchive]]:
    """Run the estimator from whitelisted records that contain no pose truth."""

    _assert_estimator_phase_inputs(estimator_specs)
    archives: dict[str, PhotoGeometryVerifierArchive] = {}
    records: list[dict[str, Any]] = []
    for index, spec in enumerate(estimator_specs, start=1):
        started = time.perf_counter()
        name = str(spec["name"])
        terrain = _rehydrate_estimator_terrain(spec, terrain_config)
        photo_path = sample_dirs[name] / "cyl" / "photo_crop.jpg"
        query = cast(dict[str, Any], cast(dict[str, Any], spec["estimator_archive"])["query_geometry"])
        rgb = _load_photo_for_query(photo_path, query)
        skyline, reconstruction = _reconstruct_photo_skyline(
            rgb,
            cast(dict[str, Any], spec["photo_evidence"]),
            cast(dict[str, Any], spec["estimator_archive"]),
        )
        evidence = extract_photo_geometry_evidence(
            rgb,
            skyline,
            edge_detector,
            depth_estimator,
            observation_provenance={
                "track": PHOTO_TRACK,
                "source": spec["photo_evidence"]["source"],
                "candidate": spec["photo_evidence"]["candidate"],
                "selection_uses_reference_truth": False,
                "evidence_generated_at_reference_pose": False,
                "source_atlas_sha256": spec["estimator_archive"]["archive_sha256"],
            },
            edge_model_provenance=edge_model_provenance,
            depth_model_provenance=depth_model_provenance,
            config=config,
        )
        archive = build_photo_geometry_verifier(
            terrain.terrain,
            terrain.high_resolution_patch,
            evidence,
            cast(dict[str, Any], spec["estimator_archive"]),
            config=config,
        )
        archives[name] = archive
        records.append(
            {
                "name": name,
                "candidate_track": PHOTO_TRACK,
                "source_photo_file_sha256": expected_photo_hashes[name],
                "photo_skyline_reconstruction": reconstruction,
                "terrain_inputs": _terrain_identity(terrain.provenance),
                "verifier_archive": archive.to_record(),
                "runtime_s": round(time.perf_counter() - started, 4),
            }
        )
        print(
            f"[{index}/{len(estimator_specs)}] {name}: ranked original atlas rank "
            f"{archive.ranked_winner.original_estimator_rank}; {records[-1]['runtime_s']:.1f}s",
            flush=True,
        )
    return records, archives


def _estimator_specs(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy only photo-estimator inputs; signed perturbations and truth stay out."""

    specs: list[dict[str, Any]] = []
    prior_fields = (
        "position",
        "yaw_deg",
        "pitch_deg",
        "horizontal_sigma_m",
        "vertical_sigma_m",
        "yaw_sigma_deg",
        "pitch_sigma_deg",
        "regime",
    )
    evidence_fields = ("source", "available", "candidate", "coverage", "agreement", "detected_candidates")
    for sample in samples:
        prior = _mapping(sample.get("prior"), "source atlas prior")
        perturbation = prior.get("perturbation")
        replicate = perturbation.get("replicate") if isinstance(perturbation, Mapping) else None
        if isinstance(replicate, bool) or not isinstance(replicate, int) or replicate < 0:
            raise ValueError("source atlas prior has no valid terrain-cache replicate")
        track = _mapping(_mapping(sample.get("tracks"), "source atlas tracks").get(PHOTO_TRACK), "photo track")
        evidence = _mapping(track.get("evidence"), "photo evidence")
        if evidence.get("selection_uses_reference_truth") is not False:
            raise ValueError("source photo skyline selection is not explicitly truth-blind")
        if evidence.get("evidence_generated_at_reference_pose") is not False:
            raise ValueError("source photo skyline was generated at the reference pose")
        whitelisted_evidence = {field: evidence[field] for field in evidence_fields}
        whitelisted_prior = {field: prior[field] for field in prior_fields}
        whitelisted_prior["terrain_cache_replicate"] = replicate
        spec = {
            "name": sample["name"],
            "coordinate_frame_origin": sample["coordinate_frame_origin"],
            "prior": whitelisted_prior,
            "terrain_expected_identity": _terrain_identity(
                _mapping(sample.get("terrain_inputs"), "source terrain inputs")
            ),
            "photo_evidence": whitelisted_evidence,
            "estimator_archive": track["estimator_archive"],
        }
        specs.append(spec)
    _assert_estimator_phase_inputs(specs)
    return specs


def _assert_estimator_phase_inputs(specs: list[dict[str, Any]]) -> None:
    """Fail closed if a later refactor adds truth or source-depth inputs."""

    encoded = json.dumps(specs, sort_keys=True).lower()
    forbidden = (
        '"reference":',
        '"evaluation":',
        '"errors":',
        '"compatibility":',
        '"photo_edge_support":',
        '"pfm_oracle":',
        '"perturbation":',
        '"realized":',
        '"requested":',
        ".pfm",
    )
    if any(token in encoded for token in forbidden):
        raise ValueError("forbidden evaluation, PFM, or signed-perturbation data crossed the estimator whitelist")


def _terrain_identity(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Retain terrain identity while dropping reference-construction annotations."""

    native = _mapping(provenance.get("native_patch"), "native patch provenance")
    native_fields = (
        "source",
        "center_source",
        "center_local_m",
        "center_geo_deg",
        "network_allowed",
        "radius_m",
        "status",
        "coverage",
        "used_by_estimator",
        "error_type",
    )
    return {
        "base_source": provenance.get("base_source"),
        "native_patch": {field: native[field] for field in native_fields if field in native},
        "regular_grid_fused_cells": provenance.get("regular_grid_fused_cells"),
        "regular_grid_is_per_cell_copy": provenance.get("regular_grid_is_per_cell_copy"),
    }


def _rehydrate_estimator_terrain(spec: dict[str, Any], terrain_config: dict[str, Any]):
    """Reconstruct and identity-check the source atlas' estimator terrain."""

    origin = GeoPoint.model_validate(spec["coordinate_frame_origin"])
    terrain = load_copernicus_terrain(
        origin.latitude_deg,
        origin.longitude_deg,
        extent_m=float(terrain_config["extent_m"]),
        grid=int(terrain_config["grid"]),
        tile_dir=COP_TILES_DIR,
    )
    prior_record = _mapping(spec.get("prior"), "whitelisted prior")
    prior = PosePrior(
        position=LocalPoint.model_validate(prior_record["position"]),
        yaw_deg=float(prior_record["yaw_deg"]),
        pitch_deg=float(prior_record["pitch_deg"]),
        horizontal_sigma_m=float(prior_record["horizontal_sigma_m"]),
        vertical_sigma_m=float(prior_record["vertical_sigma_m"]),
        yaw_sigma_deg=float(prior_record["yaw_sigma_deg"]),
        pitch_sigma_deg=float(prior_record["pitch_sigma_deg"]),
    )
    scenario = PriorScenario(
        name=cast(Any, str(prior_record["regime"])),
        prior=prior,
        use_position_prior=True,
        use_orientation_prior=True,
        perturbation={"replicate": int(prior_record["terrain_cache_replicate"])},
        contains_exact_reference=False,
        constructed_from_reference=False,
    )
    selection = provision_estimator_terrain(terrain, scenario)
    if _terrain_identity(selection.provenance) != spec["terrain_expected_identity"]:
        raise RuntimeError(f"rehydrated estimator terrain differs from source atlas for {spec['name']}")
    return selection


def _load_photo_for_query(path: Path, query: Mapping[str, Any]) -> NDArray[np.uint8]:
    width = _positive_record_int(query.get("width_px"), "atlas query width")
    height = _positive_record_int(query.get("height_px"), "atlas query height")
    with Image.open(path) as source:
        image = source.convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    result = np.asarray(image, dtype=np.uint8)
    if result.shape != (height, width, 3):
        raise RuntimeError("decoded source photo does not match the atlas query geometry")
    return result


def _find_photo_sample_dirs(data_dir: Path = GEOPOSE_DIR) -> list[Path]:
    """Discover the sole corpus input used here without requiring source PFM."""

    return sorted(
        directory
        for directory in data_dir.iterdir()
        if directory.is_dir() and (directory / "cyl" / "photo_crop.jpg").is_file()
    )


def _reconstruct_photo_skyline(
    rgb: NDArray[np.uint8],
    evidence: Mapping[str, Any],
    atlas_archive: Mapping[str, Any],
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    backend = evidence.get("source")
    if not isinstance(backend, str) or not backend:
        raise ValueError("photo track does not declare its skyline extractor")
    candidates = extract_candidates(rgb, backend=backend)
    chosen = best_skyline_candidate(candidates, min_coverage=0.25)
    if chosen is None:
        raise RuntimeError("the source photo skyline is no longer reproducible")
    name, candidate = chosen
    if name != evidence.get("candidate"):
        raise RuntimeError("reconstructed photo skyline selection differs from the source atlas")
    if sorted(candidates) != evidence.get("detected_candidates"):
        raise RuntimeError("reconstructed photo skyline candidates differ from the source atlas")
    expected_coverage = evidence.get("coverage")
    expected_agreement = evidence.get("agreement")
    if not isinstance(expected_coverage, int | float) or not math.isfinite(float(expected_coverage)):
        raise ValueError("source photo skyline coverage is missing or malformed")
    if not isinstance(expected_agreement, int | float) or not math.isfinite(float(expected_agreement)):
        raise ValueError("source photo skyline agreement is missing or malformed")
    if round(float(candidate.coverage), 6) != float(expected_coverage):
        raise RuntimeError("reconstructed photo skyline coverage differs from the source atlas")
    if round(float(candidate.agreement), 6) != float(expected_agreement):
        raise RuntimeError("reconstructed photo skyline agreement differs from the source atlas")
    rows = np.asarray(candidate.rows, dtype=np.float64)
    query = _mapping(atlas_archive.get("query_geometry"), "atlas query geometry")
    expected = query.get("observed_skyline_sha256")
    actual = _observed_skyline_sha256(rows)
    if actual != expected:
        raise RuntimeError("reconstructed photo skyline does not match the frozen atlas hash")
    return rows, {
        "extractor": backend,
        "selected_candidate": name,
        "detected_candidates": sorted(candidates),
        "coverage": round(float(candidate.coverage), 6),
        "agreement": round(float(candidate.agreement), 6),
        "observed_skyline_sha256": actual,
        "matches_source_atlas": True,
    }


class _OfflineDexinedEdges(LearnedEdges):
    """DexiNed with an explicitly supplied checkpoint and no URL loader."""

    def __init__(self, checkpoint: Path, device: str) -> None:
        import torch  # noqa: PLC0415
        from kornia.filters.dexined import DexiNed  # noqa: PLC0415

        self._torch = torch
        self._device = _torch_device(torch, device)
        model = DexiNed(pretrained=False)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = _state_dict(payload)
        model.load_state_dict(state, strict=True)
        self._model = model.to(self._device).eval()


class _OfflineDepthAnything:
    """Depth-Anything loaded exclusively from an explicit local directory."""

    name = "depth-anything"

    def __init__(self, model_dir: Path, device: str) -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation  # noqa: PLC0415

        self._torch = torch
        self._device = _torch_device(torch, device)
        self._processor = AutoImageProcessor.from_pretrained(str(model_dir), local_files_only=True)
        self._model = (
            AutoModelForDepthEstimation.from_pretrained(str(model_dir), local_files_only=True).to(self._device).eval()
        )

    def estimate(self, rgb: NDArray[np.float64]) -> NDArray[np.float64]:
        torch = self._torch
        height, width = rgb.shape[:2]
        image = Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8), mode="RGB")
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.no_grad():
            predicted = self._model(**inputs).predicted_depth
            predicted = torch.nn.functional.interpolate(
                predicted.unsqueeze(1),
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        values = predicted.detach().cpu().numpy().astype(np.float64)
        # Depth-Anything's raw prediction is inverse-depth-like (near is large).
        result = _normalize(values.max() - values)
        del inputs, predicted
        if self._device == "cuda":
            torch.cuda.empty_cache()
        return result


def _load_dexined(checkpoint: Path, device: str) -> EdgeDetector:
    try:
        return _OfflineDexinedEdges(checkpoint.resolve(strict=True), device)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"cannot load offline DexiNed checkpoint: {error}") from error


def _load_depth_anything(model_dir: Path, device: str) -> DepthEstimator:
    try:
        return _OfflineDepthAnything(model_dir.resolve(strict=True), device)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"cannot load offline Depth-Anything model: {error}") from error


def _state_dict(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("DexiNed checkpoint is not a state-dict mapping")
    nested = payload.get("state_dict")
    state = nested if isinstance(nested, Mapping) else payload
    result = {str(key).removeprefix("module."): value for key, value in state.items()}
    if not result:
        raise ValueError("DexiNed checkpoint contains no parameters")
    return result


def _torch_device(torch: Any, requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return (
        "cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested)
    )


@contextmanager
def _offline_inference_environment() -> Iterator[None]:
    """Block network entry points while local photo models load and execute."""

    names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    previous = {name: os.environ.get(name) for name in names}
    for name in names:
        os.environ[name] = "1"

    def blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("network access is disabled for the photo-geometry benchmark")

    try:
        import torch  # noqa: PLC0415

        hub_patches = (
            patch.object(torch.hub, "load_state_dict_from_url", blocked),
            patch.object(torch.hub, "load", blocked),
        )
    except ImportError:
        hub_patches = ()
    socket_patches = (
        patch.object(socket, "create_connection", blocked),
        patch.object(socket.socket, "connect", blocked),
        patch.object(socket.socket, "connect_ex", blocked),
    )
    entered = []
    try:
        for active in (*hub_patches, *socket_patches):
            active.start()
            entered.append(active)
        yield
    finally:
        for active in reversed(entered):
            active.stop()
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _offline_contract_record() -> dict[str, Any]:
    return {
        "network_socket_connect_blocked": True,
        "torch_hub_downloads_blocked": True,
        "huggingface_offline_environment": True,
        "transformers_local_files_only": True,
        "dexined_pretrained_url_loader_used": False,
    }


def _file_model_provenance(name: str, path: Path, *, kind: str) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise SystemExit(f"{name} {kind} is not a file: {resolved}")
    record = {"path": resolved.name, "size_bytes": resolved.stat().st_size, "sha256": file_sha256(resolved)}
    return {
        "name": name,
        "input": "photo_rgb",
        "offline": True,
        "source": {"kind": kind, "path": str(resolved)},
        "files": [record],
        "aggregate_sha256": hashlib.sha256(_json_bytes([record])).hexdigest(),
    }


def _directory_model_provenance(name: str, directory: Path) -> dict[str, Any]:
    resolved = directory.resolve(strict=True)
    if not resolved.is_dir():
        raise SystemExit(f"{name} model path is not a directory: {resolved}")
    files = []
    for path in sorted(resolved.rglob("*")):
        if path.is_symlink():
            try:
                target = path.resolve(strict=True)
            except OSError as error:
                raise SystemExit(f"model directory contains a broken symlink: {path}") from error
            if not target.is_file():
                raise SystemExit(f"model directory symlinks must resolve to files: {path}")
        if path.is_file():
            record = {
                "path": path.relative_to(resolved).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            if path.is_symlink():
                record["symlink_target"] = os.readlink(path)
            files.append(record)
    if not files:
        raise SystemExit(f"{name} model directory contains no files: {resolved}")
    if not any(Path(str(record["path"])).suffix in {".bin", ".safetensors", ".pth", ".pt"} for record in files):
        raise SystemExit(f"{name} model directory contains no checkpoint file")
    return {
        "name": name,
        "input": "photo_rgb",
        "offline": True,
        "source": {"kind": "local_pretrained_directory", "path": str(resolved)},
        "files": files,
        "aggregate_sha256": hashlib.sha256(_json_bytes(files)).hexdigest(),
    }


def _evaluate_frozen_archives(
    sample_records: list[dict[str, Any]],
    archives: dict[str, PhotoGeometryVerifierArchive],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sample in sample_records:
        name = str(sample["name"])
        archive = archives[name]
        truth = CameraExtrinsics.model_validate(sample["reference"])
        requested = (1, 5, 10, 16, 32, 64, 128, len(archive.candidates))
        evaluation = evaluate_photo_geometry_verifier(archive, truth, top_ks=requested)
        source_evaluation = sample["tracks"][PHOTO_TRACK].get("evaluation")
        atlas_winner = source_evaluation.get("winner_errors") if isinstance(source_evaluation, dict) else None
        records.append(
            {
                "name": name,
                "candidate_track": PHOTO_TRACK,
                "verifier_archive_schema": PHOTO_VERIFIER_ARCHIVE_SCHEMA,
                "verifier_archive_sha256": archive.archive_sha256,
                "atlas_skyline_winner": atlas_winner,
                "evaluation_only_compatibility": _evaluation_compatibility_record(sample),
                "decision": archive.decision.to_record(),
                "evaluation": evaluation.to_record(),
            }
        )
    return records


def _aggregates(samples: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = _aggregate_metrics(samples)
    buckets = sorted({str(sample["evaluation_only_compatibility"]["fit_bucket"]) for sample in samples})
    aggregate["by_fit_bucket"] = {
        bucket: _aggregate_metrics(
            [sample for sample in samples if sample["evaluation_only_compatibility"]["fit_bucket"] == bucket]
        )
        for bucket in buckets
    }
    return aggregate


def _aggregate_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = [sample["evaluation"]["ranked_winner_errors"] for sample in samples]
    returned = [sample["evaluation"]["returned_candidate_errors"] for sample in samples]
    returned_candidates = [record for record in returned if isinstance(record, dict)]
    oracle = [sample["evaluation"]["candidate_pool_gt_oracle"] for sample in samples]
    component_names = sorted(
        {name for sample in samples for name in sample["evaluation"].get("component_winner_errors", {})}
    )
    selected = len(returned_candidates)
    returned_successes = sum(record["reaches_target"] is True for record in returned_candidates)
    return {
        "samples": len(samples),
        "ranked_winner_successes": sum(record["reaches_target"] is True for record in ranked),
        "selected_decisions": selected,
        "abstentions": len(samples) - selected,
        "returned_candidate_successes": returned_successes,
        "returned_candidate_false_accepts": selected - returned_successes,
        "beam_target_successes": sum(sample["evaluation"]["first_target_beam_rank"] is not None for sample in samples),
        "candidate_pool_oracle_successes": sum(record["reaches_target"] is True for record in oracle),
        "median_ranked_winner_horizontal_m": _median_error(ranked, "horizontal_position_m"),
        "median_ranked_winner_yaw_deg": _median_error(ranked, "yaw_deg"),
        "component_winners": {
            name: _evaluated_candidate_aggregate(
                [sample["evaluation"]["component_winner_errors"][name] for sample in samples]
            )
            for name in component_names
        },
    }


def _evaluated_candidate_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "samples": len(records),
        "successes": sum(record["reaches_target"] is True for record in records),
        "median_horizontal_m": _median_error(records, "horizontal_position_m"),
        "median_yaw_deg": _median_error(records, "yaw_deg"),
    }


def _median_error(records: list[dict[str, Any]], field: str) -> float | None:
    values = [float(record["errors"][field]) for record in records if record.get("errors", {}).get(field) is not None]
    return round(statistics.median(values), 5) if values else None


def _summary_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# Photo-observable geometry verifier",
        "",
        "DexiNed and Depth-Anything consume only the source RGB. Numeric pose truth was evaluated only after "
        "`verifier.json` was durably frozen.",
        "",
        "| sample | fit bucket | skyline-only winner | geometry-ranked winner | decision | "
        "beam target rank | original atlas rank |",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    for sample in results["samples"]:
        evaluation = sample["evaluation"]
        atlas = sample.get("atlas_skyline_winner")
        atlas_text = _format_atlas_winner(atlas) if isinstance(atlas, dict) else "-"
        winner = evaluation["ranked_winner_errors"]
        returned = evaluation["returned_candidate_errors"]
        decision = "selected" if isinstance(returned, dict) else "abstained"
        beam_rank = evaluation["first_target_beam_rank"]
        fit_bucket = sample["evaluation_only_compatibility"]["fit_bucket"]
        lines.append(
            f"| {sample['name']} | {fit_bucket} | {atlas_text} | {_format_evaluated(winner)} | {decision} | "
            f"{beam_rank if beam_rank is not None else '-'} | {winner['original_estimator_rank']} |"
        )
    lines.extend(
        [
            "",
            "## Evaluation-only terrain/reference fit buckets",
            "",
            "These buckets are copied from the source atlas only after estimator freeze.",
            "",
            "| fit bucket | samples | ranked successes | selected | false accepts | beam target hits |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for bucket, aggregate in results["aggregates"]["by_fit_bucket"].items():
        lines.append(
            f"| {bucket} | {aggregate['samples']} | {aggregate['ranked_winner_successes']} | "
            f"{aggregate['selected_decisions']} | {aggregate['returned_candidate_false_accepts']} | "
            f"{aggregate['beam_target_successes']} |"
        )
    return "\n".join(lines) + "\n"


def _evaluation_compatibility_record(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Copy compact GT/terrain fit evidence only after the estimator is frozen."""

    compatibility = sample.get("compatibility")
    if not isinstance(compatibility, Mapping):
        return {
            "fit_bucket": "UNAVAILABLE",
            "policy": None,
            "source": "source_atlas.compatibility",
            "used_by_estimator": False,
            "available": False,
        }
    tier = compatibility.get("tier")
    bucket = tier if isinstance(tier, str) and tier else "UNBUCKETED"
    metrics: dict[str, float] = {}
    for name in (
        "coverage",
        "median_deg",
        "p90_deg",
        "chamfer_deg",
        "within_025_deg",
        "within_050_deg",
        "within_100_deg",
        "fold_shift_disagreement_px",
    ):
        value = compatibility.get(name)
        if isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(float(value)):
            metrics[name] = float(value)
    height = compatibility.get("height")
    height_record = (
        {
            "tier": height.get("tier"),
            "physically_plausible": height.get("physically_plausible"),
            "raw_camera_clearance_m": height.get("raw_camera_clearance_m"),
        }
        if isinstance(height, Mapping)
        else None
    )
    return {
        "fit_bucket": bucket,
        "policy": compatibility.get("policy"),
        "metrics": metrics,
        "height": height_record,
        "source": "source_atlas.compatibility",
        "used_by_estimator": False,
        "available": True,
    }


def _format_atlas_winner(record: dict[str, Any]) -> str:
    errors = record.get("errors")
    return _format_errors(errors) if isinstance(errors, dict) else "-"


def _format_evaluated(record: dict[str, Any]) -> str:
    return _format_errors(record["errors"])


def _format_errors(errors: Mapping[str, Any]) -> str:
    return f"{float(errors['horizontal_position_m']):.1f} m / {float(errors['yaw_deg']):.1f}°"


def _selected_records(results: dict[str, Any], samples_value: str) -> list[dict[str, Any]]:
    raw_samples = results.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise SystemExit("source atlas contains no samples")
    by_name = {str(sample.get("name")): sample for sample in raw_samples if isinstance(sample, dict)}
    requested = [item.strip() for item in samples_value.split(",") if item.strip()]
    if len(requested) != len(set(requested)):
        raise SystemExit("duplicate sample names are not allowed")
    if not requested:
        requested = list(by_name)
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise SystemExit(f"source atlas does not contain: {', '.join(missing)}")
    selected = [by_name[name] for name in requested]
    unavailable = []
    for sample in selected:
        tracks = sample.get("tracks")
        track = tracks.get(PHOTO_TRACK) if isinstance(tracks, dict) else None
        if (
            not isinstance(track, dict)
            or track.get("status") != "ok"
            or not isinstance(track.get("estimator_archive"), dict)
            or not isinstance(track.get("evidence"), dict)
            or track["evidence"].get("available") is not True
        ):
            unavailable.append(str(sample["name"]))
    if unavailable:
        raise SystemExit(f"candidate track {PHOTO_TRACK!r} is unavailable for: {', '.join(unavailable)}")
    return selected


def _validate_source_artifact(results: dict[str, Any], run: dict[str, Any], results_bytes: bytes) -> None:
    if results.get("schema") != ATLAS_STUDY_SCHEMA or run.get("schema") != ATLAS_STUDY_SCHEMA:
        raise SystemExit(f"source must be a completed {ATLAS_STUDY_SCHEMA} artifact")
    if run.get("status") != "complete":
        raise SystemExit("source atlas run is not complete")
    if run.get("results_sha256") != hashlib.sha256(results_bytes).hexdigest():
        raise SystemExit("source atlas results.json does not match run.json")


def _estimator_terrain_config(results: dict[str, Any]) -> dict[str, Any]:
    config = results.get("config")
    terrain = config.get("terrain") if isinstance(config, dict) else None
    if not isinstance(terrain, dict):
        raise SystemExit("source atlas terrain configuration is missing")
    extent = terrain.get("extent_m")
    grid = terrain.get("grid")
    if isinstance(extent, bool) or not isinstance(extent, int | float) or not math.isfinite(float(extent)):
        raise SystemExit("source atlas terrain extent is invalid")
    if isinstance(grid, bool) or not isinstance(grid, int) or grid < 2:
        raise SystemExit("source atlas terrain grid is invalid")
    return {"extent_m": float(extent), "grid": grid}


def _expected_photo_hashes(source_run: dict[str, Any], specs: list[dict[str, Any]]) -> dict[str, str]:
    inputs = source_run.get("inputs")
    files = inputs.get("files") if isinstance(inputs, dict) else None
    if not isinstance(files, list):
        raise SystemExit("source atlas input fingerprint is missing")
    expected = {str(record.get("path")): str(record.get("sha256")) for record in files if isinstance(record, dict)}
    result: dict[str, str] = {}
    for spec in specs:
        name = str(spec["name"])
        relative = f"{name}/cyl/photo_crop.jpg"
        digest = expected.get(relative)
        if not isinstance(digest, str) or len(digest) != 64:
            raise SystemExit(f"source atlas photo fingerprint is missing for: {name}")
        result[name] = digest
    return result


def _validate_photo_hashes(
    specs: list[dict[str, Any]],
    expected: dict[str, str],
    sample_dirs: dict[str, Path],
) -> None:
    for spec in specs:
        name = str(spec["name"])
        if file_sha256(sample_dirs[name] / "cyl" / "photo_crop.jpg") != expected[name]:
            raise SystemExit(f"source photo changed since the atlas run: {name}")


def _expected_cache_sha(source_run: dict[str, Any]) -> str:
    terrain_cache = source_run.get("terrain_cache")
    value = terrain_cache.get("aggregate_sha256") if isinstance(terrain_cache, dict) else None
    if not isinstance(value, str):
        raise SystemExit("source atlas terrain-cache fingerprint is missing")
    return value


def _compact_cache_inventory(inventory: dict[str, Any]) -> dict[str, Any]:
    caches: dict[str, Any] = {}
    for name, record in sorted(inventory.items()):
        files = record.get("files", []) if isinstance(record, dict) else []
        caches[name] = {
            "file_count": len(files),
            "total_size_bytes": sum(int(item.get("size", 0)) for item in files),
            "inventory_sha256": hashlib.sha256(_json_bytes(files)).hexdigest(),
        }
    return {
        "scope": "available_cache_inventory_not_exact_consumption",
        "caches": caches,
        "aggregate_sha256": hashlib.sha256(_json_bytes(inventory)).hexdigest(),
    }


def _implementation_record() -> dict[str, Any]:
    paths = [
        Path(__file__),
        BASE / "src/peakle/scripts/bench_pose_atlas.py",
        BASE / "src/peakle/localize/atlas_geometry.py",
        BASE / "src/peakle/localize/photo_geometry_verifier.py",
        BASE / "src/peakle/localize/skyline_atlas.py",
        BASE / "src/peakle/localize/extract.py",
        BASE / "src/peakle/segmentation.py",
        BASE / "src/peakle/depth.py",
        BASE / "src/peakle/edges.py",
        BASE / "src/peakle/localize/raycast.py",
        BASE / "src/peakle/localize/typed_outlines.py",
        BASE / "src/peakle/localize/strategy_bench.py",
        BASE / "src/peakle/localize/swissdem.py",
        BASE / "src/peakle/localize/paths.py",
        BASE / "src/peakle/terrain/copernicus.py",
        BASE / "src/peakle/domain/camera.py",
        BASE / "src/peakle/domain/coordinates.py",
        BASE / "src/peakle/domain/pose.py",
        BASE / "src/peakle/domain/projection.py",
        BASE / "src/peakle/domain/terrain.py",
    ]
    files = [{"path": str(path.relative_to(BASE)), "sha256": file_sha256(path)} for path in paths]
    relative = [record["path"] for record in files]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=BASE, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--", *relative], cwd=BASE, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    diff = subprocess.run(
        ["git", "diff", "--binary", "--", *relative], cwd=BASE, check=True, capture_output=True
    ).stdout
    return {
        "git_revision": revision,
        "aggregate_sha256": hashlib.sha256(_json_bytes(files)).hexdigest(),
        "files": files,
        "source_worktree_status": status,
        "tracked_source_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _verify_stability(
    *,
    atlas_path: Path,
    source_results_sha256: str,
    source_run_path: Path,
    source_run_sha256: str,
    estimator_specs: list[dict[str, Any]],
    expected_photo_hashes: dict[str, str],
    sample_dirs: dict[str, Path],
    cache_before: dict[str, Any],
    edge_model_before: dict[str, Any],
    edge_checkpoint: Path,
    depth_model_before: dict[str, Any],
    depth_model_dir: Path,
    implementation_before: dict[str, Any],
) -> None:
    changed: list[str] = []
    if file_sha256(atlas_path) != source_results_sha256:
        changed.append("source_atlas_results")
    if file_sha256(source_run_path) != source_run_sha256:
        changed.append("source_atlas_run_metadata")
    try:
        _validate_photo_hashes(estimator_specs, expected_photo_hashes, sample_dirs)
    except SystemExit:
        changed.append("source_photo_inputs")
    if _compact_cache_inventory(default_terrain_cache_inventory()) != cache_before:
        changed.append("terrain_cache")
    try:
        if _file_model_provenance("dexined", edge_checkpoint, kind="checkpoint") != edge_model_before:
            changed.append("dexined_checkpoint")
    except OSError, SystemExit:
        changed.append("dexined_checkpoint")
    try:
        if _directory_model_provenance("depth-anything", depth_model_dir) != depth_model_before:
            changed.append("depth_anything_model")
    except OSError, SystemExit:
        changed.append("depth_anything_model")
    if _implementation_record() != implementation_before:
        changed.append("selected_implementation_paths")
    if changed:
        raise RuntimeError(f"photo-verifier inputs changed during execution: {', '.join(changed)}")


def _environment_record() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "scipy", "pillow", "torch", "kornia", "transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def _observed_skyline_sha256(observed: NDArray[np.float64]) -> str:
    finite = np.isfinite(observed)
    normalized = np.where(finite, observed, 0.0).astype("<f8", copy=False)
    digest = hashlib.sha256()
    digest.update(b"peakle_observed_skyline_v1\0")
    digest.update(int(observed.size).to_bytes(8, "little", signed=False))
    digest.update(np.ascontiguousarray(finite.astype(np.uint8)).tobytes())
    digest.update(np.ascontiguousarray(normalized).tobytes())
    return digest.hexdigest()


def _normalize(values: NDArray[np.float64]) -> NDArray[np.float64]:
    low = float(np.percentile(values, 1.0))
    high = float(np.percentile(values, 99.0))
    if high <= low:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} is missing or malformed")
    return value


def _positive_record_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _json_object(content: bytes, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError as error:
        raise SystemExit(f"invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"expected a JSON object in {path}")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", required=True, help="completed pose-atlas results.json")
    parser.add_argument("--samples", default="", help="comma-separated subset; default is every source sample")
    parser.add_argument("--subsample", type=_positive_int, default=4)
    parser.add_argument("--dexined-checkpoint", required=True, help="local DexiNed state-dict file")
    parser.add_argument("--depth-model-dir", required=True, help="local Depth-Anything Transformers directory")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", required=True, help="new write-once output directory")
    return parser


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


if __name__ == "__main__":
    main()

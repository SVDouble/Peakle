"""Pinned, offline RoMa-architecture matcher for Peakle's JSON/NPZ contract.

The benchmark process deliberately does not import RoMa.  This worker verifies a
model manifest, a clean source checkout, and both checkpoint hashes before it adds
the checkout to ``sys.path``.  The official RoMa outdoor or compatible MINIMA
state dictionary and the DINOv2 state dictionary are passed explicitly to
``roma_outdoor``.  Torch Hub's download entry points are disabled, so inference
cannot turn a missing checkpoint into an unrecorded network fetch.

Example::

    python -m peakle.scripts.roma_match_worker --request /path/to/request.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image

REQUEST_SCHEMA = "peakle_match_batch_request_v1"
RESULT_SCHEMA = "peakle_match_batch_result_v1"
MANIFEST_SCHEMA = "peakle_roma_model_manifest_v1"
ALLOWED_MATCHER_IDS = frozenset({"roma_outdoor", "minima_roma"})
ROMA_WEIGHTS_ROLE = "roma_outdoor_weights"
DINO_WEIGHTS_ROLE = "dinov2_vitl14_weights"
REQUIRED_ARTIFACT_ROLES = frozenset({ROMA_WEIGHTS_ROLE, DINO_WEIGHTS_ROLE})
SELECTION_METHOD = "deterministic_joint_grid_confidence_v1"


class WorkerInputError(ValueError):
    """The request or manifest violates the worker contract."""


class MissingModelError(RuntimeError):
    """A required local repository, dependency path, or checkpoint is missing."""


class WorkerUnavailableError(RuntimeError):
    """The pinned model cannot run in this execution environment."""


@dataclass(frozen=True)
class VerifiedArtifact:
    role: str
    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class VerifiedRepository:
    path: Path
    commit: str
    remote: str | None
    python_paths: tuple[Path, ...]
    python_path_fingerprints: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class InferenceConfig:
    coarse_res: int | tuple[int, int]
    upsample_res: int | tuple[int, int]
    max_matches: int
    selection_cell_px: int
    min_confidence: float
    amp_dtype: str
    symmetric: bool
    use_custom_corr: bool
    upsample_preds: bool
    device: str

    def provenance(self) -> dict[str, Any]:
        return {
            "coarse_res": _resolution_record(self.coarse_res),
            "upsample_res": _resolution_record(self.upsample_res),
            "max_matches": self.max_matches,
            "selection_cell_px": self.selection_cell_px,
            "min_confidence": self.min_confidence,
            "amp_dtype": self.amp_dtype,
            "symmetric": self.symmetric,
            "use_custom_corr": self.use_custom_corr,
            "upsample_preds": self.upsample_preds,
            "device": self.device,
        }


@dataclass(frozen=True)
class VerifiedManifest:
    matcher_id: str
    path: Path
    sha256: str
    repository: VerifiedRepository
    artifacts: dict[str, VerifiedArtifact]
    inference: InferenceConfig


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    rgb: NDArray[np.uint8]
    sha256: str


@dataclass(frozen=True)
class SelectedMatches:
    query_xy_px: NDArray[np.float32]
    render_xy_px: NDArray[np.float32]
    confidence: NDArray[np.float32]
    dense_candidates: int
    valid_candidates: int
    unique_candidates: int


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True, help="peakle_match_batch_request_v1 JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request_path = args.request.expanduser().resolve()
    try:
        raw_request = _read_json_object(request_path, "matcher request")
        result_path = _declared_result_path(request_path.parent, raw_request)
    except (OSError, json.JSONDecodeError, WorkerInputError) as exc:
        print(f"invalid matcher request: {exc}", file=sys.stderr)
        return 2

    started = time.perf_counter()
    try:
        result = run_request(request_path, raw_request)
    except MissingModelError as exc:
        result = _failure_result("SKIPPED_MISSING_MODEL", exc, started)
    except WorkerUnavailableError as exc:
        result = _failure_result("SKIPPED_UNAVAILABLE", exc, started)
    except Exception as exc:  # noqa: BLE001 - preserve a structured worker failure for the caller
        result = _failure_result("error", exc, started, include_traceback=True)
    _atomic_write_json(result_path, result)
    return 0


def run_request(request_path: Path, request: dict[str, Any]) -> dict[str, Any]:
    """Validate one request, run all render pairs, and return result metadata."""

    started = time.perf_counter()
    root = request_path.parent.resolve()
    matcher_id = _validate_request_header(request)
    manifest = _verify_manifest(request["model_manifest"], matcher_id)
    query = _load_image_record(root, request["query"], "query")
    renders = request.get("renders")
    if not isinstance(renders, list) or not renders:
        raise WorkerInputError("request renders must be a non-empty list")
    ids = [_render_id(record, index) for index, record in enumerate(renders)]
    if len(set(ids)) != len(ids):
        raise WorkerInputError("every render needs a unique JSON-serializable id")

    _configure_offline_environment(int(request["seed"]))
    torch = _import_torch()
    device = _resolve_device(torch, manifest.inference.device)
    _seed_everything(torch, int(request["seed"]))
    load_started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with _blocked_torch_downloads(torch):
        model = _load_model(torch, manifest, device)
    _synchronize(torch, device)
    load_runtime_s = time.perf_counter() - load_started
    model_load_memory = _memory_record(torch, device)

    per_render: list[dict[str, Any]] = []
    output_paths: list[Path] = []
    try:
        for index, raw_render_request in enumerate(renders):
            assert isinstance(raw_render_request, dict)
            render_request = cast(dict[str, Any], raw_render_request)
            render = _load_image_record(root, render_request, f"render[{index}]")
            output_path = _job_path(root, render_request.get("matches_npz"), f"render[{index}] matches_npz")
            if output_path.exists():
                raise WorkerInputError(f"refusing to overwrite declared matcher output: {output_path.name}")
            render_seed = _derived_seed(int(request["seed"]), render_request["id"])
            _seed_everything(torch, render_seed)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            pair_started = time.perf_counter()
            with _blocked_torch_downloads(torch):
                warp, certainty = model.match(
                    Image.fromarray(query.rgb, mode="RGB"),
                    Image.fromarray(render.rgb, mode="RGB"),
                    device=device,
                )
            _synchronize(torch, device)
            matches = _deterministic_matches(
                warp,
                certainty,
                query_shape=query.rgb.shape,
                render_shape=render.rgb.shape,
                config=manifest.inference,
            )
            pair_runtime_s = time.perf_counter() - pair_started
            selected = np.ones(matches.confidence.shape[0], dtype=np.bool_)
            _atomic_write_npz(
                output_path,
                query_xy_px=matches.query_xy_px,
                render_xy_px=matches.render_xy_px,
                confidence=matches.confidence,
                selected=selected,
            )
            output_paths.append(output_path)
            per_render.append(
                {
                    "id": render_request["id"],
                    "index": index,
                    "input_path": render.path.name,
                    "input_sha256": render.sha256,
                    "input_shape": list(render.rgb.shape),
                    "output_npz": output_path.name,
                    "output_sha256": _file_sha256(output_path),
                    "runtime_s": pair_runtime_s,
                    "seed": render_seed,
                    "dense_candidates": matches.dense_candidates,
                    "valid_candidates": matches.valid_candidates,
                    "unique_candidates": matches.unique_candidates,
                    "returned_matches": int(matches.confidence.size),
                    "confidence": _confidence_record(matches.confidence),
                    "memory": _memory_record(torch, device),
                    "sampling": {
                        "method": SELECTION_METHOD,
                        "stochastic": False,
                        "max_matches": manifest.inference.max_matches,
                        "selection_cell_px": manifest.inference.selection_cell_px,
                        "min_confidence": manifest.inference.min_confidence,
                    },
                    "coordinates": "original-resolution top-left x/y pixel centres",
                }
            )
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Importing may create ignored bytecode but may never modify tracked or untracked source files.
    final_repository = _verify_repository(
        {
            "path": str(manifest.repository.path),
            "commit": manifest.repository.commit,
            "require_clean": True,
            "python_paths": [str(path) for path in manifest.repository.python_paths],
        },
        manifest.path.parent,
    )
    result = {
        "schema": RESULT_SCHEMA,
        "status": "ok",
        "matcher_id": manifest.matcher_id,
        "request_sha256": _file_sha256(request_path),
        "query": {
            "path": query.path.name,
            "shape": list(query.rgb.shape),
            "sha256": query.sha256,
        },
        "renders": per_render,
        "runtime": {
            "model_load_s": load_runtime_s,
            "pairs_s": sum(float(record["runtime_s"]) for record in per_render),
            "total_s": time.perf_counter() - started,
        },
        "model_load_memory": model_load_memory,
        "provenance": _provenance(torch, device, manifest, final_repository),
        "outputs": [
            {"path": path.name, "sha256": _file_sha256(path), "size_bytes": path.stat().st_size}
            for path in output_paths
        ],
    }
    return result


def _validate_request_header(request: dict[str, Any]) -> str:
    if request.get("schema") != REQUEST_SCHEMA:
        raise WorkerInputError(f"request schema must be {REQUEST_SCHEMA!r}")
    matcher_id = request.get("matcher_id")
    if matcher_id not in ALLOWED_MATCHER_IDS:
        raise WorkerInputError(f"this worker only accepts matcher_id in {sorted(ALLOWED_MATCHER_IDS)!r}")
    seed = request.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise WorkerInputError("request seed must be an integer in [0, 2**63)")
    contract = request.get("contract")
    if not isinstance(contract, dict):
        raise WorkerInputError("request contract must be an object")
    if contract.get("network_allowed") is not False:
        raise WorkerInputError("RoMa worker requires contract.network_allowed=false")
    if contract.get("implicit_model_downloads_allowed") is not False:
        raise WorkerInputError("RoMa worker requires contract.implicit_model_downloads_allowed=false")
    if contract.get("coordinates") != "original-resolution top-left x/y pixel centres":
        raise WorkerInputError("unsupported match-coordinate contract")
    if not isinstance(request.get("model_manifest"), dict):
        raise WorkerInputError("request model_manifest must be an object")
    if not isinstance(request.get("query"), dict):
        raise WorkerInputError("request query must be an object")
    assert isinstance(matcher_id, str)
    return matcher_id


def _verify_manifest(
    payload: dict[str, Any],
    expected_matcher_id: str,
) -> VerifiedManifest:
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise WorkerInputError(f"model manifest schema must be {MANIFEST_SCHEMA!r}")
    if payload.get("matcher_id") != expected_matcher_id:
        raise WorkerInputError("request matcher_id and normalized model manifest matcher_id differ")
    manifest_path_value = payload.get("manifest_path")
    if not isinstance(manifest_path_value, str) or not manifest_path_value:
        raise WorkerInputError("normalized model manifest must include manifest_path")
    manifest_path = Path(manifest_path_value).expanduser().resolve()
    if not manifest_path.is_file():
        raise MissingModelError(f"model manifest is missing: {manifest_path}")
    manifest_parent = manifest_path.parent
    source_payload = _read_json_object(manifest_path, "model manifest")
    if source_payload.get("schema") != MANIFEST_SCHEMA:
        raise WorkerInputError("model manifest file has an unsupported schema")
    if source_payload.get("matcher_id") != expected_matcher_id:
        raise WorkerInputError("request, normalized manifest, and source manifest matcher_id must match")
    _verify_normalized_manifest(payload, source_payload, manifest_path)
    repository_payload = payload.get("repository")
    if not isinstance(repository_payload, dict):
        raise WorkerInputError("model manifest repository must be an object")
    repository = _verify_repository(repository_payload, manifest_parent)

    records = payload.get("artifacts")
    if not isinstance(records, list):
        raise WorkerInputError("model manifest artifacts must be a list")
    artifacts: dict[str, VerifiedArtifact] = {}
    for record in records:
        artifact = _verify_artifact(record, manifest_parent)
        if artifact.role in artifacts:
            raise WorkerInputError(f"duplicate model artifact role: {artifact.role}")
        artifacts[artifact.role] = artifact
    missing = REQUIRED_ARTIFACT_ROLES - artifacts.keys()
    if missing:
        raise WorkerInputError(f"model manifest is missing required artifact role(s): {sorted(missing)}")
    inference = _inference_config(payload.get("inference", {}))
    return VerifiedManifest(
        matcher_id=expected_matcher_id,
        path=manifest_path,
        sha256=_file_sha256(manifest_path),
        repository=repository,
        artifacts=artifacts,
        inference=inference,
    )


def _verify_normalized_manifest(
    payload: dict[str, Any],
    source_payload: dict[str, Any],
    manifest_path: Path,
) -> None:
    """Require the request copy to be exactly the source manifest after path normalization."""

    source_artifacts = source_payload.get("artifacts")
    if not isinstance(source_artifacts, list):
        raise WorkerInputError("source model manifest artifacts must be a list")
    normalized_artifacts: list[dict[str, Any]] = []
    for index, record in enumerate(source_artifacts):
        if not isinstance(record, dict):
            raise WorkerInputError(f"source model artifact {index} must be an object")
        typed_record = cast(dict[str, Any], record)
        normalized_artifacts.append(
            {
                **typed_record,
                "path": str(
                    _manifest_path(
                        manifest_path.parent,
                        typed_record.get("path"),
                        f"source artifact {index}",
                    )
                ),
            }
        )
    expected = {
        **source_payload,
        "artifacts": normalized_artifacts,
        "manifest_path": str(manifest_path),
    }
    if payload != expected:
        differing = sorted(key for key in set(payload) | set(expected) if payload.get(key) != expected.get(key))
        raise WorkerInputError(f"normalized request manifest differs from its source file (fields: {differing})")


def _verify_artifact(record: Any, manifest_parent: Path) -> VerifiedArtifact:
    if not isinstance(record, dict):
        raise WorkerInputError("every model artifact must be an object")
    role = record.get("role")
    if not isinstance(role, str) or not role:
        raise WorkerInputError("every model artifact needs a non-empty role")
    path = _manifest_path(manifest_parent, record.get("path"), f"artifact {role}")
    if not path.is_file():
        raise MissingModelError(f"model artifact is missing: {path}")
    expected_sha = record.get("sha256")
    if not isinstance(expected_sha, str) or not _is_sha256(expected_sha):
        raise WorkerInputError(f"model artifact {role} needs a lowercase SHA-256")
    size = record.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise WorkerInputError(f"model artifact {role} needs a positive size_bytes")
    actual_size = path.stat().st_size
    if actual_size != size:
        raise MissingModelError(f"model artifact size mismatch for {role}: expected {size}, got {actual_size}")
    actual_sha = _file_sha256(path)
    if actual_sha != expected_sha:
        raise MissingModelError(f"model artifact hash mismatch for {role}: expected {expected_sha}, got {actual_sha}")
    return VerifiedArtifact(role=role, path=path, sha256=actual_sha, size_bytes=actual_size)


def _verify_repository(payload: dict[str, Any], manifest_parent: Path) -> VerifiedRepository:
    path = _manifest_path(manifest_parent, payload.get("path"), "RoMa repository")
    if not path.is_dir() or not (path / ".git").exists():
        raise MissingModelError(f"RoMa repository is not a git checkout: {path}")
    expected = payload.get("commit")
    if not isinstance(expected, str) or len(expected) != 40 or any(char not in "0123456789abcdef" for char in expected):
        raise WorkerInputError("RoMa repository commit must be a full lowercase 40-character hash")
    if payload.get("require_clean") is not True:
        raise WorkerInputError("RoMa repository require_clean must be true")
    actual = _git(path, "rev-parse", "HEAD")
    if actual != expected:
        raise MissingModelError(f"RoMa repository commit mismatch: expected {expected}, got {actual}")
    status = _git(path, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        detail = status.splitlines()[0][:160]
        raise MissingModelError(f"RoMa repository is dirty: {detail}")
    remote = _git(path, "config", "--get", "remote.origin.url", allow_failure=True) or None
    python_path_records = payload.get("python_paths", [])
    if not isinstance(python_path_records, list) or any(not isinstance(value, str) for value in python_path_records):
        raise WorkerInputError("RoMa repository python_paths must be a list of paths")
    python_paths = tuple(
        _manifest_path(manifest_parent, value, f"repository.python_paths[{index}]")
        for index, value in enumerate(python_path_records)
    )
    for python_path in python_paths:
        if not python_path.is_dir():
            raise MissingModelError(f"RoMa dependency search path is missing: {python_path}")
    fingerprints = tuple(_tree_fingerprint(value) for value in python_paths)
    return VerifiedRepository(
        path=path,
        commit=actual,
        remote=remote,
        python_paths=python_paths,
        python_path_fingerprints=fingerprints,
    )


def _inference_config(value: Any) -> InferenceConfig:
    if not isinstance(value, dict):
        raise WorkerInputError("model manifest inference must be an object")
    allowed = {
        "coarse_res",
        "upsample_res",
        "max_matches",
        "selection_cell_px",
        "min_confidence",
        "amp_dtype",
        "symmetric",
        "use_custom_corr",
        "upsample_preds",
        "device",
    }
    unknown = set(value) - allowed
    if unknown:
        raise WorkerInputError(f"unknown RoMa inference setting(s): {sorted(unknown)}")
    coarse = _resolution(value.get("coarse_res", 560), "coarse_res", multiple=14)
    upsample = _resolution(value.get("upsample_res", 864), "upsample_res", multiple=1)
    max_matches = _bounded_int(value.get("max_matches", 5000), "max_matches", 1, 50_000)
    cell_px = _bounded_int(value.get("selection_cell_px", 16), "selection_cell_px", 2, 256)
    min_confidence = value.get("min_confidence", 0.0)
    if isinstance(min_confidence, bool) or not isinstance(min_confidence, (int, float)):
        raise WorkerInputError("min_confidence must be numeric")
    min_confidence = float(min_confidence)
    if not 0.0 <= min_confidence <= 1.0:
        raise WorkerInputError("min_confidence must be in [0, 1]")
    amp_dtype = value.get("amp_dtype", "float16")
    if amp_dtype not in {"float16", "bfloat16", "float32"}:
        raise WorkerInputError("amp_dtype must be float16, bfloat16, or float32")
    device = value.get("device", "cuda")
    if not isinstance(device, str) or not (device == "cpu" or device == "cuda" or device.startswith("cuda:")):
        raise WorkerInputError("device must be cpu, cuda, or cuda:<index>")
    return InferenceConfig(
        coarse_res=coarse,
        upsample_res=upsample,
        max_matches=max_matches,
        selection_cell_px=cell_px,
        min_confidence=min_confidence,
        amp_dtype=amp_dtype,
        symmetric=_strict_bool(value.get("symmetric", True), "symmetric"),
        use_custom_corr=_strict_bool(value.get("use_custom_corr", False), "use_custom_corr"),
        upsample_preds=_strict_bool(value.get("upsample_preds", True), "upsample_preds"),
        device=device,
    )


def _load_model(torch: Any, manifest: VerifiedManifest, device: Any) -> Any:
    for path in reversed((manifest.repository.path, *manifest.repository.python_paths)):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    try:
        romatch = importlib.import_module("romatch")
        roma_outdoor = romatch.roma_outdoor
    except ImportError as exc:
        raise WorkerUnavailableError(f"cannot import pinned RoMa checkout: {exc}") from exc
    module_path = romatch.__file__
    if not isinstance(module_path, str):
        raise WorkerUnavailableError("pinned RoMa module has no filesystem origin")
    imported_from = Path(module_path).resolve()
    if not imported_from.is_relative_to(manifest.repository.path):
        raise WorkerUnavailableError(f"romatch resolved outside the verified repository: {imported_from}")
    roma_weights = _load_state_dict(torch, manifest.artifacts[ROMA_WEIGHTS_ROLE].path)
    dino_weights = _load_state_dict(torch, manifest.artifacts[DINO_WEIGHTS_ROLE].path)
    dtype = getattr(torch, manifest.inference.amp_dtype)
    if device.type == "cpu":
        dtype = torch.float32
    torch.set_float32_matmul_precision("highest")
    try:
        return roma_outdoor(
            device=device,
            weights=roma_weights,
            dinov2_weights=dino_weights,
            coarse_res=manifest.inference.coarse_res,
            upsample_res=manifest.inference.upsample_res,
            amp_dtype=dtype,
            symmetric=manifest.inference.symmetric,
            use_custom_corr=manifest.inference.use_custom_corr,
            upsample_preds=manifest.inference.upsample_preds,
            with_padding=False,
            do_compile=False,
        ).eval()
    except ImportError as exc:
        raise WorkerUnavailableError(f"cannot construct pinned RoMa model: {exc}") from exc
    except RuntimeError as exc:
        # A CUDA OOM means the configured matcher was available and inference was
        # attempted.  Preserve it as an execution error instead of a skipped backend.
        if _is_cuda_oom(exc):
            raise
        raise WorkerUnavailableError(f"cannot construct pinned RoMa model: {exc}") from exc


def _load_state_dict(torch: Any, path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise MissingModelError(f"cannot load verified model artifact {path.name}: {exc}") from exc


def _deterministic_matches(
    warp: Any,
    certainty: Any,
    *,
    query_shape: tuple[int, ...],
    render_shape: tuple[int, ...],
    config: InferenceConfig,
) -> SelectedMatches:
    normalized = np.asarray(warp.detach().float().cpu(), dtype=np.float64).reshape(-1, 4)
    confidence = np.asarray(certainty.detach().float().cpu(), dtype=np.float64).reshape(-1)
    dense_count = int(confidence.size)
    if normalized.shape[0] != dense_count:
        raise RuntimeError(
            f"RoMa warp/certainty size mismatch: {normalized.shape[0]} coordinates vs {dense_count} certainties"
        )
    finite = np.all(np.isfinite(normalized), axis=1) & np.isfinite(confidence)
    # RoMa clamps invalid predictions exactly onto +/-1. Its regular pixel-centre grid is strictly inside.
    inside = np.all(np.abs(normalized) < 1.0 - 1e-9, axis=1)
    keep = finite & inside & (confidence >= config.min_confidence)
    normalized = normalized[keep]
    confidence = confidence[keep]
    valid_count = int(confidence.size)
    if valid_count == 0:
        return SelectedMatches(
            query_xy_px=np.empty((0, 2), dtype=np.float32),
            render_xy_px=np.empty((0, 2), dtype=np.float32),
            confidence=np.empty(0, dtype=np.float32),
            dense_candidates=dense_count,
            valid_candidates=0,
            unique_candidates=0,
        )
    query_xy = _normalized_to_pixels(normalized[:, :2], query_shape[1], query_shape[0])
    render_xy = _normalized_to_pixels(normalized[:, 2:], render_shape[1], render_shape[0])
    in_original = (
        (query_xy[:, 0] >= 0.0)
        & (query_xy[:, 0] < query_shape[1])
        & (query_xy[:, 1] >= 0.0)
        & (query_xy[:, 1] < query_shape[0])
        & (render_xy[:, 0] >= 0.0)
        & (render_xy[:, 0] < render_shape[1])
        & (render_xy[:, 1] >= 0.0)
        & (render_xy[:, 1] < render_shape[0])
    )
    query_xy = query_xy[in_original]
    render_xy = render_xy[in_original]
    confidence = confidence[in_original]
    chosen, unique_count = _joint_grid_select(
        query_xy,
        render_xy,
        confidence,
        max_matches=config.max_matches,
        cell_px=config.selection_cell_px,
    )
    return SelectedMatches(
        query_xy_px=query_xy[chosen].astype(np.float32),
        render_xy_px=render_xy[chosen].astype(np.float32),
        confidence=confidence[chosen].astype(np.float32),
        dense_candidates=dense_count,
        valid_candidates=valid_count,
        unique_candidates=unique_count,
    )


def _joint_grid_select(
    query_xy: NDArray[np.float64],
    render_xy: NDArray[np.float64],
    confidence: NDArray[np.float64],
    *,
    max_matches: int,
    cell_px: int,
) -> tuple[NDArray[np.int64], int]:
    if confidence.size == 0:
        return np.empty(0, dtype=np.int64), 0
    # Stable confidence order with a complete coordinate tie-break makes the output platform-independent.
    order = np.lexsort((render_xy[:, 1], render_xy[:, 0], query_xy[:, 1], query_xy[:, 0], -confidence))
    # Symmetric RoMa inference can emit the same A/B point pair twice. Quantize to 1/1024 px only for de-duplication.
    seen_pairs: set[tuple[int, int, int, int]] = set()
    unique_order: list[int] = []
    for raw_index in order:
        index = int(raw_index)
        key = (
            int(round(query_xy[index, 0] * 1024.0)),
            int(round(query_xy[index, 1] * 1024.0)),
            int(round(render_xy[index, 0] * 1024.0)),
            int(round(render_xy[index, 1] * 1024.0)),
        )
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        unique_order.append(index)
    if len(unique_order) <= max_matches:
        return np.asarray(unique_order, dtype=np.int64), len(unique_order)

    q_cells = np.floor(query_xy / cell_px).astype(np.int64)
    r_cells = np.floor(render_xy / cell_px).astype(np.int64)
    q_counts: dict[tuple[int, int], int] = {}
    r_counts: dict[tuple[int, int], int] = {}
    selected: list[int] = []
    selected_set: set[int] = set()
    # Progressive per-cell caps retain high confidence while spreading matches over both images.
    cap = 1
    while len(selected) < max_matches:
        for index in unique_order:
            if index in selected_set:
                continue
            q_key = (int(q_cells[index, 0]), int(q_cells[index, 1]))
            r_key = (int(r_cells[index, 0]), int(r_cells[index, 1]))
            if q_counts.get(q_key, 0) >= cap or r_counts.get(r_key, 0) >= cap:
                continue
            selected.append(index)
            selected_set.add(index)
            q_counts[q_key] = q_counts.get(q_key, 0) + 1
            r_counts[r_key] = r_counts.get(r_key, 0) + 1
            if len(selected) == max_matches:
                break
        cap += 1
        if cap > len(unique_order):
            break
    return np.asarray(selected, dtype=np.int64), len(unique_order)


def _normalized_to_pixels(
    normalized_xy: NDArray[np.float64],
    width: int,
    height: int,
) -> NDArray[np.float64]:
    pixels = np.empty_like(normalized_xy, dtype=np.float64)
    pixels[:, 0] = width * 0.5 * (normalized_xy[:, 0] + 1.0)
    pixels[:, 1] = height * 0.5 * (normalized_xy[:, 1] + 1.0)
    return pixels


def _load_image_record(root: Path, record: Any, label: str) -> ImageRecord:
    if not isinstance(record, dict):
        raise WorkerInputError(f"{label} record must be an object")
    path = _job_path(root, record.get("path"), f"{label} path")
    if not path.is_file():
        raise WorkerInputError(f"{label} image is missing: {path.name}")
    try:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise WorkerInputError(f"cannot decode {label} image: {exc}") from exc
    shape = record.get("shape")
    if shape != list(rgb.shape):
        raise WorkerInputError(f"{label} shape mismatch: request {shape}, decoded {list(rgb.shape)}")
    expected_sha = record.get("sha256")
    if not isinstance(expected_sha, str) or not _is_sha256(expected_sha):
        raise WorkerInputError(f"{label} needs a lowercase raw-RGB SHA-256")
    actual_sha = hashlib.sha256(memoryview(np.ascontiguousarray(rgb)).cast("B")).hexdigest()
    if actual_sha != expected_sha:
        raise WorkerInputError(f"{label} raw-RGB hash mismatch: expected {expected_sha}, got {actual_sha}")
    return ImageRecord(path=path, rgb=rgb, sha256=actual_sha)


def _configure_offline_environment(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["PYTHONHASHSEED"] = str(seed)


def _import_torch() -> Any:
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:
        raise WorkerUnavailableError("PyTorch is unavailable in the matcher worker environment") from exc
    return torch


def _resolve_device(torch: Any, configured: str) -> Any:
    device = torch.device(configured)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise WorkerUnavailableError(f"configured CUDA device is unavailable: {configured}")
        try:
            torch.cuda.get_device_properties(device)
        except (AssertionError, RuntimeError) as exc:
            raise WorkerUnavailableError(f"configured CUDA device is unavailable: {configured}") from exc
    return device


def _seed_everything(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


@contextmanager
def _blocked_torch_downloads(torch: Any) -> Iterator[None]:
    original_state_dict = torch.hub.load_state_dict_from_url
    original_hub_load = torch.hub.load

    def blocked(*_args: Any, **_kwargs: Any) -> NoReturn:
        raise RuntimeError("network/model downloads are disabled by the Peakle matcher contract")

    torch.hub.load_state_dict_from_url = blocked
    torch.hub.load = blocked
    try:
        yield
    finally:
        torch.hub.load_state_dict_from_url = original_state_dict
        torch.hub.load = original_hub_load


def _provenance(
    torch: Any,
    device: Any,
    manifest: VerifiedManifest,
    repository: VerifiedRepository,
) -> dict[str, Any]:
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu = {
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "capability": list(torch.cuda.get_device_capability(device)),
            "cuda_runtime": torch.version.cuda,
            "cudnn": int(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None,
        }
    else:
        gpu = None
    return {
        "worker": {
            "schema": RESULT_SCHEMA,
            "implementation": str(Path(__file__).resolve()),
            "implementation_sha256": _file_sha256(Path(__file__).resolve()),
            "sampling_method": SELECTION_METHOD,
            "network_allowed": False,
            "implicit_model_downloads_allowed": False,
            "torch_hub_downloads_blocked": True,
        },
        "manifest": {
            "path": str(manifest.path),
            "sha256": manifest.sha256,
            "schema": MANIFEST_SCHEMA,
        },
        "repository": {
            "path": str(repository.path),
            "commit": repository.commit,
            "clean_before_and_after": True,
            "remote": repository.remote,
            "python_paths": list(repository.python_path_fingerprints),
        },
        "artifacts": [
            {
                "role": artifact.role,
                "path": str(artifact.path),
                "sha256": artifact.sha256,
                "size_bytes": artifact.size_bytes,
            }
            for artifact in sorted(manifest.artifacts.values(), key=lambda item: item.role)
        ],
        "inference": manifest.inference.provenance(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pillow": _package_version("pillow"),
            "torch": torch.__version__,
            "torchvision": _package_version("torchvision"),
            "romatch": _package_version("romatch"),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "device": {"requested": manifest.inference.device, "effective": str(device), "gpu": gpu},
    }


def _memory_record(torch: Any, device: Any) -> dict[str, int | None]:
    if device.type != "cuda":
        return {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _synchronize(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _confidence_record(confidence: NDArray[np.float32]) -> dict[str, float | None]:
    if confidence.size == 0:
        return {"min": None, "median": None, "max": None, "mean": None}
    return {
        "min": float(np.min(confidence)),
        "median": float(np.median(confidence)),
        "max": float(np.max(confidence)),
        "mean": float(np.mean(confidence)),
    }


def _failure_result(
    status: str,
    exc: Exception,
    started: float,
    *,
    include_traceback: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": status,
        "message": str(exc),
        "error_type": type(exc).__name__,
        "runtime": {"total_s": time.perf_counter() - started},
    }
    if include_traceback:
        result["traceback"] = traceback.format_exc(limit=12)
    return result


def _declared_result_path(root: Path, request: dict[str, Any]) -> Path:
    outputs = request.get("outputs")
    if not isinstance(outputs, dict):
        raise WorkerInputError("request outputs must be an object")
    return _job_path(root.resolve(), outputs.get("result_json"), "outputs.result_json")


def _job_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise WorkerInputError(f"{label} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute():
        raise WorkerInputError(f"{label} must be relative to the request directory")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise WorkerInputError(f"{label} escapes the request directory")
    return resolved


def _manifest_path(parent: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise WorkerInputError(f"{label} path must be a non-empty string")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (parent / path).resolve()


def _resolution(value: Any, label: str, *, multiple: int) -> int | tuple[int, int]:
    if isinstance(value, bool):
        raise WorkerInputError(f"{label} must be an integer or [height, width]")
    if isinstance(value, int):
        values = (value,)
        result: int | tuple[int, int] = value
    elif (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    ):
        values = tuple(value)
        result = (value[0], value[1])
    else:
        raise WorkerInputError(f"{label} must be an integer or [height, width]")
    if any(item < 64 or item > 4096 or item % multiple != 0 for item in values):
        raise WorkerInputError(f"{label} values must be in [64, 4096] and divisible by {multiple}")
    return result


def _resolution_record(value: int | tuple[int, int]) -> int | list[int]:
    return value if isinstance(value, int) else list(value)


def _bounded_int(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise WorkerInputError(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def _strict_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise WorkerInputError(f"{label} must be boolean")
    return value


def _is_cuda_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "cuda" in message and "out of memory" in message


def _render_id(record: Any, index: int) -> str:
    if not isinstance(record, dict) or "id" not in record:
        raise WorkerInputError(f"render[{index}] needs a JSON-serializable id")
    try:
        return json.dumps(
            record["id"],
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise WorkerInputError(f"render[{index}] needs a JSON-serializable id") from exc


def _derived_seed(seed: int, render_id: Any) -> int:
    encoded = json.dumps(
        {"seed": seed, "render_id": render_id},
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % (2**63)


def _tree_fingerprint(path: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for file_path in sorted(path.rglob("*")):
        if not file_path.is_file() or "__pycache__" in file_path.parts or file_path.suffix == ".pyc":
            continue
        records.append(
            {
                "path": str(file_path.relative_to(path)),
                "size_bytes": file_path.stat().st_size,
                "sha256": _file_sha256(file_path),
            }
        )
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return {
        "path": str(path),
        "file_count": len(records),
        "tree_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _git(path: Path, *args: str, allow_failure: bool = False) -> str:
    completed = subprocess.run(
        ("git", "-C", str(path), *args),
        check=False,
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    if completed.returncode != 0:
        if allow_failure:
            return ""
        detail = (completed.stderr or completed.stdout).strip()
        raise MissingModelError(f"cannot verify RoMa repository: {detail or 'git failed'}")
    return completed.stdout.strip()


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise WorkerInputError(f"{label} must be a JSON object")
    return payload


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_write_npz(
    path: Path,
    *,
    query_xy_px: NDArray[Any],
    render_xy_px: NDArray[Any],
    confidence: NDArray[Any],
    selected: NDArray[Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary,
        query_xy_px=query_xy_px,
        render_xy_px=render_xy_px,
        confidence=confidence,
        selected=selected,
    )
    temporary.replace(path)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())

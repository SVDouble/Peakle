"""Deterministic image-correspondence contracts for render matching.

Learned matchers deliberately run out of process: Peakle targets Python 3.14,
whereas the official MINIMA release pins an older Torch stack.  The NPZ/JSON
contract below makes model identity, resizing, and failures visible to the
benchmark instead of relying on an undeclared import or implicit download.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import SIFT, match_descriptors
from skimage.transform import resize

BATCH_RESULT_SCHEMA = "peakle_match_batch_result_v1"
MATCH_REQUEST_SCHEMA = "peakle_match_batch_request_v1"
MATCH_COORDINATE_CONTRACT = "original-resolution top-left x/y pixel centres"
CORRESPONDENCE_CACHE_KEY_SCHEMA = "peakle_correspondence_cache_key_v1"
CORRESPONDENCE_CACHE_ENTRY_SCHEMA = "peakle_correspondence_cache_entry_v1"
CORRESPONDENCE_CACHE_BATCH_SCHEMA = "peakle_correspondence_cache_batch_v1"
# Retain the original v1 policy value because it is itself part of existing
# cache-key material. Despite the historical name, deriving this ID does not
# require a cache directory and never writes to one.
WORKER_RENDER_ID_POLICY = "correspondence_cache_key_sha256"
WORKER_RENDER_ID_POLICY_VERSION = 1


class MatcherUnavailable(RuntimeError):
    """A configured matcher cannot run with the provisioned environment."""


class MatcherExecutionError(RuntimeError):
    """A provisioned matcher started but violated its inference contract."""


@dataclass(frozen=True)
class MatchSet:
    """Pixel correspondences in the original query and render resolutions."""

    query_xy_px: NDArray[np.float64]
    render_xy_px: NDArray[np.float64]
    confidence: NDArray[np.float64]
    selected: NDArray[np.bool_] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        query = np.asarray(self.query_xy_px, dtype=np.float64)
        render = np.asarray(self.render_xy_px, dtype=np.float64)
        confidence = np.asarray(self.confidence, dtype=np.float64)
        if query.ndim != 2 or query.shape[1] != 2:
            raise ValueError(f"query matches must have shape (N, 2), got {query.shape}")
        if render.shape != query.shape:
            raise ValueError(f"render matches must have shape {query.shape}, got {render.shape}")
        if confidence.shape != (query.shape[0],):
            raise ValueError(f"match confidence must have shape {(query.shape[0],)}, got {confidence.shape}")
        if not np.all(np.isfinite(query)) or not np.all(np.isfinite(render)):
            raise ValueError("match coordinates must be finite")
        if not np.all(np.isfinite(confidence)):
            raise ValueError("match confidence must be finite")
        selected = (
            np.ones(query.shape[0], dtype=np.bool_)
            if self.selected is None
            else np.asarray(self.selected, dtype=np.bool_)
        )
        if selected.shape != (query.shape[0],):
            raise ValueError(f"selected mask must have shape {(query.shape[0],)}, got {selected.shape}")
        object.__setattr__(self, "query_xy_px", query)
        object.__setattr__(self, "render_xy_px", render)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "selected", selected)

    @property
    def count(self) -> int:
        return int(self.query_xy_px.shape[0])

    def chosen(self) -> MatchSet:
        """Return only correspondences selected by the matcher worker."""

        selected = np.asarray(self.selected, dtype=bool)
        return MatchSet(
            query_xy_px=self.query_xy_px[selected],
            render_xy_px=self.render_xy_px[selected],
            confidence=self.confidence[selected],
            selected=np.ones(int(selected.sum()), dtype=np.bool_),
            diagnostics={**self.diagnostics, "worker_total": self.count, "worker_selected": int(selected.sum())},
            provenance=self.provenance,
        )


class DenseMatcher(Protocol):
    """Interchangeable query-to-render matcher."""

    def identity(self) -> dict[str, Any]:
        """Return stable matcher/model/preprocessing provenance."""

    def match(self, query_rgb: NDArray[np.uint8], render_rgb: NDArray[np.uint8]) -> MatchSet:
        """Match original-resolution query pixels to render pixels."""


def match_image_fan(
    matcher: DenseMatcher,
    query_rgb: NDArray[np.uint8],
    render_images: list[NDArray[np.uint8]],
) -> list[MatchSet]:
    """Use a worker's optional batch path, with a deterministic scalar fallback."""

    batch_method = getattr(matcher, "match_many", None)
    if callable(batch_method):
        matches = list(batch_method(query_rgb, render_images))
    else:
        matches = [matcher.match(query_rgb, render) for render in render_images]
    if len(matches) != len(render_images):
        raise MatcherExecutionError(
            f"matcher returned {len(matches)} result sets for {len(render_images)} render images"
        )
    return matches


@dataclass(frozen=True)
class SiftMatcher:
    """Hermetic same-modality control, not a learned cross-modal production path."""

    max_dimension_px: int = 1024
    max_ratio: float = 0.78
    max_matches: int = 1200
    upsampling: int = 1

    def identity(self) -> dict[str, Any]:
        return {
            "id": "skimage_sift_same_modality_control",
            "kind": "in_process_control",
            "cross_modal": False,
            "ranking_eligible": False,
            "implementation": "skimage.feature.SIFT + mutual ratio matching",
            "scikit_image_version": _package_version("scikit-image"),
            "parameters": {
                "max_dimension_px": self.max_dimension_px,
                "max_ratio": self.max_ratio,
                "max_matches": self.max_matches,
                "upsampling": self.upsampling,
            },
        }

    def match(self, query_rgb: NDArray[np.uint8], render_rgb: NDArray[np.uint8]) -> MatchSet:
        query = _rgb_image(query_rgb, "query")
        render = _rgb_image(render_rgb, "render")
        query_gray, query_scale = _matcher_gray(query, self.max_dimension_px)
        render_gray, render_scale = _matcher_gray(render, self.max_dimension_px)
        query_sift = SIFT(upsampling=self.upsampling)
        render_sift = SIFT(upsampling=self.upsampling)
        try:
            query_sift.detect_and_extract(query_gray)
            render_sift.detect_and_extract(render_gray)
        except RuntimeError as exc:
            return _empty_matches(
                self.identity(),
                reason="sift_no_features",
                detail=str(exc),
                query_scale=query_scale,
                render_scale=render_scale,
            )
        query_keypoints = query_sift.keypoints
        render_keypoints = render_sift.keypoints
        query_descriptors_all = query_sift.descriptors
        render_descriptors_all = render_sift.descriptors
        if (
            query_keypoints is None
            or render_keypoints is None
            or query_descriptors_all is None
            or render_descriptors_all is None
        ):
            return _empty_matches(
                self.identity(),
                reason="sift_feature_arrays_unavailable",
                query_scale=query_scale,
                render_scale=render_scale,
            )
        descriptor_matches = match_descriptors(
            query_descriptors_all,
            render_descriptors_all,
            metric="euclidean",
            cross_check=True,
            max_ratio=self.max_ratio,
        )
        if descriptor_matches.size == 0:
            return _empty_matches(
                self.identity(),
                reason="sift_no_ratio_matches",
                query_features=int(len(query_keypoints)),
                render_features=int(len(render_keypoints)),
                query_scale=query_scale,
                render_scale=render_scale,
            )
        query_indices = descriptor_matches[:, 0]
        render_indices = descriptor_matches[:, 1]
        query_descriptors = _unit_descriptors(query_descriptors_all[query_indices])
        render_descriptors = _unit_descriptors(render_descriptors_all[render_indices])
        distance = np.linalg.norm(query_descriptors - render_descriptors, axis=1)
        confidence = np.clip(1.0 - distance / 2.0, 0.0, 1.0)
        # SIFT keypoints use row/column order; the solver contract is x/y.
        query_xy = query_keypoints[query_indices][:, ::-1].astype(np.float64) / query_scale
        render_xy = render_keypoints[render_indices][:, ::-1].astype(np.float64) / render_scale
        order = np.lexsort((render_xy[:, 1], render_xy[:, 0], query_xy[:, 1], query_xy[:, 0], -confidence))
        order = order[: self.max_matches]
        return MatchSet(
            query_xy_px=query_xy[order],
            render_xy_px=render_xy[order],
            confidence=confidence[order],
            diagnostics={
                "query_features": int(len(query_keypoints)),
                "render_features": int(len(render_keypoints)),
                "ratio_matches": int(len(descriptor_matches)),
                "returned_matches": int(len(order)),
                "query_resize_scale": query_scale,
                "render_resize_scale": render_scale,
            },
            provenance=self.identity(),
        )

    def match_many(
        self,
        query_rgb: NDArray[np.uint8],
        render_images: list[NDArray[np.uint8]],
    ) -> list[MatchSet]:
        """Hermetic batch API; learned workers can avoid repeated model loads."""

        return [self.match(query_rgb, render) for render in render_images]


@dataclass(frozen=True)
class WorkerMatcher:
    """One-shot JSON/NPZ adapter for MINIMA, RoMa, or another pinned worker."""

    command: tuple[str, ...]
    matcher_id: str
    model_manifest: dict[str, Any]
    seed: int = 0
    timeout_s: float = 300.0
    keep_failed_jobs_dir: Path | None = None
    cache_dir: Path | None = None

    def __post_init__(self) -> None:
        if not self.command:
            raise ValueError("matcher worker command cannot be empty")
        if not self.matcher_id.strip():
            raise ValueError("matcher_id cannot be empty")

    def identity(self) -> dict[str, Any]:
        return {
            "id": self.matcher_id,
            "kind": "external_json_npz_worker_v1",
            "command": list(self.command),
            "seed": self.seed,
            "model_manifest": self.model_manifest,
            "worker_render_ids": {
                "policy": WORKER_RENDER_ID_POLICY,
                "policy_version": WORKER_RENDER_ID_POLICY_VERSION,
                "key_schema": CORRESPONDENCE_CACHE_KEY_SCHEMA,
                "content_addressed": True,
                "cache_independent": True,
                "duplicate_pair_policy": "execute_once_then_expand_ordered_occurrences",
            },
            "correspondence_cache": {
                "enabled": self.cache_dir is not None,
                "directory": str(self.cache_dir.resolve()) if self.cache_dir is not None else None,
                "entry_schema": CORRESPONDENCE_CACHE_ENTRY_SCHEMA,
                "key_schema": CORRESPONDENCE_CACHE_KEY_SCHEMA,
                "corrupt_entry_policy": "recompute_and_record_reason",
            },
        }

    def check_available(self) -> None:
        executable = self.command[0]
        if Path(executable).is_absolute():
            available = Path(executable).is_file()
        else:
            available = shutil.which(executable) is not None
        if not available:
            raise MatcherUnavailable(f"matcher worker executable is unavailable: {executable}")

    def match(self, query_rgb: NDArray[np.uint8], render_rgb: NDArray[np.uint8]) -> MatchSet:
        return self.match_many(query_rgb, [render_rgb])[0]

    def match_many(
        self,
        query_rgb: NDArray[np.uint8],
        render_images: list[NDArray[np.uint8]],
    ) -> list[MatchSet]:
        self.check_available()
        query = _rgb_image(query_rgb, "query")
        renders = [_rgb_image(render, f"render[{index}]") for index, render in enumerate(render_images)]
        if not renders:
            return []
        addressed = _content_addressed_worker_pairs(self, query, renders)
        if self.cache_dir is not None:
            return self._match_many_cached(query, renders, addressed)
        return self._match_many_uncached(query, renders, addressed)

    def _match_many_uncached(
        self,
        query: NDArray[np.uint8],
        renders: list[NDArray[np.uint8]],
        addressed: _ContentAddressedWorkerPairs,
    ) -> list[MatchSet]:
        unique_keys = list(addressed.key_records)
        worker_results = self._run_worker_batch(
            query,
            [addressed.render_by_key[key] for key in unique_keys],
            unique_keys,
        )
        matches_by_key = dict(zip(unique_keys, worker_results, strict=True))
        source_batches = _source_worker_batches(worker_results)
        if len(source_batches) != 1:
            raise MatcherExecutionError("one worker invocation unexpectedly produced multiple batch records")
        worker_batch = {
            **source_batches[0]["record"],
            "content_addressed_pairs": {
                "render_id_policy": WORKER_RENDER_ID_POLICY,
                "render_id_policy_version": WORKER_RENDER_ID_POLICY_VERSION,
                "requested_occurrences": len(renders),
                "unique_pairs_executed": len(unique_keys),
                "duplicate_occurrences": len(renders) - len(unique_keys),
                "ordered_render_ids": list(addressed.ordered_keys),
                "cache_enabled": False,
            },
        }
        return [
            self._ordered_occurrence_match(
                matches_by_key[key],
                occurrence_index=index,
                worker_batch=worker_batch,
            )
            for index, key in enumerate(addressed.ordered_keys)
        ]

    def _match_many_cached(
        self,
        query: NDArray[np.uint8],
        renders: list[NDArray[np.uint8]],
        addressed: _ContentAddressedWorkerPairs,
    ) -> list[MatchSet]:
        if self.cache_dir is None:
            raise RuntimeError("cached matcher path reached without a cache directory")
        cache_started = time.perf_counter()
        cache_root = self.cache_dir.expanduser().resolve()
        matches_by_key: dict[str, MatchSet] = {}
        lookup_reason_by_key: dict[str, str | None] = {}
        for key, key_record in addressed.key_records.items():
            cached, reason = _read_cache_entry(cache_root, key, key_record)
            lookup_reason_by_key[key] = reason
            if cached is not None:
                matches_by_key[key] = cached

        missing_keys = [key for key in addressed.key_records if key not in matches_by_key]
        if missing_keys:
            # Cache keys are stable JSON render ids. The worker derives its
            # per-pair RNG seed from this id, so partial-hit batches produce
            # the same inference as a cold cache-enabled batch.
            worker_results = self._run_worker_batch(
                query,
                [addressed.render_by_key[key] for key in missing_keys],
                missing_keys,
            )
            for key, match in zip(missing_keys, worker_results, strict=True):
                _write_cache_entry(
                    cache_root,
                    key,
                    addressed.key_records[key],
                    match,
                    population_reason=lookup_reason_by_key[key] or "not_found",
                )
                matches_by_key[key] = match

        source_batches = _source_worker_batches([matches_by_key[key] for key in addressed.ordered_keys])
        cache_hits = sum(lookup_reason_by_key[key] is None for key in addressed.ordered_keys)
        corrupt_recomputed = sum(
            reason is not None and reason != "not_found"
            for key, reason in ((key, lookup_reason_by_key[key]) for key in addressed.ordered_keys)
        )
        cache_batch = {
            "schema": CORRESPONDENCE_CACHE_BATCH_SCHEMA,
            "status": "ok",
            "matcher_id": self.matcher_id,
            "query": addressed.query_identity,
            "render_count": len(renders),
            "cache": {
                "enabled": True,
                "directory": str(cache_root),
                "entry_schema": CORRESPONDENCE_CACHE_ENTRY_SCHEMA,
                "key_schema": CORRESPONDENCE_CACHE_KEY_SCHEMA,
                "ordered_entry_keys": list(addressed.ordered_keys),
                "hits": cache_hits,
                "misses": len(renders) - cache_hits,
                "unique_worker_misses": len(missing_keys),
                "corrupt_entries_recomputed": corrupt_recomputed,
                "corrupt_entry_policy": "recompute_and_record_reason",
            },
            "source_worker_batches": source_batches,
            "runtime": {
                "cache_operation_wall_s": round(time.perf_counter() - cache_started, 6),
                "worker_invoked_this_call": bool(missing_keys),
                "worker_pairs_executed_this_call": len(missing_keys),
                "source_worker_runtime_semantics": (
                    "producer inference runtimes are retained inside source_worker_batches; "
                    "they are historical on cache hits"
                ),
            },
            "provenance": {
                "worker": {
                    "network_allowed": False,
                    "implicit_model_downloads_allowed": False,
                },
                "worker_cache_identity_sha256": _canonical_sha256(addressed.worker_identity),
                "worker_render_id_policy": WORKER_RENDER_ID_POLICY,
                "worker_render_id_policy_version": WORKER_RENDER_ID_POLICY_VERSION,
            },
        }
        results: list[MatchSet] = []
        for index, key in enumerate(addressed.ordered_keys):
            source = matches_by_key[key]
            reason = lookup_reason_by_key[key]
            source_diagnostics = dict(source.diagnostics)
            source_render_id = source_diagnostics.get("id")
            source_runtime_s = source_diagnostics.get("runtime_s")
            source_diagnostics.update(
                {
                    "id": index,
                    "source_worker_render_id": source_render_id,
                    "runtime_semantics": (
                        "runtime_s is the source producer inference runtime; consult cache.current_worker_runtime_s "
                        "for work performed in this call"
                    ),
                    "cache": {
                        "enabled": True,
                        "status": "hit" if reason is None else "miss",
                        "reason": reason,
                        "key": key,
                        "entry_schema": CORRESPONDENCE_CACHE_ENTRY_SCHEMA,
                        "worker_executed_this_call": reason is not None,
                        "source_worker_runtime_s": source_runtime_s,
                        "current_worker_runtime_s": source_runtime_s if reason is not None else 0.0,
                    },
                }
            )
            results.append(
                MatchSet(
                    query_xy_px=source.query_xy_px,
                    render_xy_px=source.render_xy_px,
                    confidence=source.confidence,
                    selected=source.selected,
                    diagnostics=source_diagnostics,
                    provenance={
                        **self.identity(),
                        "worker_batch": cache_batch,
                        "cache_entry": {
                            "key": key,
                            "status": "hit" if reason is None else "miss",
                            "reason": reason,
                        },
                    },
                )
            )
        return results

    def _ordered_occurrence_match(
        self,
        source: MatchSet,
        *,
        occurrence_index: int,
        worker_batch: dict[str, Any],
    ) -> MatchSet:
        source_diagnostics = dict(source.diagnostics)
        source_render_id = source_diagnostics.get("id")
        source_diagnostics.update(
            {
                "id": occurrence_index,
                "source_worker_render_id": source_render_id,
            }
        )
        return MatchSet(
            query_xy_px=source.query_xy_px,
            render_xy_px=source.render_xy_px,
            confidence=source.confidence,
            selected=source.selected,
            diagnostics=source_diagnostics,
            provenance={**self.identity(), "worker_batch": worker_batch},
        )

    def _run_worker_batch(
        self,
        query: NDArray[np.uint8],
        renders: list[NDArray[np.uint8]],
        render_ids: Sequence[int | str],
    ) -> list[MatchSet]:
        if len(render_ids) != len(renders):
            raise ValueError("worker render ids must match the render image count")
        with tempfile.TemporaryDirectory(prefix="peakle-matcher-") as temporary:
            job_dir = Path(temporary)
            query_path = job_dir / "query.png"
            request_path = job_dir / "request.json"
            result_path = job_dir / "result.json"
            Image.fromarray(query, mode="RGB").save(query_path)
            render_requests: list[dict[str, Any]] = []
            matches_paths: list[Path] = []
            for index, (render, render_id) in enumerate(zip(renders, render_ids, strict=True)):
                render_path = job_dir / f"render-{index:03d}.png"
                matches_path = job_dir / f"matches-{index:03d}.npz"
                Image.fromarray(render, mode="RGB").save(render_path)
                matches_paths.append(matches_path)
                render_requests.append(
                    {
                        "id": render_id,
                        "path": render_path.name,
                        "shape": list(render.shape),
                        "sha256": _array_sha256(render),
                        "matches_npz": matches_path.name,
                    }
                )
            request = {
                "schema": MATCH_REQUEST_SCHEMA,
                "matcher_id": self.matcher_id,
                "seed": self.seed,
                "query": {
                    "path": query_path.name,
                    "shape": list(query.shape),
                    "sha256": _array_sha256(query),
                },
                "renders": render_requests,
                "outputs": {"result_json": result_path.name},
                "model_manifest": self.model_manifest,
                "contract": {
                    "coordinates": MATCH_COORDINATE_CONTRACT,
                    "network_allowed": False,
                    "implicit_model_downloads_allowed": False,
                },
            }
            request_path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")
            try:
                worker_environment = os.environ.copy()
                worker_environment.update(
                    {
                        # The worker contract forbids implicit downloads.  These
                        # variables make the common Hugging Face/model-library
                        # paths fail closed even if the worker forgets to set
                        # their offline switches itself.
                        "HF_DATASETS_OFFLINE": "1",
                        "HF_HUB_OFFLINE": "1",
                        "PEAKLE_MATCHER_NETWORK_ALLOWED": "0",
                        "TRANSFORMERS_OFFLINE": "1",
                    }
                )
                completed = subprocess.run(
                    (*self.command, "--request", str(request_path)),
                    cwd=job_dir,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                    env=worker_environment,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                self._preserve_failed_job(job_dir, f"worker_start_or_timeout: {exc}")
                raise MatcherUnavailable(f"matcher worker did not complete: {exc}") from exc
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout).strip()[-1000:]
                self._preserve_failed_job(job_dir, detail)
                raise MatcherExecutionError(
                    f"matcher worker exited with status {completed.returncode}: {detail or 'no diagnostic'}"
                )
            if not result_path.is_file():
                self._preserve_failed_job(job_dir, "missing result.json")
                raise MatcherExecutionError("matcher worker did not produce its declared result JSON")
            try:
                result = json.loads(result_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                self._preserve_failed_job(job_dir, f"invalid result JSON: {exc}")
                raise MatcherExecutionError(f"invalid matcher result JSON: {exc}") from exc
            if not isinstance(result, dict) or result.get("schema") != BATCH_RESULT_SCHEMA:
                raise MatcherExecutionError(f"matcher worker result schema must be {BATCH_RESULT_SCHEMA!r}")
            status = result.get("status")
            if status in {"SKIPPED_MISSING_MODEL", "SKIPPED_UNAVAILABLE"}:
                raise MatcherUnavailable(str(result.get("message") or status))
            if status != "ok":
                raise MatcherExecutionError(f"matcher worker returned status {status!r}: {result.get('message', '')}")
            if result.get("matcher_id") != self.matcher_id:
                raise MatcherExecutionError(
                    f"matcher worker returned matcher_id {result.get('matcher_id')!r}, expected {self.matcher_id!r}"
                )
            if result.get("request_sha256") != _file_sha256(request_path):
                raise MatcherExecutionError("matcher worker result does not identify the exact request JSON")
            worker_provenance = result.get("provenance", {}).get("worker", {})
            if (
                worker_provenance.get("network_allowed") is not False
                or worker_provenance.get("implicit_model_downloads_allowed") is not False
            ):
                raise MatcherExecutionError("matcher worker did not attest to the offline/no-download contract")
            if any(not path.is_file() for path in matches_paths):
                self._preserve_failed_job(job_dir, "missing one or more matches NPZ files")
                raise MatcherExecutionError("successful matcher worker did not produce every declared matches NPZ")
            results: list[MatchSet] = []
            per_render = result.get("renders")
            if not isinstance(per_render, list) or len(per_render) != len(render_requests):
                raise MatcherExecutionError("matcher worker returned the wrong number of ordered render records")
            if any(
                not isinstance(record, dict) or record.get("id") != render_requests[index]["id"]
                for index, record in enumerate(per_render)
            ):
                raise MatcherExecutionError("matcher worker render records do not preserve request id order")
            worker_batch = _compact_worker_batch(result)
            for index, matches_path in enumerate(matches_paths):
                try:
                    with np.load(matches_path, allow_pickle=False) as payload:
                        query_xy = np.asarray(payload["query_xy_px"], dtype=np.float64)
                        render_xy = np.asarray(payload["render_xy_px"], dtype=np.float64)
                        confidence = np.asarray(payload["confidence"], dtype=np.float64)
                        selected = (
                            np.asarray(payload["selected"], dtype=np.bool_)
                            if "selected" in payload.files
                            else np.ones(query_xy.shape[0], dtype=np.bool_)
                        )
                except (OSError, KeyError, ValueError) as exc:
                    self._preserve_failed_job(job_dir, f"invalid matches NPZ {index}: {exc}")
                    raise MatcherExecutionError(f"invalid matcher matches NPZ for render {index}: {exc}") from exc
                render_diagnostics = per_render[index]
                results.append(
                    MatchSet(
                        query_xy_px=query_xy,
                        render_xy_px=render_xy,
                        confidence=confidence,
                        selected=selected,
                        diagnostics=render_diagnostics,
                        provenance={**self.identity(), "worker_batch": worker_batch},
                    )
                )
            return results

    def _preserve_failed_job(self, job_dir: Path, diagnostic: str) -> None:
        if self.keep_failed_jobs_dir is None:
            return
        self.keep_failed_jobs_dir.mkdir(parents=True, exist_ok=True)
        target = self.keep_failed_jobs_dir / f"failed-{hashlib.sha256(diagnostic.encode()).hexdigest()[:12]}"
        if not target.exists():
            shutil.copytree(job_dir, target)
            (target / "peakle_failure.txt").write_text(diagnostic + "\n")


def load_model_manifest(path: Path) -> dict[str, Any]:
    """Load a matcher manifest and verify every declared local file hash."""

    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("matcher model manifest must be a JSON object")
    artifacts = payload.get("artifacts", [])
    if not isinstance(artifacts, list):
        raise ValueError("matcher model manifest artifacts must be a list")
    normalized_artifacts: list[dict[str, Any]] = []
    for record in artifacts:
        if not isinstance(record, dict) or not record.get("path") or not record.get("sha256"):
            raise ValueError("each matcher artifact needs path and sha256")
        artifact_path = Path(str(record["path"])).expanduser()
        if not artifact_path.is_absolute():
            artifact_path = path.parent / artifact_path
        if not artifact_path.is_file():
            raise MatcherUnavailable(f"matcher artifact is missing: {artifact_path}")
        actual = _file_sha256(artifact_path)
        if actual != record["sha256"]:
            raise MatcherUnavailable(
                f"matcher artifact hash mismatch for {artifact_path}: expected {record['sha256']}, got {actual}"
            )
        normalized_artifacts.append({**record, "path": str(artifact_path.resolve())})
    return {**payload, "artifacts": normalized_artifacts, "manifest_path": str(path.resolve())}


def _rgb_image(value: NDArray[np.uint8], name: str) -> NDArray[np.uint8]:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{name} image must be HxWx3 RGB, got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(np.rint(image), 0.0, 255.0).astype(np.uint8)
    return image


def _matcher_gray(rgb: NDArray[np.uint8], max_dimension: int) -> tuple[NDArray[np.float64], float]:
    if max_dimension < 64:
        raise ValueError("matcher max dimension must be at least 64 pixels")
    scale = min(1.0, max_dimension / max(rgb.shape[:2]))
    gray = rgb2gray(rgb).astype(np.float64)
    if scale < 1.0:
        gray = resize(
            gray,
            (max(1, round(gray.shape[0] * scale)), max(1, round(gray.shape[1] * scale))),
            order=1,
            anti_aliasing=True,
            preserve_range=True,
        ).astype(np.float64)
    return gray, scale


def _unit_descriptors(descriptors: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(descriptors, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def _empty_matches(identity: dict[str, Any], *, reason: str, **diagnostics: Any) -> MatchSet:
    return MatchSet(
        query_xy_px=np.empty((0, 2), dtype=np.float64),
        render_xy_px=np.empty((0, 2), dtype=np.float64),
        confidence=np.empty(0, dtype=np.float64),
        diagnostics={"reason": reason, **diagnostics},
        provenance=identity,
    )


def _array_sha256(array: NDArray[np.uint8]) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def _compact_worker_batch(result: dict[str, Any]) -> dict[str, Any]:
    """Keep shared run provenance once without copying all per-render records."""

    renders = result.get("renders")
    outputs = result.get("outputs")
    compact = {key: value for key, value in result.items() if key not in {"renders", "outputs"}}
    compact["render_count"] = len(renders) if isinstance(renders, list) else None
    compact["output_count"] = len(outputs) if isinstance(outputs, list) else None
    if isinstance(outputs, list):
        canonical_outputs = json.dumps(outputs, separators=(",", ":"), sort_keys=True).encode()
        compact["output_records_sha256"] = hashlib.sha256(canonical_outputs).hexdigest()
    return compact


def _image_identity(image: NDArray[np.uint8]) -> dict[str, Any]:
    return {
        "shape": list(image.shape),
        "dtype": str(image.dtype),
        "raw_rgb_sha256": _array_sha256(image),
    }


@dataclass(frozen=True)
class _ContentAddressedWorkerPairs:
    query_identity: dict[str, Any]
    worker_identity: dict[str, Any]
    ordered_keys: tuple[str, ...]
    key_records: dict[str, dict[str, Any]]
    render_by_key: dict[str, NDArray[np.uint8]]


def _content_addressed_worker_pairs(
    matcher: WorkerMatcher,
    query: NDArray[np.uint8],
    renders: list[NDArray[np.uint8]],
) -> _ContentAddressedWorkerPairs:
    """Derive worker IDs independently of whether persistent caching is enabled."""

    worker_identity = _worker_cache_identity(matcher)
    query_identity = _image_identity(query)
    ordered_keys: list[str] = []
    key_records: dict[str, dict[str, Any]] = {}
    render_by_key: dict[str, NDArray[np.uint8]] = {}
    for render in renders:
        key_record = {
            "schema": CORRESPONDENCE_CACHE_KEY_SCHEMA,
            "query": query_identity,
            "render": _image_identity(render),
            "worker": worker_identity,
            "contract": {
                "coordinates": MATCH_COORDINATE_CONTRACT,
                "network_allowed": False,
                "implicit_model_downloads_allowed": False,
            },
        }
        key = _canonical_sha256(key_record)
        ordered_keys.append(key)
        if key not in key_records:
            key_records[key] = key_record
            render_by_key[key] = render
    return _ContentAddressedWorkerPairs(
        query_identity=query_identity,
        worker_identity=worker_identity,
        ordered_keys=tuple(ordered_keys),
        key_records=key_records,
        render_by_key=render_by_key,
    )


def _worker_cache_identity(matcher: WorkerMatcher) -> dict[str, Any]:
    normalized_manifest = matcher.model_manifest
    manifest_path_value = normalized_manifest.get("manifest_path")
    manifest_path = (
        Path(str(manifest_path_value)).expanduser()
        if isinstance(manifest_path_value, str) and manifest_path_value
        else None
    )
    source_manifest_sha256 = (
        _file_sha256(manifest_path) if manifest_path is not None and manifest_path.is_file() else None
    )
    artifacts = normalized_manifest.get("artifacts")
    artifact_identities = (
        [
            {
                "role": record.get("role"),
                "sha256": record.get("sha256"),
                "size_bytes": record.get("size_bytes"),
            }
            for record in artifacts
            if isinstance(record, dict)
        ]
        if isinstance(artifacts, list)
        else []
    )
    return {
        "matcher_id": matcher.matcher_id,
        "request_seed": matcher.seed,
        "determinism": {
            "worker_render_id_policy": WORKER_RENDER_ID_POLICY,
            "worker_seed_derivation": "sha256_canonical_json_of_request_seed_and_render_id",
            "network_allowed": False,
            "implicit_model_downloads_allowed": False,
        },
        "normalized_manifest": normalized_manifest,
        "normalized_manifest_sha256": _canonical_sha256(normalized_manifest),
        "source_manifest_sha256": source_manifest_sha256,
        "artifact_identities": artifact_identities,
        "inference_config": normalized_manifest.get("inference"),
        "worker_command": _command_identity(matcher.command),
    }


def _command_identity(command: tuple[str, ...]) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for index, token in enumerate(command):
        candidate: Path | None
        if index == 0:
            resolved_executable = token if Path(token).is_absolute() else shutil.which(token)
            candidate = Path(resolved_executable) if resolved_executable is not None else None
        else:
            path = Path(token).expanduser()
            candidate = path if path.is_absolute() else Path.cwd() / path
        if candidate is None or not candidate.is_file():
            continue
        resolved = candidate.resolve()
        files.append(
            {
                "argument_index": index,
                "argument": token,
                "resolved_path": str(resolved),
                "sha256": _file_sha256(resolved),
                "size_bytes": resolved.stat().st_size,
            }
        )
    if not files or files[0]["argument_index"] != 0:
        raise MatcherUnavailable("cannot content-address the matcher worker executable")
    return {
        "argv": list(command),
        "argv_sha256": _canonical_sha256(list(command)),
        "files": files,
    }


def _cache_entry_directory(cache_root: Path, key: str) -> Path:
    return cache_root / CORRESPONDENCE_CACHE_ENTRY_SCHEMA / key[:2]


def _cache_metadata_path(cache_root: Path, key: str) -> Path:
    return _cache_entry_directory(cache_root, key) / f"{key}.json"


def _read_cache_entry(
    cache_root: Path,
    key: str,
    expected_key_record: dict[str, Any],
) -> tuple[MatchSet | None, str | None]:
    metadata_path = _cache_metadata_path(cache_root, key)
    if not metadata_path.is_file():
        entry_directory = metadata_path.parent
        if entry_directory.is_dir() and any(entry_directory.glob(f"{key}.*.npz")):
            return None, "orphan_npz_without_metadata"
        return None, "not_found"
    try:
        metadata = json.loads(metadata_path.read_text())
        if not isinstance(metadata, dict):
            raise ValueError("metadata is not a JSON object")
        if metadata.get("schema") != CORRESPONDENCE_CACHE_ENTRY_SCHEMA:
            raise ValueError("entry schema mismatch")
        if metadata.get("key") != key or _canonical_sha256(expected_key_record) != key:
            raise ValueError("entry key mismatch")
        if metadata.get("key_record") != expected_key_record:
            raise ValueError("entry key material mismatch")
        npz_name = metadata.get("npz_file")
        if not isinstance(npz_name, str) or Path(npz_name).name != npz_name:
            raise ValueError("entry NPZ filename is invalid")
        npz_path = metadata_path.parent / npz_name
        if not npz_path.is_file():
            raise ValueError("entry NPZ is missing")
        expected_npz_sha256 = metadata.get("npz_sha256")
        if not isinstance(expected_npz_sha256, str) or _file_sha256(npz_path) != expected_npz_sha256:
            raise ValueError("entry NPZ hash mismatch")
        with np.load(npz_path, allow_pickle=False) as payload:
            if set(payload.files) != {"query_xy_px", "render_xy_px", "confidence", "selected"}:
                raise ValueError("entry NPZ array names mismatch")
            query_xy = np.asarray(payload["query_xy_px"])
            render_xy = np.asarray(payload["render_xy_px"])
            confidence = np.asarray(payload["confidence"])
            selected = np.asarray(payload["selected"])
        arrays = _array_schema(query_xy, render_xy, confidence, selected)
        if metadata.get("arrays") != arrays:
            raise ValueError("entry array schema mismatch")
        if query_xy.dtype != np.float64 or render_xy.dtype != np.float64 or confidence.dtype != np.float64:
            raise ValueError("entry coordinate/confidence arrays must be float64")
        if selected.dtype != np.bool_:
            raise ValueError("entry selected array must be bool")
        render_diagnostics = metadata.get("render_diagnostics")
        worker_batch = metadata.get("worker_batch")
        if not isinstance(render_diagnostics, dict) or not isinstance(worker_batch, dict):
            raise ValueError("entry producer diagnostics are invalid")
        _validate_cache_producer(worker_batch, expected_key_record)
        match = MatchSet(
            query_xy_px=query_xy,
            render_xy_px=render_xy,
            confidence=confidence,
            selected=selected,
            diagnostics=render_diagnostics,
            provenance={"worker_batch": worker_batch},
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return None, f"corrupt_recomputed:{type(exc).__name__}:{str(exc)[:160]}"
    return match, None


def _write_cache_entry(
    cache_root: Path,
    key: str,
    key_record: dict[str, Any],
    match: MatchSet,
    *,
    population_reason: str,
) -> None:
    worker_batch = match.provenance.get("worker_batch")
    if not isinstance(worker_batch, dict):
        raise MatcherExecutionError("successful matcher result lacks auditable worker batch provenance")
    try:
        _validate_cache_producer(worker_batch, key_record)
    except ValueError as exc:
        raise MatcherExecutionError(f"matcher result is unsafe to cache: {exc}") from exc
    entry_directory = _cache_entry_directory(cache_root, key)
    entry_directory.mkdir(parents=True, exist_ok=True)
    temporary_npz: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=entry_directory,
            prefix=f".{key}.",
            suffix=".npz.tmp",
            delete=False,
        ) as handle:
            temporary_npz = Path(handle.name)
            np.savez_compressed(
                handle,
                query_xy_px=np.asarray(match.query_xy_px, dtype=np.float64),
                render_xy_px=np.asarray(match.render_xy_px, dtype=np.float64),
                confidence=np.asarray(match.confidence, dtype=np.float64),
                selected=np.asarray(match.selected, dtype=np.bool_),
            )
            handle.flush()
            os.fsync(handle.fileno())
        npz_sha256 = _file_sha256(temporary_npz)
        npz_path = entry_directory / f"{key}.{npz_sha256}.npz"
        os.replace(temporary_npz, npz_path)
        temporary_npz = None
        metadata = {
            "schema": CORRESPONDENCE_CACHE_ENTRY_SCHEMA,
            "key": key,
            "key_record": key_record,
            "npz_file": npz_path.name,
            "npz_sha256": npz_sha256,
            "npz_size_bytes": npz_path.stat().st_size,
            "arrays": _array_schema(
                match.query_xy_px,
                match.render_xy_px,
                match.confidence,
                np.asarray(match.selected, dtype=np.bool_),
            ),
            "render_diagnostics": match.diagnostics,
            "worker_batch": worker_batch,
            "worker_batch_sha256": _canonical_sha256(worker_batch),
            "population_reason": population_reason,
            "write_policy": "atomic_npz_then_atomic_json_commit_marker",
        }
        _atomic_write_json(_cache_metadata_path(cache_root, key), metadata)
    except OSError as exc:
        raise MatcherExecutionError(f"cannot atomically write matcher cache entry {key[:12]}: {exc}") from exc
    finally:
        if temporary_npz is not None:
            temporary_npz.unlink(missing_ok=True)


def _validate_cache_producer(worker_batch: dict[str, Any], key_record: dict[str, Any]) -> None:
    worker_identity = key_record.get("worker")
    if not isinstance(worker_identity, dict):
        raise ValueError("cache key has no worker identity")
    if worker_batch.get("schema") != BATCH_RESULT_SCHEMA or worker_batch.get("status") != "ok":
        raise ValueError("producer batch is not a successful worker result")
    if worker_batch.get("matcher_id") != worker_identity.get("matcher_id"):
        raise ValueError("producer matcher id differs from cache key")
    provenance = worker_batch.get("provenance")
    worker_provenance = provenance.get("worker") if isinstance(provenance, dict) else None
    if not isinstance(worker_provenance, dict) or (
        worker_provenance.get("network_allowed") is not False
        or worker_provenance.get("implicit_model_downloads_allowed") is not False
    ):
        raise ValueError("producer did not attest to the offline/no-download contract")
    command = worker_identity.get("worker_command")
    command_files = command.get("files") if isinstance(command, dict) else None
    command_hashes = (
        {record.get("sha256") for record in command_files if isinstance(record, dict)}
        if isinstance(command_files, list)
        else set()
    )
    implementation_sha256 = worker_provenance.get("implementation_sha256")
    if implementation_sha256 not in command_hashes:
        raise ValueError("producer worker implementation hash differs from the command files")
    expected_manifest_sha256 = worker_identity.get("source_manifest_sha256")
    if expected_manifest_sha256 is not None:
        producer_manifest = provenance.get("manifest") if isinstance(provenance, dict) else None
        if not isinstance(producer_manifest, dict) or producer_manifest.get("sha256") != expected_manifest_sha256:
            raise ValueError("producer source manifest hash differs from the cache key")
    expected_artifacts = {
        (record.get("role"), record.get("sha256"))
        for record in worker_identity.get("artifact_identities", [])
        if isinstance(record, dict)
    }
    if expected_artifacts:
        producer_artifacts = provenance.get("artifacts") if isinstance(provenance, dict) else None
        actual_artifacts = (
            {(record.get("role"), record.get("sha256")) for record in producer_artifacts if isinstance(record, dict)}
            if isinstance(producer_artifacts, list)
            else set()
        )
        if actual_artifacts != expected_artifacts:
            raise ValueError("producer artifact identities differ from the cache key")


def _array_schema(
    query_xy: NDArray[Any],
    render_xy: NDArray[Any],
    confidence: NDArray[Any],
    selected: NDArray[Any],
) -> dict[str, Any]:
    return {
        "query_xy_px": {"shape": list(np.asarray(query_xy).shape), "dtype": str(np.asarray(query_xy).dtype)},
        "render_xy_px": {"shape": list(np.asarray(render_xy).shape), "dtype": str(np.asarray(render_xy).dtype)},
        "confidence": {"shape": list(np.asarray(confidence).shape), "dtype": str(np.asarray(confidence).dtype)},
        "selected": {"shape": list(np.asarray(selected).shape), "dtype": str(np.asarray(selected).dtype)},
    }


def _source_worker_batches(matches: list[MatchSet]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for match in matches:
        worker_batch = match.provenance.get("worker_batch")
        if not isinstance(worker_batch, dict):
            raise MatcherExecutionError("matcher cache entry lost its source worker provenance")
        digest = _canonical_sha256(worker_batch)
        if digest in seen:
            continue
        seen.add(digest)
        records.append({"sha256": digest, "record": worker_batch})
    return records


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None

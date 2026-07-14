"""Frozen round-zero photo-beam PnP benchmark and post-freeze GT audit."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from peakle.domain.camera import CameraExtrinsics
from peakle.domain.coordinates import LocalPoint
from peakle.localize.photo_beam_render_pnp import PhotoBeamRenderSeedBridge
from peakle.localize.render_match_pnp import (
    RENDER_SEED_BATCH_SCHEMA,
    RenderMatchBatchResult,
    RenderSeed,
)

PHOTO_BEAM_PNP_ARCHIVE_SCHEMA = "peakle_photo_beam_pnp_benchmark_archive_v1"
PHOTO_BEAM_PNP_EVALUATION_SCHEMA = "peakle_photo_beam_pnp_benchmark_evaluation_v1"
POSITION_TARGET_M = 100.0
YAW_TARGET_DEG = 5.0


@dataclass(frozen=True, slots=True)
class PhotoBeamPnpBenchmarkArchive:
    """Immutable canonical JSON payload frozen before any truth is accepted."""

    basis_json: str
    archive_sha256: str

    def to_record(self) -> dict[str, Any]:
        basis = json.loads(self.basis_json)
        if _canonical_sha256(basis) != self.archive_sha256:
            raise RuntimeError("photo-beam PnP benchmark archive changed after freezing")
        return {**basis, "archive_sha256": self.archive_sha256}


def freeze_photo_beam_pnp_benchmark(
    bridge: PhotoBeamRenderSeedBridge,
    batch: RenderMatchBatchResult,
    *,
    lane_label: str,
    lane_config: Mapping[str, Any],
) -> PhotoBeamPnpBenchmarkArchive:
    """Freeze one complete estimator batch without accepting reference truth."""

    label = _lane_label(lane_label)
    bridge_record = bridge.to_record()
    bridge_ids = bridge.ordered_candidate_ids
    batch_ids = batch.ordered_candidate_ids
    if not bridge_ids or bridge_ids != batch_ids:
        raise ValueError("bridge and render/PnP batch candidate order or count differs")
    if len(set(batch_ids)) != len(batch_ids):
        raise ValueError("render/PnP batch candidate IDs must be unique")

    lane_config_record = _json_mapping(lane_config, "lane config")
    provenance = _json_mapping(batch.provenance, "batch provenance")
    rng_roots = _validate_batch_provenance(provenance, batch_ids)
    bridge_seeds = bridge.render_seeds
    results: list[dict[str, Any]] = []
    for beam_rank, (expected, item) in enumerate(zip(bridge_seeds, batch.results, strict=True), start=1):
        if item.candidate_id != expected.candidate_id:
            raise ValueError("render/PnP result candidate ID differs from its bridge seed")
        if _render_seed_record(item.render_seed) != _render_seed_record(expected.render_seed):
            raise ValueError(f"render/PnP input seed differs from bridge for {item.candidate_id!r}")
        if isinstance(item.rng_seed, bool) or not isinstance(item.rng_seed, int) or not 0 <= item.rng_seed < 2**32:
            raise ValueError("render/PnP candidate RNG seed must be uint32")
        if rng_roots[item.candidate_id] != item.rng_seed:
            raise ValueError("batch provenance and result RNG seeds differ")
        status = item.result.status
        extrinsics = item.result.extrinsics
        if status not in {"solved", "abstained"}:
            raise ValueError("render/PnP result status is unsupported")
        if (status == "solved") != (extrinsics is not None):
            raise ValueError("render/PnP status and accepted extrinsics are inconsistent")
        diagnostics = _json_mapping(item.result.diagnostics, "result diagnostics")
        _validate_result_diagnostics(diagnostics, item.candidate_id, item.rng_seed)
        results.append(
            {
                "candidate_id": item.candidate_id,
                "beam_rank": beam_rank,
                "input_seed": _render_seed_record(item.render_seed),
                "rng_seed": item.rng_seed,
                "status": status,
                "accepted_extrinsics": extrinsics.model_dump(mode="json") if extrinsics is not None else None,
                "diagnostics": diagnostics,
            }
        )

    source = {
        "bridge_sha256": _sha256(bridge_record.get("bridge_sha256"), "source bridge"),
        "photo_verifier_archive_sha256": _sha256(bridge.source_verifier_archive_sha256, "source verifier archive"),
        "skyline_atlas_archive_sha256": _sha256(bridge.source_atlas_sha256, "source atlas"),
    }
    basis = {
        "schema": PHOTO_BEAM_PNP_ARCHIVE_SCHEMA,
        "uses_reference_truth": False,
        "source": source,
        "lane": {"label": label, "config": lane_config_record},
        "ordered_candidate_ids": list(bridge_ids),
        "candidate_count": len(bridge_ids),
        "batch_provenance": provenance,
        "results": results,
    }
    return PhotoBeamPnpBenchmarkArchive(
        basis_json=_canonical_json(basis),
        archive_sha256=_canonical_sha256(basis),
    )


def validate_frozen_photo_beam_pnp_benchmark(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate, hash-check, and detach a serialized pre-truth archive."""

    archive = _json_mapping(record, "photo-beam PnP benchmark archive")
    _require_keys(
        archive,
        {
            "schema",
            "uses_reference_truth",
            "source",
            "lane",
            "ordered_candidate_ids",
            "candidate_count",
            "batch_provenance",
            "results",
            "archive_sha256",
        },
        "benchmark archive",
    )
    if archive.get("schema") != PHOTO_BEAM_PNP_ARCHIVE_SCHEMA:
        raise ValueError(f"expected {PHOTO_BEAM_PNP_ARCHIVE_SCHEMA} benchmark archive")
    if archive.get("uses_reference_truth") is not False:
        raise ValueError("benchmark archive crossed the reference-truth boundary")
    expected_sha = _sha256(archive.get("archive_sha256"), "benchmark archive")
    basis = dict(archive)
    basis.pop("archive_sha256")
    if _canonical_sha256(basis) != expected_sha:
        raise ValueError("benchmark archive SHA-256 does not match its contents")

    source = _mapping(archive.get("source"), "benchmark source")
    _require_keys(
        source, {"bridge_sha256", "photo_verifier_archive_sha256", "skyline_atlas_archive_sha256"}, "benchmark source"
    )
    for key in source:
        _sha256(source[key], key)
    lane = _mapping(archive.get("lane"), "benchmark lane")
    _require_keys(lane, {"label", "config"}, "benchmark lane")
    _lane_label(lane.get("label"))
    _json_mapping(lane.get("config"), "lane config")

    ids = archive.get("ordered_candidate_ids")
    if (
        not isinstance(ids, list)
        or not ids
        or any(not isinstance(candidate_id, str) or not candidate_id for candidate_id in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("benchmark ordered candidate IDs are invalid")
    if archive.get("candidate_count") != len(ids):
        raise ValueError("benchmark candidate count differs from ordered IDs")
    provenance = _mapping(archive.get("batch_provenance"), "batch provenance")
    rng_roots = _validate_batch_provenance(provenance, tuple(ids))

    results = archive.get("results")
    if not isinstance(results, list) or len(results) != len(ids):
        raise ValueError("benchmark result count differs from ordered IDs")
    for beam_rank, (candidate_id, value) in enumerate(zip(ids, results, strict=True), start=1):
        if not isinstance(candidate_id, str):  # narrowed above; retained for static auditability
            raise ValueError("benchmark candidate ID must be a string")
        result = _mapping(value, "benchmark seed result")
        _require_keys(
            result,
            {"candidate_id", "beam_rank", "input_seed", "rng_seed", "status", "accepted_extrinsics", "diagnostics"},
            "benchmark seed result",
        )
        if result.get("candidate_id") != candidate_id or result.get("beam_rank") != beam_rank:
            raise ValueError("benchmark result ID/rank linkage is inconsistent")
        _validated_seed_record(result.get("input_seed"))
        rng_seed = result.get("rng_seed")
        if isinstance(rng_seed, bool) or not isinstance(rng_seed, int) or not 0 <= rng_seed < 2**32:
            raise ValueError("benchmark candidate RNG seed must be uint32")
        if rng_roots[candidate_id] != rng_seed:
            raise ValueError("benchmark provenance and result RNG seeds differ")
        status = result.get("status")
        accepted = result.get("accepted_extrinsics")
        if status not in {"solved", "abstained"} or (status == "solved") != (accepted is not None):
            raise ValueError("benchmark status and accepted output are inconsistent")
        if accepted is not None:
            _validated_extrinsics(accepted, "accepted output")
        diagnostics = _mapping(result.get("diagnostics"), "result diagnostics")
        _validate_result_diagnostics(diagnostics, candidate_id, rng_seed)
    return archive


def evaluate_photo_beam_pnp_benchmark(
    archive_record: Mapping[str, Any],
    truth: CameraExtrinsics,
) -> dict[str, Any]:
    """Evaluate fixed stages after freezing; never select an estimator output."""

    archive = validate_frozen_photo_beam_pnp_benchmark(archive_record)
    per_seed: list[dict[str, Any]] = []
    stage_rows: dict[str, list[dict[str, Any]]] = {
        "input_seed": [],
        "pre_validation_pnp": [],
        "accepted_output": [],
    }
    for result in archive["results"]:
        candidate_id = str(result["candidate_id"])
        beam_rank = int(result["beam_rank"])
        input_pose = _seed_extrinsics(result["input_seed"])
        pre_pose, pre_source = _pre_validation_candidate(result["diagnostics"])
        accepted_pose = (
            _validated_extrinsics(result["accepted_extrinsics"], "accepted output")
            if result["accepted_extrinsics"] is not None
            else None
        )
        input_eval = _evaluated_pose(candidate_id, beam_rank, input_pose, truth, source="input_seed")
        pre_eval = (
            _evaluated_pose(candidate_id, beam_rank, pre_pose, truth, source=pre_source)
            if pre_pose is not None and pre_source is not None
            else None
        )
        accepted_eval = (
            _evaluated_pose(candidate_id, beam_rank, accepted_pose, truth, source="accepted_output")
            if accepted_pose is not None
            else None
        )
        stage_rows["input_seed"].append(input_eval)
        if pre_eval is not None:
            stage_rows["pre_validation_pnp"].append(pre_eval)
        if accepted_eval is not None:
            stage_rows["accepted_output"].append(accepted_eval)
        diagnostics = result["diagnostics"]
        per_seed.append(
            {
                "candidate_id": candidate_id,
                "beam_rank": beam_rank,
                "input_seed": input_eval,
                "pre_validation_pnp": pre_eval,
                "accepted_output": accepted_eval,
                "match_counts": _match_counts(diagnostics),
                "candidate_validation": _validation_outcome(diagnostics),
            }
        )

    input_summary = _stage_summary(stage_rows["input_seed"], len(per_seed))
    pre_summary = _stage_summary(stage_rows["pre_validation_pnp"], len(per_seed))
    accepted_summary = _stage_summary(stage_rows["accepted_output"], len(per_seed))
    return {
        "schema": PHOTO_BEAM_PNP_EVALUATION_SCHEMA,
        "archive_sha256": archive["archive_sha256"],
        "reference_data_used": True,
        "used_by_estimator": False,
        "estimator_selection_performed": False,
        "target": {
            "horizontal_position_m_lte": POSITION_TARGET_M,
            "absolute_yaw_deg_lte": YAW_TARGET_DEG,
            "pitch_used": False,
            "roll_used": False,
        },
        "input_seed": input_summary,
        "pre_validation_pnp": pre_summary,
        "accepted_output": accepted_summary,
        "per_seed": per_seed,
        "gt_oracles": {
            "evaluation_only": True,
            "used_by_estimator": False,
            "changes_estimator_selection": False,
            "input_seed": _stage_oracle(stage_rows["input_seed"]),
            "pre_validation_pnp": _stage_oracle(stage_rows["pre_validation_pnp"]),
            "accepted_output": _stage_oracle(stage_rows["accepted_output"]),
        },
    }


def _validate_batch_provenance(provenance: Mapping[str, Any], ids: tuple[str, ...]) -> dict[str, Any]:
    if provenance.get("schema") != RENDER_SEED_BATCH_SCHEMA or provenance.get("uses_reference_truth") is not False:
        raise ValueError("batch provenance is not a truth-free round-zero seed batch")
    if provenance.get("ordered_candidate_ids") != list(ids) or provenance.get("candidate_count") != len(ids):
        raise ValueError("batch provenance candidate ID linkage is inconsistent")
    rng = _mapping(provenance.get("rng"), "batch RNG provenance")
    roots = _mapping(rng.get("candidate_root_seeds"), "candidate RNG roots")
    if set(roots) != set(ids):
        raise ValueError("batch RNG candidate ID linkage is inconsistent")
    return roots


def _validate_result_diagnostics(diagnostics: Mapping[str, Any], candidate_id: str, rng_seed: int) -> None:
    batch = _mapping(diagnostics.get("batch"), "result batch diagnostics")
    if batch.get("candidate_id") != candidate_id or batch.get("candidate_rng_seed") != rng_seed:
        raise ValueError("result diagnostics candidate ID/RNG linkage is inconsistent")
    validation = diagnostics.get("candidate_validation")
    if isinstance(validation, Mapping) and validation.get("uses_reference_truth") is not False:
        raise ValueError("candidate validation diagnostics crossed the truth boundary")


def _render_seed_record(seed: RenderSeed) -> dict[str, Any]:
    seed.validate()
    return {
        "position": seed.position.model_dump(mode="json"),
        "yaw_deg": seed.yaw_deg,
        "query_pnp_initial_pitch_deg": seed.pitch_deg,
    }


def _validated_seed_record(value: Any) -> dict[str, Any]:
    seed = _mapping(value, "input render seed")
    _require_keys(seed, {"position", "yaw_deg", "query_pnp_initial_pitch_deg"}, "input render seed")
    position = _mapping(seed.get("position"), "input seed position")
    _require_keys(position, {"east_m", "north_m", "up_m"}, "input seed position")
    RenderSeed(
        position=LocalPoint(
            east_m=_finite_float(position.get("east_m"), "seed east"),
            north_m=_finite_float(position.get("north_m"), "seed north"),
            up_m=_finite_float(position.get("up_m"), "seed up"),
        ),
        yaw_deg=_finite_float(seed.get("yaw_deg"), "seed yaw"),
        pitch_deg=_finite_float(seed.get("query_pnp_initial_pitch_deg"), "seed pitch"),
    ).validate()
    return seed


def _seed_extrinsics(value: Any) -> CameraExtrinsics:
    seed = _validated_seed_record(value)
    return CameraExtrinsics(
        position=LocalPoint(**seed["position"]),
        yaw_deg=seed["yaw_deg"],
        pitch_deg=seed["query_pnp_initial_pitch_deg"],
        roll_deg=0.0,
    )


def _pre_validation_candidate(diagnostics: Mapping[str, Any]) -> tuple[CameraExtrinsics | None, str | None]:
    final_pnp = diagnostics.get("final_pnp")
    if isinstance(final_pnp, Mapping) and final_pnp.get("candidate_pose") is not None:
        return _validated_extrinsics(final_pnp["candidate_pose"], "pre-validation PnP candidate"), "final_pnp"
    if diagnostics.get("rejected_candidate_pose") is not None:
        return _validated_extrinsics(diagnostics["rejected_candidate_pose"], "rejected candidate"), "rejected"
    frames = diagnostics.get("frames")
    if isinstance(frames, list):
        for frame in frames:
            pnp = frame.get("pnp") if isinstance(frame, Mapping) else None
            if isinstance(pnp, Mapping) and pnp.get("candidate_pose") is not None:
                return _validated_extrinsics(pnp["candidate_pose"], "frame PnP candidate"), "frame_pnp"
    return None, None


def _evaluated_pose(
    candidate_id: str,
    beam_rank: int,
    pose: CameraExtrinsics,
    truth: CameraExtrinsics,
    *,
    source: str,
) -> dict[str, Any]:
    horizontal = math.hypot(
        pose.position.east_m - truth.position.east_m,
        pose.position.north_m - truth.position.north_m,
    )
    vertical = abs(pose.position.up_m - truth.position.up_m)
    yaw = abs((pose.yaw_deg - truth.yaw_deg + 180.0) % 360.0 - 180.0)
    return {
        "candidate_id": candidate_id,
        "beam_rank": beam_rank,
        "source": source,
        "errors": {
            "horizontal_position_m": _rounded(horizontal),
            "vertical_m": _rounded(vertical),
            "position_3d_m": _rounded(math.hypot(horizontal, vertical)),
            "yaw_deg": _rounded(yaw),
            "pitch_deg": None,
            "roll_deg": None,
        },
        "reaches_target": horizontal <= POSITION_TARGET_M and yaw <= YAW_TARGET_DEG,
    }


def _stage_summary(rows: list[dict[str, Any]], beam_count: int) -> dict[str, Any]:
    successful = [row for row in rows if row["reaches_target"]]
    return {
        "beam_count": beam_count,
        "candidate_count_available": len(rows),
        "target_candidate_count": len(successful),
        "target_recall": bool(successful),
        "target_fraction_of_beam": _rounded(len(successful) / beam_count),
        "first_target_beam_rank": min((row["beam_rank"] for row in successful), default=None),
    }


def _stage_oracle(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return min(rows, key=_oracle_key) if rows else None


def _oracle_key(row: Mapping[str, Any]) -> tuple[float, float, float, int, str]:
    errors = row["errors"]
    return (
        errors["horizontal_position_m"] / POSITION_TARGET_M + errors["yaw_deg"] / YAW_TARGET_DEG,
        errors["horizontal_position_m"],
        errors["yaw_deg"],
        row["beam_rank"],
        row["candidate_id"],
    )


def _match_counts(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    frames = diagnostics.get("frames")
    frame = frames[0] if isinstance(frames, list) and frames and isinstance(frames[0], Mapping) else {}
    counts = frame.get("match_stage_counts") if isinstance(frame, Mapping) else None
    return _json_mapping(counts, "match stage counts") if isinstance(counts, Mapping) else {}


def _validation_outcome(diagnostics: Mapping[str, Any]) -> dict[str, Any] | None:
    validation = diagnostics.get("candidate_validation")
    if not isinstance(validation, Mapping):
        return None
    return {
        "enabled": validation.get("enabled"),
        "passed": validation.get("passed"),
        "status": validation.get("status"),
        "failures": list(validation.get("failures") or []),
        "uses_reference_truth": validation.get("uses_reference_truth"),
    }


def _validated_extrinsics(value: Any, name: str) -> CameraExtrinsics:
    record = _mapping(value, name)
    _require_keys(record, {"position", "yaw_deg", "pitch_deg", "roll_deg"}, name)
    position = _mapping(record.get("position"), f"{name} position")
    _require_keys(position, {"east_m", "north_m", "up_m"}, f"{name} position")
    try:
        return CameraExtrinsics.model_validate(record)
    except Exception as exc:
        raise ValueError(f"{name} camera extrinsics are invalid") from exc


def _lane_label(value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or len(value) > 128 or "\0" in value:
        raise ValueError("benchmark lane label must be a trimmed non-empty string")
    return value


def _json_mapping(value: Any, name: str) -> dict[str, Any]:
    try:
        copied = json.loads(_canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only finite canonical JSON values") from exc
    if not isinstance(copied, dict):
        raise ValueError(f"{name} must be a JSON object")
    return copied


def _canonical_json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _require_keys(record: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(record) != expected:
        raise ValueError(f"{name} fields are unsupported")


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} SHA-256 is missing or malformed")
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _rounded(value: float) -> float:
    return round(float(value), 6)

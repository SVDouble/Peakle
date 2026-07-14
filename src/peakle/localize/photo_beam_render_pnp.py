"""Truth-blind bridge from a frozen photo beam to round-zero render seeds."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from peakle.domain.coordinates import LocalPoint
from peakle.localize.photo_geometry_verifier import validate_frozen_photo_geometry_verifier
from peakle.localize.render_match_pnp import IdentifiedRenderSeed, RenderSeed

PHOTO_BEAM_RENDER_SEED_BRIDGE_SCHEMA = "peakle_photo_beam_render_seed_bridge_v1"
PHOTO_BEAM_RENDER_SEED_BRIDGE_METHOD = "validated_complete_ordered_beam_identity_mapping_v1"


@dataclass(frozen=True, slots=True)
class PhotoBeamRenderSeed:
    """Immutable scalar form of one atlas hypothesis admitted to round zero."""

    candidate_id: str
    beam_rank: int
    verifier_rank: int
    original_estimator_rank: int
    east_m: float
    north_m: float
    up_m: float
    yaw_deg: float
    query_initial_pitch_deg: float
    discarded_atlas_roll_nuisance_deg: float

    def identified_render_seed(self) -> IdentifiedRenderSeed:
        """Create a detached seed accepted by the batched render/PnP API."""

        hypothesis = IdentifiedRenderSeed(
            candidate_id=self.candidate_id,
            render_seed=RenderSeed(
                position=LocalPoint(east_m=self.east_m, north_m=self.north_m, up_m=self.up_m),
                yaw_deg=self.yaw_deg,
                pitch_deg=self.query_initial_pitch_deg,
            ),
        )
        hypothesis.validate()
        return hypothesis

    def to_record(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "beam_rank": self.beam_rank,
            "verifier_rank": self.verifier_rank,
            "original_estimator_rank": self.original_estimator_rank,
            "render_seed": {
                "position": {
                    "east_m": self.east_m,
                    "north_m": self.north_m,
                    "up_m": self.up_m,
                },
                "exact_heading_yaw_deg": self.yaw_deg,
                "query_pnp_initial_pitch_deg": self.query_initial_pitch_deg,
                "physical_render_roll_deg_in_round_zero_api": 0.0,
            },
            "discarded_atlas_roll_nuisance_deg": self.discarded_atlas_roll_nuisance_deg,
        }


@dataclass(frozen=True, slots=True)
class PhotoBeamRenderSeedBridge:
    """Frozen, hash-linked identity mapping of the complete verifier beam."""

    source_verifier_archive_sha256: str
    source_atlas_sha256: str
    photo_rgb_sha256: str
    observed_skyline_sha256: str
    edge_model_sha256: str
    depth_model_sha256: str
    observation_provenance_json: str
    seeds: tuple[PhotoBeamRenderSeed, ...]
    bridge_sha256: str

    @property
    def ordered_candidate_ids(self) -> tuple[str, ...]:
        return tuple(seed.candidate_id for seed in self.seeds)

    @property
    def render_seeds(self) -> tuple[IdentifiedRenderSeed, ...]:
        """Return fresh detached hypotheses in the frozen source-beam order."""

        return tuple(seed.identified_render_seed() for seed in self.seeds)

    def to_record(self) -> dict[str, Any]:
        basis = _bridge_basis(
            source_verifier_archive_sha256=self.source_verifier_archive_sha256,
            source_atlas_sha256=self.source_atlas_sha256,
            photo_rgb_sha256=self.photo_rgb_sha256,
            observed_skyline_sha256=self.observed_skyline_sha256,
            edge_model_sha256=self.edge_model_sha256,
            depth_model_sha256=self.depth_model_sha256,
            observation_provenance_json=self.observation_provenance_json,
            seeds=self.seeds,
        )
        if _canonical_sha256(basis) != self.bridge_sha256:
            raise RuntimeError("photo-beam render-seed bridge changed after freezing")
        return {**basis, "bridge_sha256": self.bridge_sha256}


def build_photo_beam_render_seed_bridge(
    archive_record: Mapping[str, Any],
) -> PhotoBeamRenderSeedBridge:
    """Validate a serialized verifier and preserve its entire ordered beam."""

    archive = validate_frozen_photo_geometry_verifier(archive_record)
    archive_sha = _sha256(archive.get("archive_sha256"), "photo verifier archive")
    atlas_sha = _sha256(archive.get("source_atlas_sha256"), "source atlas")
    evidence = _mapping(archive.get("evidence"), "photo verifier evidence")
    observation = _mapping(evidence.get("observation"), "photo observation provenance")
    models = _mapping(evidence.get("models"), "photo evidence models")
    edge_model = _mapping(models.get("edge"), "photo edge model")
    depth_model = _mapping(models.get("depth"), "photo depth model")

    candidates = archive.get("candidates")
    beam_ids = archive.get("beam_candidate_ids")
    if not isinstance(candidates, list) or not isinstance(beam_ids, list):
        raise ValueError("validated photo verifier is missing its candidate beam")
    candidate_by_id: dict[str, dict[str, Any]] = {}
    for value in candidates:
        candidate = _mapping(value, "photo verifier candidate")
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("photo verifier candidate ID is missing")
        if candidate_id in candidate_by_id:
            raise ValueError(f"duplicate photo verifier candidate ID: {candidate_id!r}")
        candidate_by_id[candidate_id] = candidate
    if not beam_ids or len(set(beam_ids)) != len(beam_ids):
        raise ValueError("photo verifier beam IDs must be non-empty and unique")

    seeds: list[PhotoBeamRenderSeed] = []
    for beam_rank, candidate_id in enumerate(beam_ids, start=1):
        if not isinstance(candidate_id, str) or candidate_id not in candidate_by_id:
            raise ValueError(f"photo verifier beam references missing candidate: {candidate_id!r}")
        candidate = candidate_by_id[candidate_id]
        if candidate.get("beam_rank") != beam_rank:
            raise ValueError(f"photo verifier beam rank is inconsistent for {candidate_id!r}")
        atlas_candidate = _mapping(candidate.get("atlas_candidate"), "nested atlas candidate")
        if atlas_candidate.get("candidate_id") != candidate_id:
            raise ValueError(f"nested atlas candidate ID is inconsistent for {candidate_id!r}")
        pose = _mapping(atlas_candidate.get("pose"), "nested atlas candidate pose")
        position = _mapping(pose.get("position"), "nested atlas candidate position")
        seed = PhotoBeamRenderSeed(
            candidate_id=candidate_id,
            beam_rank=beam_rank,
            verifier_rank=_positive_int(candidate.get("verifier_rank"), "verifier rank"),
            original_estimator_rank=_positive_int(
                candidate.get("original_estimator_rank"),
                "original estimator rank",
            ),
            east_m=_finite_float(position.get("east_m"), "candidate east"),
            north_m=_finite_float(position.get("north_m"), "candidate north"),
            up_m=_finite_float(position.get("up_m"), "candidate up"),
            yaw_deg=_finite_float(pose.get("yaw_deg"), "candidate yaw"),
            query_initial_pitch_deg=_finite_float(pose.get("pitch_deg"), "candidate query-initial pitch"),
            discarded_atlas_roll_nuisance_deg=_finite_float(
                pose.get("roll_deg"),
                "candidate roll nuisance",
            ),
        )
        seed.identified_render_seed()
        seeds.append(seed)

    frozen_seeds = tuple(seeds)
    observation_json = json.dumps(
        observation,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    basis = _bridge_basis(
        source_verifier_archive_sha256=archive_sha,
        source_atlas_sha256=atlas_sha,
        photo_rgb_sha256=_sha256(evidence.get("photo_rgb_sha256"), "photo RGB"),
        observed_skyline_sha256=_sha256(evidence.get("observed_skyline_sha256"), "observed skyline"),
        edge_model_sha256=_sha256(edge_model.get("aggregate_sha256"), "photo edge model"),
        depth_model_sha256=_sha256(depth_model.get("aggregate_sha256"), "photo depth model"),
        observation_provenance_json=observation_json,
        seeds=frozen_seeds,
    )
    return PhotoBeamRenderSeedBridge(
        source_verifier_archive_sha256=archive_sha,
        source_atlas_sha256=atlas_sha,
        photo_rgb_sha256=str(evidence["photo_rgb_sha256"]),
        observed_skyline_sha256=str(evidence["observed_skyline_sha256"]),
        edge_model_sha256=str(edge_model["aggregate_sha256"]),
        depth_model_sha256=str(depth_model["aggregate_sha256"]),
        observation_provenance_json=observation_json,
        seeds=frozen_seeds,
        bridge_sha256=_canonical_sha256(basis),
    )


def _bridge_basis(
    *,
    source_verifier_archive_sha256: str,
    source_atlas_sha256: str,
    photo_rgb_sha256: str,
    observed_skyline_sha256: str,
    edge_model_sha256: str,
    depth_model_sha256: str,
    observation_provenance_json: str,
    seeds: tuple[PhotoBeamRenderSeed, ...],
) -> dict[str, Any]:
    return {
        "schema": PHOTO_BEAM_RENDER_SEED_BRIDGE_SCHEMA,
        "method": PHOTO_BEAM_RENDER_SEED_BRIDGE_METHOD,
        "source": {
            "photo_verifier_archive_sha256": source_verifier_archive_sha256,
            "skyline_atlas_archive_sha256": source_atlas_sha256,
            "photo_rgb_sha256": photo_rgb_sha256,
            "observed_skyline_sha256": observed_skyline_sha256,
            "model_sha256": {"edge": edge_model_sha256, "depth": depth_model_sha256},
            "observation": json.loads(observation_provenance_json),
        },
        "truth_boundary": {
            "photo_observable_evidence_only": True,
            "source_depth_reference_used": False,
            "numeric_evaluation_reference_used": False,
            "bridge_inputs": ["serialized_frozen_photo_geometry_verifier_archive"],
            "truth_or_evaluation_inputs": [],
        },
        "beam": {
            "source_policy": "greedy_position_yaw_basin_nms",
            "order_preserved": True,
            "truncated": False,
            "reranked": False,
            "seed_count": len(seeds),
            "ordered_candidate_ids": [seed.candidate_id for seed in seeds],
        },
        "pose_semantics": {
            "atlas_pitch": "query_pnp_initialization_only_not_physical_render_pitch",
            "atlas_roll": "discarded_crop_slope_nuisance_not_forwarded_to_render_seed",
            "round_zero_physical_render_roll_deg": 0.0,
        },
        "seeds": [seed.to_record() for seed in seeds],
    }


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} SHA-256 is missing or malformed")
    return value


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()

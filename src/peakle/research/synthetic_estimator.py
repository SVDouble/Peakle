"""Serialized, truth-free subprocess boundary for the synthetic estimator."""

from __future__ import annotations

import argparse
import builtins
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics
from peakle.domain.coordinates import LocalPoint
from peakle.domain.terrain import TerrainMap, TerrainSpec
from peakle.io.artifacts import write_once_bytes
from peakle.localize.synthetic_pipeline_bench import (
    SyntheticSearchConfig,
    build_synthetic_candidate_archive,
    canonical_json_bytes,
)

REQUEST_SCHEMA = "peakle_synthetic_estimator_request_v1"
_ARRAY_NAMES = ("x_m", "y_m", "elevation_m", "latitude_deg", "longitude_deg")


class EstimatorInputError(ValueError):
    pass


class _StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class ArtifactRef(_StrictRecord):
    filename: str
    sha256: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0)

    @field_validator("filename")
    @classmethod
    def _safe_filename(cls, value: str) -> str:
        return _safe_component(value)

    @classmethod
    def for_bytes(cls, filename: str, content: builtins.bytes) -> ArtifactRef:
        return cls(filename=filename, sha256=_sha256(content), bytes=len(content))


class EstimatorIntrinsics(_StrictRecord):
    width_px: int = Field(ge=1)
    height_px: int = Field(ge=1)
    focal_length_px: float = Field(gt=0.0)
    principal_x_px: float
    principal_y_px: float

    def to_domain(self) -> CameraIntrinsics:
        return CameraIntrinsics(**self.model_dump())

    @model_validator(mode="after")
    def _validate_domain(self) -> EstimatorIntrinsics:
        self.to_domain()
        return self


class EstimatorPosition(_StrictRecord):
    east_m: float
    north_m: float
    up_m: float


class EstimatorPrior(_StrictRecord):
    position: EstimatorPosition
    yaw_deg: float = Field(ge=-360.0, le=360.0)
    pitch_deg: float = Field(ge=-89.0, le=89.0)
    roll_deg: float = Field(ge=-180.0, le=180.0)

    def to_domain(self) -> CameraExtrinsics:
        return CameraExtrinsics(
            position=LocalPoint(**self.position.model_dump()),
            **self.model_dump(exclude={"position"}),
        )


class EstimatorSearchConfig(_StrictRecord):
    position_spacing_m: float = Field(gt=0.0)
    position_radius_steps: int = Field(ge=1)
    yaw_spacing_deg: float = Field(gt=0.0)
    yaw_radius_steps: int = Field(ge=1)
    eye_height_m: float = Field(gt=0.0)
    render_stride: int = Field(ge=1)
    depth_log_cap: float = Field(gt=0.0)
    outline_distance_cap_px: float = Field(gt=0.0)
    skyline_residual_cap_px: float = Field(gt=0.0)
    target_position_m: float = Field(gt=0.0)
    target_yaw_deg: float = Field(gt=0.0)
    ambiguity_score_delta: float = Field(ge=0.0)
    ambiguity_position_separation_m: float = Field(gt=0.0)
    ambiguity_yaw_separation_deg: float = Field(gt=0.0)
    minimum_extraction_coverage: float = Field(ge=0.0, le=1.0)
    minimum_extraction_agreement: float = Field(ge=0.0, le=1.0)


class SyntheticEstimatorRequest(_StrictRecord):
    schema_id: Literal["peakle_synthetic_estimator_request_v1"] = Field(REQUEST_SCHEMA, alias="schema")
    estimator_terrain: ArtifactRef
    webgl_semantic_u8: ArtifactRef
    webgl_geometry_f32le: ArtifactRef
    intrinsics: EstimatorIntrinsics
    prior: EstimatorPrior
    search_config: EstimatorSearchConfig
    query_renderer_family: Literal["chromium_webgl2_raw"] = "chromium_webgl2_raw"
    output_filename: str

    @model_validator(mode="before")
    @classmethod
    def _reject_truth_fields(cls, value: Any) -> Any:
        _reject_forbidden_fields(value)
        return value

    @field_validator("output_filename")
    @classmethod
    def _safe_output_filename(cls, value: str) -> str:
        return _safe_component(value)


@dataclass(frozen=True, slots=True)
class ImmutableArtifact:
    ref: ArtifactRef
    content: bytes

    def __post_init__(self) -> None:
        if self.ref.bytes != len(self.content) or self.ref.sha256 != _sha256(self.content):
            raise ValueError("immutable artifact content does not match its reference")


@dataclass(frozen=True, slots=True)
class SyntheticEstimatorRun:
    archive: dict[str, Any]
    archive_artifact: ImmutableArtifact
    request_artifact: ImmutableArtifact
    terrain_artifact: ImmutableArtifact
    request: SyntheticEstimatorRequest


def encode_terrain_bundle(terrain: TerrainMap) -> bytes:
    arrays = {}
    for name in _ARRAY_NAMES:
        array = np.asarray(getattr(terrain, name), dtype=np.float64)
        arrays[name] = {"dtype": "float64", "shape": list(array.shape), "values": array.ravel().tolist()}
    return canonical_json_bytes(
        {"schema": "peakle_terrain_bundle_v1", "spec": terrain.spec.model_dump(mode="json"), "arrays": arrays}
    )


def decode_terrain_bundle(content: bytes) -> TerrainMap:
    payload = json.loads(content)
    _expect_keys(payload, {"schema", "spec", "arrays"}, "terrain bundle")
    if payload["schema"] != "peakle_terrain_bundle_v1":
        raise EstimatorInputError("unknown terrain bundle schema")
    spec_payload = payload["spec"]
    _expect_keys(
        spec_payload,
        {"origin", "width_m", "height_m", "grid_width", "grid_height", "min_elevation_m", "max_elevation_m", "seed"},
        "terrain spec",
    )
    _expect_keys(spec_payload["origin"], {"latitude_deg", "longitude_deg", "elevation_m"}, "terrain origin")
    spec = TerrainSpec.model_validate(spec_payload, strict=True)
    _expect_keys(payload["arrays"], set(_ARRAY_NAMES), "terrain arrays")
    arrays = {name: _decode_array(payload["arrays"][name], name) for name in _ARRAY_NAMES}
    return TerrainMap(spec=spec, **arrays)


def execute_estimator_request(request_path: Path) -> bytes:
    request_path = request_path.expanduser().resolve()
    raw = json.loads(request_path.read_bytes())
    request = SyntheticEstimatorRequest.model_validate(raw)
    root = request_path.parent
    terrain = decode_terrain_bundle(_read_verified(root, request.estimator_terrain))
    semantic = _read_verified(root, request.webgl_semantic_u8)
    geometry = _read_verified(root, request.webgl_geometry_f32le)
    skyline, depth = _query_arrays(semantic, geometry, request.intrinsics)
    archive = build_synthetic_candidate_archive(
        terrain,
        request.intrinsics.to_domain(),
        request.prior.to_domain(),
        {"oracle_mask": skyline},
        depth,
        config=SyntheticSearchConfig(**request.search_config.model_dump()),
        query_renderer_family=request.query_renderer_family,
    )
    encoded = canonical_json_bytes(archive)
    write_once_bytes(root / request.output_filename, encoded)
    return encoded


def run_synthetic_estimator(
    terrain: TerrainMap,
    intrinsics: CameraIntrinsics,
    prior: CameraExtrinsics,
    config: SyntheticSearchConfig,
    *,
    semantic_u8: bytes,
    geometry_f32le: bytes,
    query_renderer_family: Literal["chromium_webgl2_raw"] = "chromium_webgl2_raw",
    terrain_filename: str = "estimator-terrain.json",
    semantic_filename: str = "query-semantic.u8",
    geometry_filename: str = "query-geometry.f32le",
    request_filename: str = "estimator-request.json",
    output_filename: str = "candidate-archive.json",
    timeout_s: float = 600.0,
) -> SyntheticEstimatorRun:
    if not math.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError("timeout_s must be finite and positive")
    terrain_bytes = encode_terrain_bundle(terrain)
    terrain_ref = ArtifactRef.for_bytes(terrain_filename, terrain_bytes)
    semantic_ref = ArtifactRef.for_bytes(semantic_filename, semantic_u8)
    geometry_ref = ArtifactRef.for_bytes(geometry_filename, geometry_f32le)
    request = SyntheticEstimatorRequest(
        estimator_terrain=terrain_ref,
        webgl_semantic_u8=semantic_ref,
        webgl_geometry_f32le=geometry_ref,
        intrinsics=EstimatorIntrinsics(**intrinsics.model_dump()),
        prior=EstimatorPrior(**prior.model_dump()),
        search_config=EstimatorSearchConfig(**asdict(config)),
        query_renderer_family=query_renderer_family,
        output_filename=output_filename,
    )
    request_bytes = canonical_json_bytes(request.model_dump(mode="json", by_alias=True))
    with tempfile.TemporaryDirectory(prefix="peakle-estimator-") as temporary:
        root = Path(temporary)
        for ref, content in ((terrain_ref, terrain_bytes), (semantic_ref, semantic_u8), (geometry_ref, geometry_f32le)):
            write_once_bytes(root / ref.filename, content)
        request_ref = ArtifactRef.for_bytes(request_filename, request_bytes)
        write_once_bytes(root / request_ref.filename, request_bytes)
        completed = subprocess.run(
            (sys.executable, "-m", "peakle.research.synthetic_estimator", "--request", request_ref.filename),
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(f"synthetic estimator worker failed: {detail or completed.returncode}")
        archive_bytes = (root / output_filename).read_bytes()
    archive = json.loads(archive_bytes)
    if canonical_json_bytes(archive) != archive_bytes:
        raise RuntimeError("synthetic estimator worker returned non-canonical archive bytes")
    return SyntheticEstimatorRun(
        archive=archive,
        archive_artifact=ImmutableArtifact(ArtifactRef.for_bytes(output_filename, archive_bytes), archive_bytes),
        request_artifact=ImmutableArtifact(request_ref, request_bytes),
        terrain_artifact=ImmutableArtifact(terrain_ref, terrain_bytes),
        request=request,
    )


def _query_arrays(
    semantic_bytes: bytes, geometry_bytes: bytes, intrinsics: EstimatorIntrinsics
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    height, width = intrinsics.height_px, intrinsics.width_px
    if len(semantic_bytes) != height * width or len(geometry_bytes) != height * width * 4 * 4:
        raise EstimatorInputError("query arrays do not match the declared image dimensions")
    semantic = np.frombuffer(semantic_bytes, dtype=np.uint8).reshape(height, width)
    values = set(int(value) for value in np.unique(semantic))
    if not values <= {0, 1}:
        raise EstimatorInputError(f"unknown WebGL semantic IDs: {sorted(values)}")
    geometry = np.frombuffer(geometry_bytes, dtype="<f4").reshape(height, width, 4)
    if not np.all(np.isfinite(geometry)):
        raise EstimatorInputError("WebGL geometry contains non-finite values")
    mask = semantic == 1
    if np.any(mask & (geometry[..., 3] <= 0.0)):
        raise EstimatorInputError("terrain pixels need positive WebGL forward depth")
    depth = np.where(mask, geometry[..., 3], np.nan).astype(np.float64)
    skyline = np.full(width, np.nan, dtype=np.float64)
    occupied = np.any(mask, axis=0)
    skyline[occupied] = np.argmax(mask[:, occupied], axis=0).astype(np.float64)
    return skyline, depth


def _read_verified(root: Path, ref: ArtifactRef) -> bytes:
    content = (root / ref.filename).read_bytes()
    if len(content) != ref.bytes or _sha256(content) != ref.sha256:
        raise EstimatorInputError(f"artifact hash or byte count mismatch: {ref.filename}")
    return content


def _decode_array(value: Any, name: str) -> NDArray[np.float64]:
    _expect_keys(value, {"dtype", "shape", "values"}, f"terrain array {name}")
    if value["dtype"] != "float64" or not isinstance(value["shape"], list) or not isinstance(value["values"], list):
        raise EstimatorInputError(f"invalid terrain array encoding: {name}")
    shape = value["shape"]
    if not shape or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in shape):
        raise EstimatorInputError(f"invalid terrain array shape: {name}")
    values = value["values"]
    if any(isinstance(item, bool) or not isinstance(item, int | float) or not math.isfinite(item) for item in values):
        raise EstimatorInputError(f"invalid terrain array value: {name}")
    if math.prod(shape) != len(values):
        raise EstimatorInputError(f"terrain array length does not match shape: {name}")
    return np.asarray(values, dtype=np.float64).reshape(tuple(shape))


def _expect_keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise EstimatorInputError(f"{label} must contain exactly {sorted(expected)}")


def _reject_forbidden_fields(value: Any) -> None:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="python", by_alias=True)
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            forbidden = (
                "truth" in normalized
                or normalized
                in {"gt", "ground_truth", "reference_pose", "query_pose", "query_camera", "query_intrinsics"}
                or "query_extrinsic" in normalized
                or "matrix" in normalized
                or normalized.startswith("expected_decision")
                or normalized.startswith("expected_action")
                or normalized.startswith("expected_pose")
            )
            if forbidden:
                raise EstimatorInputError(f"forbidden estimator request field: {key}")
            _reject_forbidden_fields(item)
    elif isinstance(value, list | tuple):
        for item in value:
            _reject_forbidden_fields(item)


def _safe_component(value: str) -> str:
    if not value or len(value) > 255 or Path(value).name != value or value in {".", ".."}:
        raise ValueError("artifact filename must be one safe path component")
    if any(not (character.isascii() and (character.isalnum() or character in "._-")) for character in value):
        raise ValueError("artifact filename contains unsupported characters")
    return value


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        execute_estimator_request(args.request)
    except Exception as exc:  # noqa: BLE001 - CLI exposes a single safe failure boundary
        print(f"invalid synthetic estimator request: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

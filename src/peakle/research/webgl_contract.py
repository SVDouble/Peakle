"""Typed immutable boundary for independent WebGL query observations."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

ARTIFACT_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

_QUERY_ARTIFACT_SUFFIXES = {
    "sealed_truth_scene_json": ".scene.json",
    "rgb_rgba8": ".rgb.rgba8",
    "normal_xyz_depth_f32le": ".geometry.f32le",
    "semantic_u8": ".semantic.u8",
}
_OBSERVATION_ROLES = frozenset({"rgb_rgba8", "normal_xyz_depth_f32le", "semantic_u8"})


class _FrozenRecord(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False, populate_by_name=True, serialize_by_alias=True)


class WebGLCapabilities(_FrozenRecord):
    version: str
    shading_language_version: str
    vendor: str
    renderer: str
    unmasked_vendor: str | None
    unmasked_renderer: str | None
    max_texture_size: int
    max_color_attachments: int
    max_draw_buffers: int
    ext_color_buffer_float: Literal[True]


class CameraWorldAxes(_FrozenRecord):
    right: tuple[float, float, float]
    down: tuple[float, float, float]
    forward: tuple[float, float, float]


class QueryOutputHashes(_FrozenRecord):
    rgb_rgba8_sha256: str
    normal_xyz_depth_f32le_sha256: str
    semantic_u8_sha256: str


class QueryArtifactFile(_FrozenRecord):
    filename: str
    role: Literal["sealed_truth_scene_json", "rgb_rgba8", "normal_xyz_depth_f32le", "semantic_u8"]
    media_type: Literal["application/json", "application/octet-stream"]
    dtype: Literal["uint8", "float32_little_endian"] | None
    shape: tuple[int, ...] | None
    sha256: str
    bytes: int


class RendererRuntimeResource(_FrozenRecord):
    name: str
    path: str
    role: Literal["renderer_executable", "software_rasterizer_library", "driver_manifest"]
    sha256: str


class WebGLQueryProvenance(_FrozenRecord):
    schema_id: Literal["peakle_independent_webgl_query_provenance_v1"] = Field(
        "peakle_independent_webgl_query_provenance_v1",
        alias="schema",
    )
    renderer_family: Literal["chromium_webgl2_raw"] = "chromium_webgl2_raw"
    independence_class: Literal["independent_rasterizer_same_scene_model"] = "independent_rasterizer_same_scene_model"
    calls_peakle_renderer_or_projection_helpers: Literal[False] = False
    network_policy: Literal["disabled_browser_network_file_input_only"] = "disabled_browser_network_file_input_only"
    input_schema: Literal["peakle_independent_webgl_query_input_v1"] = "peakle_independent_webgl_query_input_v1"
    output_schema: Literal["peakle_independent_webgl_query_output_v1"] = "peakle_independent_webgl_query_output_v1"
    output_row_order: Literal["top_to_bottom"] = "top_to_bottom"
    output_byte_order: Literal["little"] = "little"
    browser_version: str
    browser_executable: str
    browser_executable_sha256: str
    renderer_runtime_resources: tuple[RendererRuntimeResource, ...]
    browser_flags: tuple[str, ...]
    worker_sha256: str
    input_scene_sha256: str
    positions_f32le_sha256: str
    triangles_u32le_sha256: str
    output_hashes: QueryOutputHashes
    camera_world_axes: CameraWorldAxes
    camera_world_from_rdf_matrix_col_major: tuple[float, ...] = Field(min_length=16, max_length=16)
    capabilities: WebGLCapabilities
    renderer_stratum_sha256: str
    runtime_s: float


@dataclass(frozen=True, slots=True)
class WebGLQueryRender:
    """Validated query observations returned by the external renderer."""

    rgb: NDArray[np.uint8]
    terrain_mask: NDArray[np.bool_]
    forward_depth_m: NDArray[np.float64]
    camera_normals: NDArray[np.float64]
    skyline_profile: NDArray[np.float64]
    sealed_input_scene_json: bytes
    provenance: WebGLQueryProvenance

    def __post_init__(self) -> None:
        height, width = self.terrain_mask.shape
        expected = (height, width)
        if self.rgb.shape != (*expected, 3):
            raise ValueError("query RGB and semantic mask dimensions differ")
        if self.forward_depth_m.shape != expected or self.camera_normals.shape != (*expected, 3):
            raise ValueError("query geometry and semantic mask dimensions differ")
        if self.skyline_profile.shape != (width,):
            raise ValueError("query skyline width differs from the render")
        mask = np.asarray(self.terrain_mask, dtype=np.bool_)
        depth = np.asarray(self.forward_depth_m, dtype=np.float64)
        normals = np.asarray(self.camera_normals, dtype=np.float64)
        if np.any(mask & (~np.isfinite(depth) | (depth <= 0.0))):
            raise ValueError("terrain pixels need finite positive forward depth")
        if np.any(~mask & np.isfinite(depth)):
            raise ValueError("sky pixels must not carry metric depth")
        lengths = np.linalg.norm(normals[mask], axis=1)
        if lengths.size and (not np.all(np.isfinite(lengths)) or np.max(np.abs(lengths - 1.0)) > 2e-4):
            raise ValueError("terrain camera normals must be unit vectors")
        if np.any(np.isfinite(normals[~mask])):
            raise ValueError("sky pixels must not carry camera normals")
        if sha256_bytes(self.sealed_input_scene_json) != self.provenance.input_scene_sha256:
            raise ValueError("sealed input scene does not match its provenance hash")
        for field_name in ("rgb", "terrain_mask", "forward_depth_m", "camera_normals", "skyline_profile"):
            readonly = np.asarray(getattr(self, field_name)).view()
            readonly.flags.writeable = False
            object.__setattr__(self, field_name, readonly)


@dataclass(frozen=True, slots=True)
class FrozenWebGLQueryArtifact:
    """Immutable raw query arrays ready for flat artifact publication."""

    files: MappingProxyType[str, bytes]
    manifest: tuple[QueryArtifactFile, ...]


@dataclass(frozen=True, slots=True)
class LoadedWebGLQueryArtifact:
    """Hash-validated frozen observations, without sealed truth by default."""

    rgb: NDArray[np.uint8]
    terrain_mask: NDArray[np.bool_]
    forward_depth_m: NDArray[np.float64]
    camera_normals: NDArray[np.float64]
    sealed_input_scene_json: bytes | None = None


def freeze_webgl_query_artifact(render: WebGLQueryRender, artifact_id: str) -> FrozenWebGLQueryArtifact:
    """Re-encode and hash the browser outputs before estimator execution."""

    if ARTIFACT_ID_PATTERN.fullmatch(artifact_id) is None:
        raise ValueError("query artifact_id must contain only letters, digits, dot, underscore, or hyphen")
    height, width = render.terrain_mask.shape
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    rgba[..., :3] = render.rgb
    rgba[..., 3] = 255
    geometry = np.zeros((height, width, 4), dtype="<f4")
    geometry[render.terrain_mask, :3] = render.camera_normals[render.terrain_mask]
    geometry[render.terrain_mask, 3] = render.forward_depth_m[render.terrain_mask]
    semantic = np.asarray(render.terrain_mask, dtype=np.uint8)
    records = (
        ("rgb.rgba8", "rgb_rgba8", "uint8", rgba, render.provenance.output_hashes.rgb_rgba8_sha256),
        (
            "geometry.f32le",
            "normal_xyz_depth_f32le",
            "float32_little_endian",
            geometry,
            render.provenance.output_hashes.normal_xyz_depth_f32le_sha256,
        ),
        ("semantic.u8", "semantic_u8", "uint8", semantic, render.provenance.output_hashes.semantic_u8_sha256),
    )
    scene_filename = f"{artifact_id}.scene.json"
    scene_content = render.sealed_input_scene_json
    scene_hash = sha256_bytes(scene_content)
    if scene_hash != render.provenance.input_scene_sha256:
        raise RuntimeError("frozen query scene does not match the renderer input hash")
    files: dict[str, bytes] = {scene_filename: scene_content}
    manifest: list[QueryArtifactFile] = [
        QueryArtifactFile(
            filename=scene_filename,
            role="sealed_truth_scene_json",
            media_type="application/json",
            dtype=None,
            shape=None,
            sha256=scene_hash,
            bytes=len(scene_content),
        )
    ]
    for suffix, role, dtype, array, expected_hash in records:
        filename = f"{artifact_id}.{suffix}"
        content = np.ascontiguousarray(array).tobytes()
        actual_hash = sha256_bytes(content)
        if actual_hash != expected_hash:
            raise RuntimeError(f"frozen query {role} bytes do not match the browser output hash")
        files[filename] = content
        manifest.append(
            QueryArtifactFile(
                filename=filename,
                role=role,
                media_type="application/octet-stream",
                dtype=dtype,
                shape=tuple(int(value) for value in array.shape),
                sha256=actual_hash,
                bytes=len(content),
            )
        )
    return FrozenWebGLQueryArtifact(files=MappingProxyType(files), manifest=tuple(manifest))


def load_frozen_webgl_query_artifact(
    manifest: Sequence[QueryArtifactFile],
    files: Mapping[str, bytes],
    *,
    include_sealed_truth: bool = False,
) -> LoadedWebGLQueryArtifact:
    """Validate and decode a frozen WebGL query without crossing the truth boundary.

    The manifest must describe the complete four-file artifact. ``files`` may omit
    the sealed scene for the ordinary observation-only path; if it is present, its
    value is deliberately not requested unless ``include_sealed_truth`` is true.
    """

    records = _validate_query_manifest(manifest)
    observation_names = {records[role].filename for role in _OBSERVATION_ROLES}
    supplied_names = set(files)
    allowed_names = observation_names | {records["sealed_truth_scene_json"].filename}
    missing = observation_names - supplied_names
    extra = supplied_names - allowed_names
    if missing:
        raise ValueError(f"frozen query files are missing: {', '.join(sorted(missing))}")
    if extra:
        raise ValueError(f"frozen query files contain undeclared entries: {', '.join(sorted(extra))}")

    rgb_record = records["rgb_rgba8"]
    geometry_record = records["normal_xyz_depth_f32le"]
    semantic_record = records["semantic_u8"]
    rgba = _decode_array(rgb_record, files, np.dtype(np.uint8))
    geometry = _decode_array(geometry_record, files, np.dtype("<f4"))
    semantic = _decode_array(semantic_record, files, np.dtype(np.uint8))

    height, width, _ = rgba.shape
    if geometry.shape != (height, width, 4) or semantic.shape != (height, width):
        raise ValueError("frozen query RGB, geometry, and semantic shapes differ")
    if np.any(rgba[..., 3] != 255):
        raise ValueError("frozen query RGBA alpha channel must be opaque")
    if np.any((semantic != 0) & (semantic != 1)):
        raise ValueError("frozen query semantic mask must contain only 0 and 1")

    terrain_mask = semantic.astype(np.bool_)
    packed_normals = np.asarray(geometry[..., :3], dtype=np.float64)
    packed_depth = np.asarray(geometry[..., 3], dtype=np.float64)
    if np.any(geometry[~terrain_mask] != 0.0):
        raise ValueError("frozen query sky pixels must have zero packed geometry")
    if np.any(~np.isfinite(packed_depth[terrain_mask]) | (packed_depth[terrain_mask] <= 0.0)):
        raise ValueError("frozen query terrain pixels need finite positive forward depth")
    normal_lengths = np.linalg.norm(packed_normals[terrain_mask], axis=1)
    if normal_lengths.size and (not np.all(np.isfinite(normal_lengths)) or np.max(np.abs(normal_lengths - 1.0)) > 2e-4):
        raise ValueError("frozen query terrain camera normals must be unit vectors")

    forward_depth = packed_depth.copy()
    camera_normals = packed_normals.copy()
    forward_depth[~terrain_mask] = np.nan
    camera_normals[~terrain_mask] = np.nan
    rgb = np.asarray(rgba[..., :3], dtype=np.uint8)
    for array in (rgb, terrain_mask, forward_depth, camera_normals):
        array.flags.writeable = False

    sealed_scene: bytes | None = None
    if include_sealed_truth:
        scene_record = records["sealed_truth_scene_json"]
        if scene_record.filename not in supplied_names:
            raise ValueError(f"frozen query files are missing sealed truth: {scene_record.filename}")
        sealed_scene = _validated_file_bytes(scene_record, files)
    return LoadedWebGLQueryArtifact(
        rgb=rgb,
        terrain_mask=terrain_mask,
        forward_depth_m=forward_depth,
        camera_normals=camera_normals,
        sealed_input_scene_json=sealed_scene,
    )


def load_frozen_webgl_query_rgb(
    manifest: Sequence[QueryArtifactFile],
    files: Mapping[str, bytes],
) -> NDArray[np.uint8]:
    """Open only hash-validated RGB bytes, leaving geometry and truth sealed."""

    records = _validate_query_manifest(manifest)
    rgb_record = records["rgb_rgba8"]
    supplied_names = set(files)
    declared_names = {record.filename for record in records.values()}
    if rgb_record.filename not in supplied_names:
        raise ValueError(f"frozen query files are missing RGB: {rgb_record.filename}")
    extra = supplied_names - declared_names
    if extra:
        raise ValueError(f"frozen query files contain undeclared entries: {', '.join(sorted(extra))}")
    rgba = _decode_array(rgb_record, files, np.dtype(np.uint8))
    if np.any(rgba[..., 3] != 255):
        raise ValueError("frozen query RGBA alpha channel must be opaque")
    rgb = np.asarray(rgba[..., :3], dtype=np.uint8)
    rgb.flags.writeable = False
    return rgb


def _validate_query_manifest(manifest: Sequence[QueryArtifactFile]) -> dict[str, QueryArtifactFile]:
    expected_roles = set(_QUERY_ARTIFACT_SUFFIXES)
    records: dict[str, QueryArtifactFile] = {}
    filenames: set[str] = set()
    artifact_ids: set[str] = set()
    for record in manifest:
        if not isinstance(record, QueryArtifactFile):
            raise TypeError("frozen query manifest entries must be QueryArtifactFile records")
        if record.filename in filenames:
            raise ValueError(f"frozen query manifest has duplicate filename: {record.filename}")
        if record.role in records:
            raise ValueError(f"frozen query manifest has duplicate role: {record.role}")
        suffix = _QUERY_ARTIFACT_SUFFIXES[record.role]
        if not record.filename.endswith(suffix):
            raise ValueError(f"frozen query filename does not match role {record.role}: {record.filename}")
        artifact_id = record.filename[: -len(suffix)]
        if not artifact_id or ARTIFACT_ID_PATTERN.fullmatch(artifact_id) is None:
            raise ValueError(f"invalid frozen query artifact filename: {record.filename}")
        if _SHA256_PATTERN.fullmatch(record.sha256) is None:
            raise ValueError(f"invalid SHA-256 for frozen query file: {record.filename}")
        if record.bytes < 0:
            raise ValueError(f"invalid byte count for frozen query file: {record.filename}")
        filenames.add(record.filename)
        artifact_ids.add(artifact_id)
        records[record.role] = record
    missing_roles = expected_roles - set(records)
    extra_roles = set(records) - expected_roles
    if missing_roles or extra_roles:
        details = []
        if missing_roles:
            details.append(f"missing roles {', '.join(sorted(missing_roles))}")
        if extra_roles:
            details.append(f"unexpected roles {', '.join(sorted(extra_roles))}")
        raise ValueError(f"invalid frozen query manifest: {'; '.join(details)}")
    if len(artifact_ids) != 1:
        raise ValueError("frozen query manifest filenames do not share one artifact id")

    scene = records["sealed_truth_scene_json"]
    if scene.media_type != "application/json" or scene.dtype is not None or scene.shape is not None:
        raise ValueError("sealed truth scene record has invalid media type, dtype, or shape")
    _validate_array_record(records["rgb_rgba8"], "uint8", dimensions=3, channels=4, itemsize=1)
    _validate_array_record(
        records["normal_xyz_depth_f32le"],
        "float32_little_endian",
        dimensions=3,
        channels=4,
        itemsize=4,
    )
    _validate_array_record(records["semantic_u8"], "uint8", dimensions=2, channels=None, itemsize=1)
    return records


def _validate_array_record(
    record: QueryArtifactFile,
    dtype: str,
    *,
    dimensions: int,
    channels: int | None,
    itemsize: int,
) -> None:
    if record.media_type != "application/octet-stream" or record.dtype != dtype:
        raise ValueError(f"frozen query {record.role} record has invalid media type or dtype")
    if record.shape is None or len(record.shape) != dimensions or any(size <= 0 for size in record.shape):
        raise ValueError(f"frozen query {record.role} record has invalid shape")
    if channels is not None and record.shape[-1] != channels:
        raise ValueError(f"frozen query {record.role} record must have {channels} channels")
    expected_bytes = math.prod(record.shape) * itemsize
    if record.bytes != expected_bytes:
        raise ValueError(f"frozen query {record.role} record byte count does not match its dtype and shape")


def _decode_array(
    record: QueryArtifactFile,
    files: Mapping[str, bytes],
    dtype: np.dtype,
) -> NDArray:
    content = _validated_file_bytes(record, files)
    assert record.shape is not None
    return np.frombuffer(content, dtype=dtype).reshape(record.shape)


def _validated_file_bytes(record: QueryArtifactFile, files: Mapping[str, bytes]) -> bytes:
    content = files[record.filename]
    if not isinstance(content, bytes):
        raise TypeError(f"frozen query file is not immutable bytes: {record.filename}")
    if len(content) != record.bytes:
        raise ValueError(f"frozen query file byte count differs from its manifest: {record.filename}")
    if sha256_bytes(content) != record.sha256:
        raise ValueError(f"frozen query file SHA-256 differs from its manifest: {record.filename}")
    return content


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

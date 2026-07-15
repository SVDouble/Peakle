"""Typed immutable boundary for independent WebGL query observations."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

ARTIFACT_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

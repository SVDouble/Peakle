"""Isolated Chromium/WebGL query renderer for independent-rasterizer studies.

This truth-side adapter intentionally does not import Peakle's rendering or
projection modules. The browser consumes only explicit mesh/camera numbers and
returns immutable image-space observations before an estimator is invoked.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import math
import re
import shutil
import subprocess
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics
from peakle.domain.terrain import TerrainMap
from peakle.research.webgl_contract import (
    CameraWorldAxes,
    QueryOutputHashes,
    RendererRuntimeResource,
    WebGLCapabilities,
    WebGLQueryProvenance,
    WebGLQueryRender,
    sha256_bytes,
)

INPUT_SCHEMA = "peakle_independent_webgl_query_input_v1"
OUTPUT_SCHEMA = "peakle_independent_webgl_query_output_v1"
RESULT_PATTERN = re.compile(r'<pre id="peakle-result">(.*?)</pre>', re.DOTALL)


def terrain_mesh_buffers(terrain: TerrainMap) -> tuple[NDArray[np.float32], NDArray[np.uint32]]:
    """Export explicit ENU vertices and fixed upward-wound heightfield triangles."""

    x_grid, y_grid = np.meshgrid(terrain.x_m, terrain.y_m)
    positions = np.column_stack((x_grid.ravel(), y_grid.ravel(), terrain.elevation_m.ravel())).astype("<f4")
    width = terrain.x_m.size
    height = terrain.y_m.size
    triangles = np.empty(((height - 1) * (width - 1) * 2, 3), dtype="<u4")
    cursor = 0
    for row in range(height - 1):
        base = row * width
        below = (row + 1) * width
        for column in range(width - 1):
            a, b = base + column, base + column + 1
            c, d = below + column, below + column + 1
            triangles[cursor] = (a, b, c)
            triangles[cursor + 1] = (b, d, c)
            cursor += 2
    return positions, triangles


def render_terrain_webgl_query(
    terrain: TerrainMap,
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
    *,
    chromium_path: str | Path | None = None,
    timeout_s: float = 30.0,
) -> WebGLQueryRender:
    """Render a full-resolution terrain mesh in an isolated browser process."""

    positions, triangles = terrain_mesh_buffers(terrain)
    return render_webgl_mesh_query(
        positions,
        triangles,
        intrinsics,
        extrinsics,
        elevation_range_m=(float(np.min(terrain.elevation_m)), float(np.max(terrain.elevation_m))),
        chromium_path=chromium_path,
        timeout_s=timeout_s,
    )


def render_webgl_mesh_query(
    positions_enu_m: NDArray[np.floating],
    triangles: NDArray[np.generic],
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
    *,
    elevation_range_m: tuple[float, float] | None = None,
    chromium_path: str | Path | None = None,
    timeout_s: float = 30.0,
) -> WebGLQueryRender:
    """Render one neutral triangle mesh without using Peakle rendering code."""

    positions, indices = _validated_mesh(positions_enu_m, triangles)
    if not math.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError("timeout_s must be finite and positive")
    elevation_min, elevation_max = elevation_range_m or (
        float(np.min(positions[:, 2])),
        float(np.max(positions[:, 2])),
    )
    if not math.isfinite(elevation_min) or not math.isfinite(elevation_max) or elevation_max < elevation_min:
        raise ValueError("elevation_range_m must be finite and ordered")

    browser = _browser_path(chromium_path)
    worker_path = Path(__file__).with_name("webgl_query.html")
    worker = worker_path.read_text(encoding="utf-8")
    scene = _scene_record(positions, indices, intrinsics, extrinsics, elevation_min, elevation_max)
    scene_bytes = _canonical_json(scene)
    page = worker.replace("__PEAKLE_SCENE_JSON__", scene_bytes.decode(), 1)
    if "__PEAKLE_SCENE_JSON__" in page:
        raise RuntimeError("independent-renderer scene placeholder was not replaced exactly once")

    flags = (
        "--headless",
        "--disable-background-networking",
        "--disable-component-update",
        "--disable-domain-reliability",
        "--disable-sync",
        "--metrics-recording-only",
        "--no-first-run",
        "--no-proxy-server",
        "--use-gl=angle",
        "--use-angle=swiftshader",
        "--dump-dom",
    )
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="peakle-webgl-query-") as temporary:
        root = Path(temporary)
        page_path = root / "query.html"
        page_path.write_text(page, encoding="utf-8")
        command = (
            str(browser),
            f"--user-data-dir={root / 'profile'}",
            *flags,
            page_path.as_uri(),
        )
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    runtime_s = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(f"independent Chromium renderer exited {completed.returncode}: {completed.stderr[-2000:]}")
    output = _browser_result(completed.stdout)
    if not output.get("ok"):
        raise RuntimeError(f"independent WebGL renderer failed: {output.get('error', 'unknown error')}")
    raw_result = output.get("result")
    if not isinstance(raw_result, dict):
        raise RuntimeError("independent WebGL renderer returned no result record")
    result = cast(dict[str, object], raw_result)
    return _decode_render(
        result,
        browser=browser,
        browser_flags=flags,
        worker_bytes=worker.encode(),
        scene_bytes=scene_bytes,
        positions=positions,
        triangles=indices,
        runtime_s=runtime_s,
    )


def _validated_mesh(
    positions: NDArray[np.floating], triangles: NDArray[np.generic]
) -> tuple[NDArray[np.float32], NDArray[np.uint32]]:
    vertices = np.ascontiguousarray(positions, dtype="<f4")
    raw_indices = np.asarray(triangles)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] < 3:
        raise ValueError("positions must have shape (N, 3) with at least three vertices")
    if raw_indices.ndim != 2 or raw_indices.shape[1] != 3 or raw_indices.shape[0] < 1:
        raise ValueError("triangles must have shape (M, 3) with at least one triangle")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("mesh positions must be finite")
    if not np.issubdtype(raw_indices.dtype, np.integer) or np.issubdtype(raw_indices.dtype, np.bool_):
        raise ValueError("triangle indices must use an integer dtype")
    if int(np.min(raw_indices)) < 0:
        raise ValueError("triangle indices must be non-negative")
    if int(np.max(raw_indices)) > np.iinfo(np.uint32).max:
        raise ValueError("triangle index exceeds the uint32 interchange range")
    if int(np.max(raw_indices)) >= vertices.shape[0]:
        raise ValueError("triangle index lies outside the vertex buffer")
    indices = np.ascontiguousarray(raw_indices, dtype="<u4")
    repeated = (indices[:, 0] == indices[:, 1]) | (indices[:, 1] == indices[:, 2]) | (indices[:, 0] == indices[:, 2])
    if np.any(repeated):
        raise ValueError("triangles may not repeat a vertex")
    points = vertices[indices.astype(np.int64)]
    doubled_area = np.linalg.norm(np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1)
    if np.any(doubled_area <= 1e-8):
        raise ValueError("triangles must have non-zero area")
    return vertices, indices


def _scene_record(
    positions: NDArray[np.float32],
    triangles: NDArray[np.uint32],
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
    elevation_min: float,
    elevation_max: float,
) -> dict[str, object]:
    return {
        "schema": INPUT_SCHEMA,
        "coordinate_system": "right_handed_local_enu_metres",
        "camera_axes": "right_down_forward",
        "intrinsics": intrinsics.model_dump(mode="json"),
        "camera": {
            "position_enu_m": list(extrinsics.position.as_tuple()),
            "yaw_deg": extrinsics.yaw_deg,
            "pitch_deg": extrinsics.pitch_deg,
            "roll_deg": extrinsics.roll_deg,
            "near_clip_m": 1.0,
        },
        "mesh": {
            "positions_shape": list(positions.shape),
            "positions_dtype": "float32_little_endian",
            "positions_f32le_base64": base64.b64encode(positions.tobytes()).decode(),
            "triangles_shape": list(triangles.shape),
            "triangles_dtype": "uint32_little_endian",
            "triangles_u32le_base64": base64.b64encode(triangles.tobytes()).decode(),
            "elevation_min_m": elevation_min,
            "elevation_max_m": elevation_max,
            "semantic_class": 1,
        },
    }


def _decode_render(
    result: dict[str, object],
    *,
    browser: Path,
    browser_flags: tuple[str, ...],
    worker_bytes: bytes,
    scene_bytes: bytes,
    positions: NDArray[np.float32],
    triangles: NDArray[np.uint32],
    runtime_s: float,
) -> WebGLQueryRender:
    if result.get("schema") != OUTPUT_SCHEMA or result.get("row_order") != "top_to_bottom":
        raise RuntimeError("independent WebGL renderer returned an unknown output contract")
    if result.get("byte_order") != "little":
        raise RuntimeError("independent WebGL renderer did not return little-endian arrays")
    width = _positive_int_field(result, "width_px")
    height = _positive_int_field(result, "height_px")
    rgb_bytes = _decoded_bytes(result, "rgb_rgba8_base64", width * height * 4)
    geometry_bytes = _decoded_bytes(result, "normal_xyz_depth_f32le_base64", width * height * 16)
    semantic_bytes = _decoded_bytes(result, "semantic_u8_base64", width * height)
    rgb_rgba = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(height, width, 4).copy()
    geometry = np.frombuffer(geometry_bytes, dtype="<f4").reshape(height, width, 4).copy()
    semantic = np.frombuffer(semantic_bytes, dtype=np.uint8).reshape(height, width).copy()
    values = set(int(value) for value in np.unique(semantic))
    if not values <= {0, 1}:
        raise RuntimeError(f"independent WebGL renderer returned unknown semantic IDs: {sorted(values)}")
    mask = semantic == 1
    depth = np.where(mask, geometry[..., 3], np.nan).astype(np.float64)
    normals = np.where(mask[..., None], geometry[..., :3], np.nan).astype(np.float64)
    skyline = _skyline(mask)
    browser_version, browser_hash, runtime_resources = _browser_identity(browser)
    capabilities = WebGLCapabilities.model_validate(result["capabilities"])
    worker_hash = sha256_bytes(worker_bytes)
    renderer_stratum_hash = sha256_bytes(
        _canonical_json(
            {
                "input_schema": INPUT_SCHEMA,
                "output_schema": OUTPUT_SCHEMA,
                "output_row_order": "top_to_bottom",
                "output_byte_order": "little",
                "browser_version": browser_version,
                "browser_executable_sha256": browser_hash,
                "renderer_runtime_resources": [resource.model_dump(mode="json") for resource in runtime_resources],
                "browser_flags": browser_flags,
                "worker_sha256": worker_hash,
                "capabilities": capabilities.model_dump(mode="json"),
            }
        )
    )
    provenance = WebGLQueryProvenance(
        browser_version=browser_version,
        browser_executable=str(browser),
        browser_executable_sha256=browser_hash,
        renderer_runtime_resources=runtime_resources,
        browser_flags=browser_flags,
        worker_sha256=worker_hash,
        input_scene_sha256=sha256_bytes(scene_bytes),
        positions_f32le_sha256=sha256_bytes(positions.tobytes()),
        triangles_u32le_sha256=sha256_bytes(triangles.tobytes()),
        output_hashes=QueryOutputHashes(
            rgb_rgba8_sha256=sha256_bytes(rgb_bytes),
            normal_xyz_depth_f32le_sha256=sha256_bytes(geometry_bytes),
            semantic_u8_sha256=sha256_bytes(semantic_bytes),
        ),
        camera_world_axes=CameraWorldAxes.model_validate(result["camera_world_axes"]),
        camera_world_from_rdf_matrix_col_major=_finite_float_tuple_field(
            result, "camera_world_from_rdf_matrix_col_major", 16
        ),
        capabilities=capabilities,
        renderer_stratum_sha256=renderer_stratum_hash,
        runtime_s=round(runtime_s, 6),
    )
    return WebGLQueryRender(
        rgb=np.asarray(rgb_rgba[..., :3], dtype=np.uint8),
        terrain_mask=mask,
        forward_depth_m=depth,
        camera_normals=normals,
        skyline_profile=skyline,
        sealed_input_scene_json=scene_bytes,
        provenance=provenance,
    )


def _decoded_bytes(result: dict[str, object], field: str, expected: int) -> bytes:
    value = result.get(field)
    if not isinstance(value, str):
        raise RuntimeError(f"independent WebGL renderer omitted {field}")
    decoded = base64.b64decode(value, validate=True)
    if len(decoded) != expected:
        raise RuntimeError(f"independent WebGL renderer returned {len(decoded)} bytes for {field}; expected {expected}")
    return decoded


def _positive_int_field(result: dict[str, object], field: str) -> int:
    value = result.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RuntimeError(f"independent WebGL renderer returned invalid {field}")
    return value


def _finite_float_tuple_field(result: dict[str, object], field: str, length: int) -> tuple[float, ...]:
    value = result.get(field)
    if not isinstance(value, list) or len(value) != length:
        raise RuntimeError(f"independent WebGL renderer returned invalid {field}")
    converted_values: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise RuntimeError(f"independent WebGL renderer returned non-numeric {field}")
        converted_values.append(float(item))
    converted = tuple(converted_values)
    if not all(math.isfinite(item) for item in converted):
        raise RuntimeError(f"independent WebGL renderer returned non-finite {field}")
    return converted


def _skyline(mask: NDArray[np.bool_]) -> NDArray[np.float64]:
    width = mask.shape[1]
    rows = np.full(width, np.nan, dtype=np.float64)
    occupied = np.any(mask, axis=0)
    rows[occupied] = np.argmax(mask[:, occupied], axis=0).astype(np.float64)
    return rows


def _browser_result(document: str) -> dict[str, object]:
    match = RESULT_PATTERN.search(document)
    if match is None:
        raise RuntimeError("independent Chromium renderer returned no result element")
    value = json.loads(html.unescape(match.group(1)))
    if not isinstance(value, dict):
        raise RuntimeError("independent Chromium renderer result must be a JSON object")
    return value


def _browser_path(value: str | Path | None) -> Path:
    candidate = str(value) if value is not None else shutil.which("chromium")
    if candidate is None:
        raise FileNotFoundError("Chromium is required for the independent WebGL renderer")
    path = Path(candidate).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Chromium executable does not exist: {path}")
    return path


@lru_cache(maxsize=4)
def _browser_identity(browser: Path) -> tuple[str, str, tuple[RendererRuntimeResource, ...]]:
    completed = subprocess.run((str(browser), "--version"), check=True, capture_output=True, text=True, timeout=10)
    resources = tuple(
        RendererRuntimeResource(
            name=path.name,
            path=str(path),
            role=_renderer_resource_role(path),
            sha256=_file_sha256(path),
        )
        for path in _renderer_resource_paths(browser)
    )
    return completed.stdout.strip(), _file_sha256(browser), resources


def _renderer_resource_paths(browser: Path) -> tuple[Path, ...]:
    roots = (browser.parent, browser.parent.parent / "lib/chromium", Path("/usr/lib/chromium"))
    names = ("chromium", "libvk_swiftshader.so", "libEGL.so", "libGLESv2.so", "vk_swiftshader_icd.json")
    paths: dict[str, Path] = {}
    for root in roots:
        for name in names:
            candidate = root / name
            if candidate.is_file():
                paths[str(candidate.resolve())] = candidate.resolve()
    return tuple(paths[key] for key in sorted(paths))


def _renderer_resource_role(
    path: Path,
) -> Literal["renderer_executable", "software_rasterizer_library", "driver_manifest"]:
    if path.name == "chromium":
        return "renderer_executable"
    if path.suffix == ".json":
        return "driver_manifest"
    return "software_rasterizer_library"


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

from __future__ import annotations

import hashlib
import json
import math
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from peakle.domain.terrain import TerrainMap
from peakle.localize.atlas_geometry import (
    render_cyltan_candidate_depth,
    terrain_diagonal_range_m,
    validate_frozen_cyltan_atlas,
)
from peakle.localize.skyline_atlas import ATLAS_ARCHIVE_SCHEMA
from peakle.localize.swissdem import Patch


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _candidate() -> dict[str, Any]:
    return {
        "candidate_id": "candidate-1",
        "estimator_rank": 1,
        "pose": {
            "position": {"east_m": 12.0, "north_m": -8.0, "up_m": 2030.0},
            "yaw_deg": 42.0,
            "pitch_deg": 0.0,
            "roll_deg": 1.5,
        },
        "vertical_shift_px": -4.0,
    }


def _atlas() -> dict[str, Any]:
    candidate = _candidate()
    record = {
        "schema": ATLAS_ARCHIVE_SCHEMA,
        "numeric_evaluation_reference_used": False,
        "query_geometry": {
            "projection": "cyltan",
            "width_px": 16,
            "height_px": 12,
            "horizontal_fov_deg": 60.0,
        },
        "candidate_pool": "spatially_diverse_yaw_shortlist",
        "candidate_count": 1,
        "selected_candidate_id": candidate["candidate_id"],
        "candidates": [candidate],
    }
    record["archive_sha256"] = _canonical_sha256(record)
    return record


def _terrain() -> TerrainMap:
    return cast(
        TerrainMap,
        SimpleNamespace(
            x_m=np.asarray([-3000.0, 4000.0]),
            y_m=np.asarray([-1000.0, 5000.0]),
        ),
    )


def test_frozen_atlas_validation_rejects_truth_or_content_changes() -> None:
    atlas = _atlas()
    validated = validate_frozen_cyltan_atlas(atlas)

    assert validated == atlas
    assert validated is not atlas
    assert validated["candidates"] is not atlas["candidates"]
    assert validated["candidates"][0] is not atlas["candidates"][0]

    truth_used = _atlas()
    truth_used["numeric_evaluation_reference_used"] = True
    truth_used.pop("archive_sha256")
    truth_used["archive_sha256"] = _canonical_sha256(truth_used)
    with pytest.raises(ValueError, match="estimator-only"):
        validate_frozen_cyltan_atlas(truth_used)

    tampered = _atlas()
    tampered["candidates"][0]["vertical_shift_px"] = 99.0
    with pytest.raises(ValueError, match="SHA-256"):
        validate_frozen_cyltan_atlas(tampered)


def test_exact_cyltan_render_forwards_pose_nuisances_and_returns_ray_range() -> None:
    calls: list[dict[str, Any]] = []
    elevation_rad = math.radians(60.0)
    patch = cast(Patch, object())

    def renderer(
        _terrain,
        camera_up,
        azimuths,
        width,
        height,
        fov,
        vertical_shift,
        east,
        north,
        roll,
        *,
        sub,
        patch,
    ):
        calls.append(
            {
                "camera_up": camera_up,
                "azimuths": azimuths,
                "width": width,
                "height": height,
                "fov": fov,
                "vertical_shift": vertical_shift,
                "east": east,
                "north": north,
                "roll": roll,
                "sub": sub,
                "patch": patch,
            }
        )
        shape = (3, 4)
        return (
            np.full(shape, 500.0),
            np.zeros(shape, dtype=int),
            np.full(shape, elevation_rad),
            np.zeros((shape[1], 2)),
            np.asarray([25.0, 8750.0]),
            np.arange(shape[0]),
        )

    rendered = render_cyltan_candidate_depth(
        _terrain(),
        patch,
        _candidate(),
        16,
        12,
        60.0,
        subsample=4,
        depth_renderer=renderer,
    )

    assert rendered.candidate_ray_depth == pytest.approx(np.full((3, 4), 1000.0))
    assert rendered.max_range_m == 8750.0
    assert rendered.atlas_candidate == _candidate()
    assert calls[0]["camera_up"] == 2030.0
    assert calls[0]["azimuths"][7:9] == pytest.approx([40.125, 43.875])
    assert calls[0] | {"azimuths": None} == {
        "camera_up": 2030.0,
        "azimuths": None,
        "width": 16,
        "height": 12,
        "fov": 60.0,
        "vertical_shift": -4.0,
        "east": 12.0,
        "north": -8.0,
        "roll": 1.5,
        "sub": 4,
        "patch": patch,
    }


def test_terrain_diagonal_is_a_stable_full_extent_cap() -> None:
    assert terrain_diagonal_range_m(_terrain()) == pytest.approx(math.hypot(7000.0, 6000.0))

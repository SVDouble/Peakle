from __future__ import annotations

import numpy as np
import pytest

from peakle.localize.compatibility import gt_dem_compatibility, raw_camera_clearance_compatibility


def _ridge(width: int = 640) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, width)
    return 220.0 - 90.0 * np.exp(-(((x + 0.28) / 0.13) ** 2)) - 55.0 * np.exp(-(((x - 0.37) / 0.21) ** 2))


def test_compatibility_is_invariant_to_global_crop_shift() -> None:
    source = _ridge() + 73.0
    metric = gt_dem_compatibility(
        source,
        _ridge(),
        width_px=len(source),
        height_px=480,
        horizontal_fov_deg=60.0,
        yaw_deg=120.0,
    )

    assert metric.tier == "MAP_A"
    assert metric.median_deg < 1e-6
    assert metric.p90_deg < 1e-6
    assert metric.crop_shift_px == pytest.approx(73.0)


def test_compatibility_rejects_wrong_horizontal_alignment() -> None:
    source = _ridge()
    wrong = np.roll(source, 110)
    metric = gt_dem_compatibility(
        source,
        wrong,
        width_px=len(source),
        height_px=480,
        horizontal_fov_deg=60.0,
        yaw_deg=120.0,
    )

    assert metric.tier == "MAP_C"
    assert metric.p90_deg > 1.5


def test_compatibility_fails_closed_on_low_coverage() -> None:
    source = np.full(100, np.nan)
    dem = np.full(100, np.nan)
    source[:5] = 40.0
    dem[:5] = 40.0

    metric = gt_dem_compatibility(
        source,
        dem,
        width_px=100,
        height_px=80,
        horizontal_fov_deg=45.0,
        yaw_deg=0.0,
    )

    assert metric.tier == "MAP_C"
    assert metric.coverage == pytest.approx(0.05)


def test_raw_camera_clearance_is_a_separate_physical_gate() -> None:
    assert raw_camera_clearance_compatibility(1502.0, 1500.0)["tier"] == "HEIGHT_A"
    assert raw_camera_clearance_compatibility(1495.0, 1500.0)["tier"] == "HEIGHT_B"
    result = raw_camera_clearance_compatibility(1400.0, 1500.0)
    assert result["tier"] == "HEIGHT_C"
    assert result["physically_plausible"] is False

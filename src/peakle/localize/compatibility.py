"""Independent source-depth ↔ DEM compatibility metrics for pose benchmarks.

This module deliberately knows nothing about GT-v2 refinement, photo extraction, or solver
outputs. It answers one question before localization begins: can the selected terrain model
reproduce the dataset's source-depth horizon at the unmodified metadata pose?
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from peakle.domain.projection import ProjectionName, azimuths_deg, elevation_rad_from_rows

COMPATIBILITY_POLICY = "gt_dem_compat_v1"
HEIGHT_COMPATIBILITY_POLICY = "raw_camera_clearance_v1"


@dataclass(frozen=True)
class GtDemCompatibility:
    """Fixed-pose skyline agreement after cross-fitted vertical crop alignment."""

    policy: str
    tier: str
    chamfer_deg: float
    median_deg: float
    p90_deg: float
    coverage: float
    crop_shift_px: float
    fold_shift_disagreement_px: float
    within_025_deg: float
    within_050_deg: float
    within_100_deg: float

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


def gt_dem_compatibility(
    source_rows: np.ndarray,
    dem_rows: np.ndarray,
    *,
    width_px: int,
    height_px: int,
    horizontal_fov_deg: float,
    yaw_deg: float,
    projection: ProjectionName = "cyltan",
    block_width: int = 32,
    trim_quantile: float = 0.90,
) -> GtDemCompatibility:
    """Compare source-depth and DEM skyline geometry without changing the pose.

    GeoPose cylindrical crops carry an unknown global vertical crop offset. To avoid fitting and
    grading that nuisance on the same columns, alternating column blocks form two folds: each
    fold's shift is estimated on the other fold and applied only to held-out columns. Position,
    yaw, FOV, and terrain remain fixed.
    """

    source = np.asarray(source_rows, dtype=float)
    dem = np.asarray(dem_rows, dtype=float)
    if source.shape != (width_px,) or dem.shape != (width_px,):
        msg = f"expected two ({width_px},) skyline profiles, got {source.shape} and {dem.shape}"
        raise ValueError(msg)
    if block_width < 1:
        raise ValueError("block_width must be positive")
    valid = np.isfinite(source) & np.isfinite(dem)
    coverage = float(valid.mean())
    if valid.sum() < max(8, round(0.1 * width_px)):
        return GtDemCompatibility(
            policy=COMPATIBILITY_POLICY,
            tier="MAP_C",
            chamfer_deg=float("inf"),
            median_deg=float("inf"),
            p90_deg=float("inf"),
            coverage=coverage,
            crop_shift_px=float("nan"),
            fold_shift_disagreement_px=float("nan"),
            within_025_deg=0.0,
            within_050_deg=0.0,
            within_100_deg=0.0,
        )

    blocks = np.arange(width_px) // block_width
    folds = blocks % 2
    aligned_dem = dem.copy()
    fold_shifts: list[float] = []
    for held_out in (0, 1):
        training = valid & (folds != held_out)
        if training.sum() < 4:
            training = valid
        shift = float(np.median(source[training] - dem[training]))
        fold_shifts.append(shift)
        aligned_dem[folds == held_out] += shift

    source_el = np.degrees(elevation_rad_from_rows(source, width_px, height_px, horizontal_fov_deg, projection))
    dem_el = np.degrees(elevation_rad_from_rows(aligned_dem, width_px, height_px, horizontal_fov_deg, projection))
    residual = np.abs(source_el[valid] - dem_el[valid])
    azimuth = azimuths_deg(width_px, horizontal_fov_deg, yaw_deg, projection)
    source_curve = np.column_stack((azimuth[valid], source_el[valid]))
    dem_curve = np.column_stack((azimuth[valid], dem_el[valid]))
    directed_a = _directed_curve_distances(source_curve, dem_curve)
    directed_b = _directed_curve_distances(dem_curve, source_curve)
    symmetric = np.concatenate((directed_a, directed_b))
    trim_at = float(np.quantile(symmetric, np.clip(trim_quantile, 0.5, 1.0)))
    trimmed = symmetric[symmetric <= trim_at]
    chamfer = float(np.mean(trimmed)) if trimmed.size else float("inf")
    median = float(np.median(residual))
    p90 = float(np.percentile(residual, 90.0))
    tier = compatibility_tier(median, p90, coverage)
    return GtDemCompatibility(
        policy=COMPATIBILITY_POLICY,
        tier=tier,
        chamfer_deg=round(chamfer, 5),
        median_deg=round(median, 5),
        p90_deg=round(p90, 5),
        coverage=round(coverage, 5),
        crop_shift_px=round(float(np.mean(fold_shifts)), 3),
        fold_shift_disagreement_px=round(abs(fold_shifts[0] - fold_shifts[1]), 3),
        within_025_deg=round(float(np.mean(residual <= 0.25)), 5),
        within_050_deg=round(float(np.mean(residual <= 0.50)), 5),
        within_100_deg=round(float(np.mean(residual <= 1.00)), 5),
    )


def compatibility_tier(median_deg: float, p90_deg: float, coverage: float) -> str:
    """Initial transparent tiers; calibrate once on the MANUAL corpus, then version changes."""

    if median_deg <= 0.25 and p90_deg <= 0.75 and coverage >= 0.90:
        return "MAP_A"
    if median_deg <= 0.50 and p90_deg <= 1.50 and coverage >= 0.80:
        return "MAP_B"
    return "MAP_C"


def raw_camera_clearance_compatibility(
    camera_elevation_m: float,
    ground_elevation_m: float,
) -> dict[str, str | float | bool]:
    """Independent, provisional altitude/datum check at the raw metadata position.

    Skyline agreement can look excellent after fitting a vertical crop offset even when the
    metadata camera lies below the selected DEM. Keep this physical check separate from the
    outline-based MAP tier so neither metric can hide the other's failure.
    """

    clearance_m = float(camera_elevation_m - ground_elevation_m)
    if -2.0 <= clearance_m <= 10.0:
        tier = "HEIGHT_A"
    elif -10.0 <= clearance_m <= 50.0:
        tier = "HEIGHT_B"
    else:
        tier = "HEIGHT_C"
    return {
        "policy": HEIGHT_COMPATIBILITY_POLICY,
        "tier": tier,
        "raw_camera_clearance_m": round(clearance_m, 3),
        "physically_plausible": tier in {"HEIGHT_A", "HEIGHT_B"},
        "thresholds_provisional": True,
    }


def _directed_curve_distances(source: np.ndarray, target: np.ndarray, block_size: int = 256) -> np.ndarray:
    """Nearest angular 2-D distance for each source point, in bounded-memory blocks."""

    distances = np.empty(len(source), dtype=float)
    for start in range(0, len(source), block_size):
        block = source[start : start + block_size]
        delta = block[:, None, :] - target[None, :, :]
        distances[start : start + len(block)] = np.sqrt(np.min(np.sum(delta * delta, axis=2), axis=1))
    return distances

"""Pose-recovery benchmark across map sizes and shrinking prior uncertainty.

Measures how accurately and how fast a strategy recovers a camera pose from its
skyline outline as the prior widens from tight down to none, on maps of growing
physical size. This is the harness behind questions like "how well can we localize
by an outline on a 50 km x 50 km map without a prior, and how fast?".
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from peakle.config import AppSettings, load_settings
from peakle.domain.pose import PosePrior
from peakle.optimization.solve import solve_pose
from peakle.scene.scene import Scene
from peakle.scene.state import build_intrinsics, noisy_prior
from peakle.terrain.generator import TerrainGenerator
from peakle.terrain.peak_detection import PeakDetector

SUCCESS_RADIUS_M = 500.0


@dataclass(frozen=True)
class PriorLevel:
    """One rung of the prior-uncertainty sweep.

    Attributes:
        label: Human-readable level name.
        sigma_scale: Multiplier on the base pose-prior sigmas/noise, or None for
            no position prior at all.
        strategy: Solver strategy to use at this level.
    """

    label: str
    sigma_scale: float | None
    strategy: str


@dataclass(frozen=True)
class BenchmarkRow:
    """Aggregated results for one (map size, prior level) cell."""

    map_km: float
    level: str
    strategy: str
    views: int
    pos_err_med_m: float
    yaw_err_med_deg: float
    mae_med_px: float
    time_med_s: float
    success_rate: float


DEFAULT_LEVELS = (
    PriorLevel("prior x1", 1.0, "powell"),
    PriorLevel("prior x4", 4.0, "powell"),
    PriorLevel("prior x16", 16.0, "powell"),
    PriorLevel("no prior", None, "global"),
)


def run_benchmark(
    map_sizes_km: tuple[float, ...] = (14.0, 50.0),
    views_per_map: int = 4,
    levels: tuple[PriorLevel, ...] = DEFAULT_LEVELS,
    settings: AppSettings | None = None,
    seed: int = 7,
) -> list[BenchmarkRow]:
    """Runs the sweep and returns aggregated rows."""

    settings = settings or load_settings()
    rows: list[BenchmarkRow] = []
    for map_km in map_sizes_km:
        scene = _scene_for_size(settings, map_km, seed)
        views = _make_views(scene, views_per_map, map_km)
        for level in levels:
            samples = [_solve_view(scene, view, level, settings, seed) for view in views]
            rows.append(_aggregate(map_km, level, samples))
    return rows


def _scene_for_size(settings: AppSettings, map_km: float, seed: int) -> Scene:
    scene = Scene.from_settings(settings)
    spec = settings.terrain.model_copy(update={"width_m": map_km * 1000.0, "height_m": map_km * 1000.0, "seed": seed})
    scene._terrain_spec = spec
    scene.terrain = TerrainGenerator(spec).generate()
    scene.peaks = PeakDetector(settings.peak_detection).detect(scene.terrain)
    scene.intrinsics = build_intrinsics(
        settings.render.image_width, settings.render.image_height, settings.render.horizontal_fov_deg
    )
    return scene


def _make_views(scene: Scene, count: int, map_km: float) -> list[object]:
    """Places `count` cameras at varied spots, each looking toward a tall peak."""

    peaks = sorted(scene.peaks, key=lambda peak: peak.elevation_m, reverse=True)
    extent = map_km * 1000.0
    east_lo, east_hi = scene.terrain.x_m[0] + 500, scene.terrain.x_m[-1] - 500
    north_lo, north_hi = scene.terrain.y_m[0] + 500, scene.terrain.y_m[-1] - 500
    views = []
    for index in range(count):
        target = peaks[index % len(peaks)].local_position
        east = float(np.clip(target.east_m - extent * (0.10 + 0.04 * index), east_lo, east_hi))
        north = float(np.clip(scene.terrain.y_m[0] + extent * (0.05 + 0.03 * index), north_lo, north_hi))
        yaw = float(np.degrees(np.arctan2(target.east_m - east, target.north_m - north)))
        views.append(scene.create_view(east_m=east, north_m=north, yaw_deg=yaw, pitch_deg=3.0, eye_height_m=120.0))
    return views


def _solve_view(
    scene: Scene, view: object, level: PriorLevel, settings: AppSettings, seed: int
) -> tuple[float, float, float, float]:
    truth = view.true_extrinsics  # type: ignore[attr-defined]
    prior = _scaled_prior(settings, truth, level, seed, view.id)  # type: ignore[attr-defined]
    use_position_prior = level.sigma_scale is not None
    start = time.perf_counter()
    result = solve_pose(
        terrain=scene.terrain,
        contour=view.contour,  # type: ignore[attr-defined]
        intrinsics=scene.intrinsics,
        prior=prior,
        strategy=level.strategy,
        terrain_stride=settings.optimization.objective_terrain_stride,
        truth=truth,
        use_position_prior=use_position_prior,
    )
    elapsed = time.perf_counter() - start
    metrics = result.estimate.metrics
    return (
        metrics.position_error_m or float("nan"),
        metrics.yaw_error_deg or float("nan"),
        metrics.contour_mae_px,
        elapsed,
    )


def _scaled_prior(settings: AppSettings, truth: object, level: PriorLevel, seed: int, view_id: str) -> PosePrior:
    scale = level.sigma_scale if level.sigma_scale is not None else 16.0
    noise = settings.pose_noise.model_copy(
        update={
            "horizontal_noise_m": settings.pose_noise.horizontal_noise_m * scale,
            "horizontal_sigma_m": settings.pose_noise.horizontal_sigma_m * scale,
        }
    )
    rng = np.random.default_rng(abs(hash((seed, view_id, level.label))) % (2**32))
    return noisy_prior(noise, truth.position, truth.yaw_deg, truth.pitch_deg, rng)  # type: ignore[attr-defined]


def _aggregate(map_km: float, level: PriorLevel, samples: list[tuple[float, float, float, float]]) -> BenchmarkRow:
    pos = np.array([s[0] for s in samples])
    yaw = np.array([s[1] for s in samples])
    mae = np.array([s[2] for s in samples])
    times = np.array([s[3] for s in samples])
    return BenchmarkRow(
        map_km=map_km,
        level=level.label,
        strategy=level.strategy,
        views=len(samples),
        pos_err_med_m=float(np.nanmedian(pos)),
        yaw_err_med_deg=float(np.nanmedian(yaw)),
        mae_med_px=float(np.nanmedian(mae)),
        time_med_s=float(np.nanmedian(times)),
        success_rate=float(np.mean(pos < SUCCESS_RADIUS_M)),
    )


def format_table(rows: list[BenchmarkRow]) -> str:
    """Renders benchmark rows as a fixed-width table."""

    columns = f"{'map':>6} {'prior level':<11} {'strategy':<9} {'n':>2} "
    header = columns + f"{'pos_err':>9} {'yaw_err':>8} {'mae':>7} {'time':>7} {'success':>8}"
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row.map_km:>5.0f}k {row.level:<11} {row.strategy:<9} {row.views:>2} "
            f"{row.pos_err_med_m:>8.0f}m {row.yaw_err_med_deg:>7.1f}° {row.mae_med_px:>6.1f}px "
            f"{row.time_med_s:>6.1f}s {row.success_rate * 100:>6.0f}%"
        )
    return "\n".join(lines)

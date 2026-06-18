"""Pluggable full-pose solving with a convergence trace.

Every strategy minimizes the shared `PoseObjective` (5-DOF pose with a prior
penalty and a Huber contour loss) and records a best-so-far trace so the
frontend can animate convergence without running any solver itself. The
objective renders at a reduced resolution for interactivity; final fit metrics
are recomputed at full resolution.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
from scipy.optimize import differential_evolution, minimize

from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics
from peakle.domain.contours import SkylineContour
from peakle.domain.pose import FitMetrics, PoseEstimate, PosePrior
from peakle.domain.terrain import TerrainMap
from peakle.optimization.objective import PoseObjective
from peakle.optimization.pose_search import add_synthetic_truth_metrics
from peakle.optimization.scoring import residual_summary, robust_contour_residuals
from peakle.rendering.rasterizer import SyntheticRenderer

SOLVE_SAMPLE_WIDTH = 360
SOLVE_TARGET_COLUMNS = 22
MAX_TRACE_FRAMES = 60
STRATEGIES = ("powell", "nelder", "evolution")

AVAILABLE_STRATEGIES = [
    {
        "name": "powell",
        "label": "Powell (local)",
        "blurb": "Coarse pose scan around the prior, then Powell direction-set refinement.",
    },
    {
        "name": "nelder",
        "label": "Nelder-Mead (local)",
        "blurb": "Derivative-free simplex refining the full pose from the prior.",
    },
    {
        "name": "evolution",
        "label": "Differential evolution",
        "blurb": "Population-based search across the bounded pose box around the prior.",
    },
]


class SolveFrame(BaseModel):
    """One recorded step of a solver's convergence.

    Attributes:
        east_m: Best camera easting so far.
        north_m: Best camera northing so far.
        up_m: Best camera height so far.
        yaw_deg: Best heading so far.
        pitch_deg: Best pitch so far.
        score: Objective value at the best pose so far.
        evaluations: Cumulative objective evaluations.
        profile: Predicted skyline profile in sample rows (null for empty columns).
    """

    east_m: float
    north_m: float
    up_m: float
    yaw_deg: float
    pitch_deg: float
    score: float
    evaluations: int
    profile: list[float | None]


class PoseSolveResult(BaseModel):
    """A solved pose plus the convergence trace and the observed outline.

    Attributes:
        strategy: Strategy key used.
        estimate: Estimated pose and fit diagnostics.
        evaluations: Total objective evaluations.
        sample_width: Reduced profile width in columns.
        sample_height: Reduced profile height in rows.
        observed_profile: Observed outline in sample rows (null for empty columns).
        trace: Ordered convergence frames.
    """

    strategy: str
    estimate: PoseEstimate
    evaluations: int
    sample_width: int
    sample_height: int
    observed_profile: list[float | None]
    trace: list[SolveFrame]


def solve_pose(
    terrain: TerrainMap,
    contour: SkylineContour,
    intrinsics: CameraIntrinsics,
    prior: PosePrior,
    strategy: str,
    terrain_stride: int = 6,
    truth: CameraExtrinsics | None = None,
    seed: int | None = None,
) -> PoseSolveResult:
    """Recovers a full camera pose from an observed contour and a prior.

    Args:
        terrain: Terrain model used to render candidate skylines.
        contour: Observed skyline contour at full image resolution.
        intrinsics: Known camera intrinsics.
        prior: Noisy pose prior.
        strategy: One of `powell`, `nelder`, or `evolution`.
        terrain_stride: Terrain subsampling stride for the objective.
        truth: Optional ground-truth pose for error metrics.
        seed: Optional seed for stochastic strategies.

    Returns:
        The estimated pose, fit metrics, and a convergence trace.
    """

    if strategy not in STRATEGIES:
        msg = f"unknown strategy {strategy!r}; expected one of {STRATEGIES}"
        raise ValueError(msg)

    renderer = SyntheticRenderer()
    reduced = _reduced_intrinsics(intrinsics, SOLVE_SAMPLE_WIDTH)
    observed_reduced = _reduce_profile(contour.to_profile(), intrinsics, reduced)
    # Coarsen the terrain on large grids so a full solve stays interactive; the
    # nice full-resolution view image and final metrics are unaffected.
    working_stride = max(terrain_stride, round(terrain.spec.grid_width / SOLVE_TARGET_COLUMNS))
    objective = PoseObjective(
        terrain=terrain,
        observed_profile=observed_reduced,
        intrinsics=reduced,
        prior=prior,
        renderer=renderer,
        terrain_stride=working_stride,
    )
    traced = _TracedSolve(objective, terrain, renderer, reduced, working_stride)

    if strategy == "powell":
        _run_powell(traced)
    elif strategy == "nelder":
        _run_nelder_mead(traced)
    else:
        _run_differential_evolution(traced, seed)
    if not traced.frames:
        traced.record()

    estimate = _final_estimate(traced, terrain, intrinsics, contour, renderer, terrain_stride)
    if truth is not None:
        estimate = add_synthetic_truth_metrics(estimate, truth)

    return PoseSolveResult(
        strategy=strategy,
        estimate=estimate,
        evaluations=traced.evaluations,
        sample_width=reduced.width_px,
        sample_height=reduced.height_px,
        observed_profile=_jsonable_profile(observed_reduced),
        trace=_subsample_frames(traced.frames, MAX_TRACE_FRAMES),
    )


class _TracedSolve:
    """Wraps a `PoseObjective` with eval counting and best-so-far trace frames."""

    def __init__(
        self,
        objective: PoseObjective,
        terrain: TerrainMap,
        renderer: SyntheticRenderer,
        reduced: CameraIntrinsics,
        terrain_stride: int,
    ) -> None:
        self.objective = objective
        self.terrain = terrain
        self.renderer = renderer
        self.reduced = reduced
        self.terrain_stride = terrain_stride
        self.evaluations = 0
        self.best_score = float("inf")
        self.best_theta = objective.theta_from_prior()
        self.frames: list[SolveFrame] = []

    def cost(self, theta: NDArray[np.float64]) -> float:
        self.evaluations += 1
        value = self.objective.score(np.asarray(theta, dtype=np.float64))
        if value < self.best_score:
            self.best_score = value
            self.best_theta = np.asarray(theta, dtype=np.float64).copy()
        return value

    def record(self, *_args: object) -> None:
        extrinsics = self.objective.extrinsics_from_theta(self.best_theta)
        profile = self.renderer.skyline_profile(self.terrain, self.reduced, extrinsics, stride=self.terrain_stride)
        self.frames.append(
            SolveFrame(
                east_m=extrinsics.position.east_m,
                north_m=extrinsics.position.north_m,
                up_m=extrinsics.position.up_m,
                yaw_deg=extrinsics.yaw_deg,
                pitch_deg=extrinsics.pitch_deg,
                score=self.best_score,
                evaluations=self.evaluations,
                profile=_jsonable_profile(profile),
            )
        )


def _run_powell(traced: _TracedSolve) -> None:
    _coarse_pose_search(traced)
    minimize(
        traced.cost,
        traced.best_theta,
        method="Powell",
        bounds=traced.objective.bounds(),
        callback=traced.record,
        options={"maxiter": 80, "xtol": 1e-2, "ftol": 1e-2, "disp": False},
    )


def _run_nelder_mead(traced: _TracedSolve) -> None:
    # Seed a simplex scaled to each parameter's prior uncertainty; scipy's default
    # 5% relative step is wildly mis-scaled for metre-valued position coordinates.
    prior = traced.objective.prior
    theta0 = traced.objective.theta_from_prior()
    steps = np.array(
        [
            prior.horizontal_sigma_m,
            prior.horizontal_sigma_m,
            prior.vertical_sigma_m,
            prior.yaw_sigma_deg,
            prior.pitch_sigma_deg,
        ],
        dtype=np.float64,
    )
    simplex = np.vstack([theta0, *(theta0 + np.diag(steps))])
    minimize(
        traced.cost,
        theta0,
        method="Nelder-Mead",
        bounds=traced.objective.bounds(),
        callback=traced.record,
        options={"maxiter": 300, "xatol": 1e-2, "fatol": 1e-2, "initial_simplex": simplex, "disp": False},
    )


def _run_differential_evolution(traced: _TracedSolve, seed: int | None) -> None:
    differential_evolution(
        traced.cost,
        bounds=traced.objective.bounds(),
        callback=lambda *_args, **_kwargs: traced.record(),
        maxiter=20,
        popsize=6,
        tol=1e-3,
        seed=seed,
        polish=False,
    )


def _coarse_pose_search(traced: _TracedSolve) -> None:
    """Bounded coarse scan over orientation and horizontal position.

    Mirrors `PoseOptimizer._coarse_orientation_search`, recording a frame every
    few cells so the trace shows the global scan converging.
    """

    prior = traced.objective.prior
    start = traced.objective.theta_from_prior()
    east_values = _offset_values(prior.position.east_m, prior.horizontal_sigma_m)
    north_values = _offset_values(prior.position.north_m, prior.horizontal_sigma_m)
    yaw_values = np.linspace(prior.yaw_deg - prior.yaw_sigma_deg * 1.8, prior.yaw_deg + prior.yaw_sigma_deg * 1.8, 11)
    pitch_values = np.linspace(
        prior.pitch_deg - prior.pitch_sigma_deg * 1.8,
        prior.pitch_deg + prior.pitch_sigma_deg * 1.8,
        7,
    )
    count = 0
    for east_m in east_values:
        for north_m in north_values:
            for yaw_deg in yaw_values:
                for pitch_deg in pitch_values:
                    theta = start.copy()
                    theta[0] = east_m
                    theta[1] = north_m
                    theta[3] = yaw_deg
                    theta[4] = pitch_deg
                    traced.cost(theta)
                    count += 1
                    if count % 32 == 0:
                        traced.record()
    traced.record()


def _final_estimate(
    traced: _TracedSolve,
    terrain: TerrainMap,
    intrinsics: CameraIntrinsics,
    contour: SkylineContour,
    renderer: SyntheticRenderer,
    terrain_stride: int,
) -> PoseEstimate:
    """Builds the final pose estimate with full-resolution fit metrics."""

    extrinsics = traced.objective.extrinsics_from_theta(traced.best_theta)
    predicted = renderer.skyline_profile(terrain, intrinsics, extrinsics, stride=1)
    residuals = robust_contour_residuals(predicted, contour.to_profile())
    mae, p95, valid_columns = residual_summary(residuals)
    metrics = FitMetrics(
        score=float(traced.best_score),
        contour_mae_px=mae,
        contour_p95_px=p95,
        valid_columns=valid_columns,
        iterations=len(traced.frames),
        success=bool(np.isfinite(traced.best_score)),
        message=f"{len(traced.frames)} recorded steps",
        position_error_m=None,
        yaw_error_deg=None,
        pitch_error_deg=None,
    )
    return PoseEstimate(extrinsics=extrinsics, metrics=metrics)


def _reduced_intrinsics(intrinsics: CameraIntrinsics, target_width: int) -> CameraIntrinsics:
    """Scales intrinsics to a reduced render width while preserving FOV."""

    if intrinsics.width_px <= target_width:
        return intrinsics
    scale = target_width / intrinsics.width_px
    return CameraIntrinsics(
        width_px=max(1, round(intrinsics.width_px * scale)),
        height_px=max(1, round(intrinsics.height_px * scale)),
        focal_length_px=intrinsics.focal_length_px * scale,
        principal_x_px=intrinsics.principal_x_px * scale,
        principal_y_px=intrinsics.principal_y_px * scale,
    )


def _reduce_profile(
    profile: NDArray[np.float64],
    full: CameraIntrinsics,
    reduced: CameraIntrinsics,
) -> NDArray[np.float64]:
    """Resamples a full-resolution profile to the reduced render scale."""

    columns = np.arange(profile.size, dtype=np.float64)
    targets = np.linspace(0.0, profile.size - 1, reduced.width_px)
    resampled = np.interp(targets, columns, profile)
    return resampled * (reduced.height_px / full.height_px)


def _subsample_frames(frames: list[SolveFrame], limit: int) -> list[SolveFrame]:
    """Evenly subsamples frames to at most `limit`, always keeping the last."""

    if len(frames) <= limit:
        return frames
    indices = np.linspace(0, len(frames) - 1, limit).round().astype(int)
    indices[-1] = len(frames) - 1
    seen: set[int] = set()
    kept: list[SolveFrame] = []
    for index in indices:
        position = int(index)
        if position not in seen:
            seen.add(position)
            kept.append(frames[position])
    return kept


def _jsonable_profile(profile: NDArray[np.float64]) -> list[float | None]:
    """Converts a profile to JSON, mapping non-finite columns to null."""

    return [float(value) if np.isfinite(value) else None for value in profile]


def _offset_values(center: float, sigma: float) -> list[float]:
    return [center - 0.75 * sigma, center, center + 0.75 * sigma]

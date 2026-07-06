"""Pluggable full-pose solving with a convergence trace.

Every strategy minimizes the shared `PoseObjective` (5-DOF pose with a prior
penalty and a Huber contour loss) and records a best-so-far trace so the
frontend can animate convergence without running any solver itself. The
objective renders at a reduced resolution for interactivity; final fit metrics
are recomputed at full resolution.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
from scipy.optimize import differential_evolution, minimize

from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics
from peakle.domain.contours import SkylineContour
from peakle.domain.pose import FitMetrics, PoseEstimate, PosePrior
from peakle.domain.terrain import TerrainMap
from peakle.optimization.horizon import horizon_seeds
from peakle.optimization.objective import PoseObjective
from peakle.optimization.pose_search import add_synthetic_truth_metrics
from peakle.optimization.scoring import match_confidence, residual_summary, robust_contour_residuals
from peakle.rendering.rasterizer import SyntheticRenderer

SOLVE_SAMPLE_WIDTH = 360
SOLVE_TARGET_COLUMNS = 22
MAX_TRACE_FRAMES = 60
STRATEGIES = ("powell", "nelder", "evolution", "global", "horizon")

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
    {
        "name": "global",
        "label": "Global (no prior)",
        "blurb": "Prior-free: 360 deg horizon correlation across the whole map for position + yaw, then local refine.",
    },
    {
        "name": "horizon",
        "label": "Horizon (validated)",
        "blurb": "The ray-cast 360 deg horizon solver validated on GeoPose3K: median-centered "
                 "shift chamfer with top-K basin polish and honesty diagnostics "
                 "(alias ratio, SNR, verdict). Solves yaw/pitch at the prior position.",
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


class PoseCandidate(BaseModel):
    """One plausible pose from a prior-free search.

    Attributes:
        extrinsics: Candidate camera pose.
        score: Objective value (lower is a better skyline fit).
    """

    extrinsics: CameraExtrinsics
    score: float


class PoseSolveResult(BaseModel):
    """A solved pose plus the convergence trace and the observed outline.

    Attributes:
        strategy: Strategy key used.
        estimate: Estimated pose and fit diagnostics.
        evaluations: Total objective evaluations.
        sample_width: Reduced profile width in columns.
        sample_height: Reduced profile height in rows.
        observed_profile: Observed outline in sample rows (null for empty columns).
        predicted_profile: Final predicted outline at the reduced sample width,
            for the convergence plot; the per-frame trace profiles stay coarse.
        predicted_profile_full: Final predicted outline at full image width, for a
            crisp overlay on the rendered photo (length = image width).
        candidates: Distinct plausible poses (best first). For a prior-free search
            of an ambiguous outline this holds every well-separated location that
            fits; for prior-based strategies it is just the single estimate.
        trace: Ordered convergence frames.
    """

    strategy: str
    estimate: PoseEstimate
    evaluations: int
    sample_width: int
    sample_height: int
    observed_profile: list[float | None]
    predicted_profile: list[float | None]
    predicted_profile_full: list[float | None]
    candidates: list[PoseCandidate]
    trace: list[SolveFrame]
    diagnostics: dict[str, Any] | None = None


def solve_pose(
    terrain: TerrainMap,
    contour: SkylineContour,
    intrinsics: CameraIntrinsics,
    prior: PosePrior,
    strategy: str,
    terrain_stride: int = 6,
    truth: CameraExtrinsics | None = None,
    seed: int | None = None,
    use_position_prior: bool = True,
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

    if strategy == "horizon":
        return _solve_horizon(terrain, contour, intrinsics, prior, truth)

    renderer = SyntheticRenderer()
    reduced = _reduced_intrinsics(intrinsics, SOLVE_SAMPLE_WIDTH)
    observed_reduced = _reduce_profile(contour.to_profile(), intrinsics, reduced)
    # The global strategy is prior-free by definition: drop both the position and
    # orientation priors so the search spans the whole map and full yaw circle.
    if strategy == "global":
        use_position_prior = False
    # The objective uses the fast upsampled point-projection skyline (set up
    # inside PoseObjective); the trace display still renders at a coarse stride
    # for cheap animation frames.
    working_stride = max(terrain_stride, round(terrain.spec.grid_width / SOLVE_TARGET_COLUMNS))
    objective = PoseObjective(
        terrain=terrain,
        observed_profile=observed_reduced,
        intrinsics=reduced,
        prior=prior,
        renderer=renderer,
        terrain_stride=working_stride,
        use_position_prior=use_position_prior,
        use_orientation_prior=strategy != "global",
    )
    traced = _TracedSolve(objective, terrain, renderer, reduced, working_stride)

    if strategy == "powell":
        _run_powell(traced)
    elif strategy == "nelder":
        _run_nelder_mead(traced)
    elif strategy == "global":
        _run_global(traced)
    else:
        _run_differential_evolution(traced, seed)
    if not traced.frames:
        traced.record()

    estimate = _final_estimate(traced, terrain, intrinsics, contour, renderer, terrain_stride)
    if truth is not None:
        estimate = add_synthetic_truth_metrics(estimate, truth)

    # Reduced predicted outline (accurate triangle skyline) drives the plot and
    # the match metric; the crisp full-width overlay uses the fast skyline.
    predicted_reduced = renderer.skyline_profile(terrain, reduced, estimate.extrinsics, stride=1)
    predicted_full = renderer.fast_skyline(objective._points, intrinsics, estimate.extrinsics)

    if traced.candidates:
        candidates = [
            PoseCandidate(extrinsics=objective.extrinsics_from_theta(theta), score=score)
            for score, theta in traced.candidates
        ]
    else:
        candidates = [PoseCandidate(extrinsics=estimate.extrinsics, score=float(traced.best_score))]

    return PoseSolveResult(
        strategy=strategy,
        estimate=estimate,
        evaluations=traced.evaluations,
        sample_width=reduced.width_px,
        sample_height=reduced.height_px,
        observed_profile=_jsonable_profile(observed_reduced),
        predicted_profile=_jsonable_profile(predicted_reduced),
        predicted_profile_full=_jsonable_profile(predicted_full),
        candidates=candidates,
        trace=_subsample_frames(traced.frames, MAX_TRACE_FRAMES),
    )




def _solve_horizon(
    terrain: TerrainMap,
    contour: SkylineContour,
    intrinsics: CameraIntrinsics,
    prior: PosePrior,
    truth: CameraExtrinsics | None,
) -> PoseSolveResult:
    """The validated peakle.localize horizon solver, adapted to the workbench.

    Orientation-only by design: the 360° elevation profile is computed once at the
    prior position and every (yaw, pitch) hypothesis is a resampling of it, scored
    with the median-centered shift chamfer + top-K basin polish. Its honesty
    diagnostics (alias ratio, SNR, verdict) ride along in ``diagnostics``.
    """

    from peakle.localize.solve import HorizonProfile, solve_orientation

    position = prior.position
    width, height = intrinsics.width_px, intrinsics.height_px
    hfov_deg = math.degrees(2.0 * math.atan(width / (2.0 * intrinsics.focal_length_px)))
    obs = contour.to_profile()

    grid_step = float(min(terrain.x_m[1] - terrain.x_m[0], terrain.y_m[1] - terrain.y_m[0]))
    profile = HorizonProfile(
        terrain, position.up_m, step=max(grid_step / 2.0, 10.0),
        cam_e=position.east_m, cam_n=position.north_m,
    )
    solve = solve_orientation(obs, height, profile, fov_deg=hfov_deg, projection="pinhole")

    def extrinsics_at(yaw_deg: float, pitch_deg: float) -> CameraExtrinsics:
        return CameraExtrinsics(
            position=position, yaw_deg=((yaw_deg + 180.0) % 360.0) - 180.0, pitch_deg=pitch_deg, roll_deg=0.0
        )

    estimate_extrinsics = extrinsics_at(solve.yaw_deg, solve.pitch_deg)
    metrics = FitMetrics(
        score=solve.chamfer_px,
        contour_mae_px=solve.chamfer_px,
        contour_p95_px=solve.chamfer_px,
        valid_columns=int(round(solve.coverage * width)),
        iterations=1,
        success=True,
        message=f"verdict {solve.verdict}",
        confidence=0.9 if solve.verdict == "CONFIRMED" else 0.4,
        position_error_m=None,
        yaw_error_deg=None,
        pitch_error_deg=None,
    )
    estimate = PoseEstimate(extrinsics=estimate_extrinsics, metrics=metrics)
    if truth is not None:
        estimate = add_synthetic_truth_metrics(estimate, truth)

    reduced = _reduced_intrinsics(intrinsics, SOLVE_SAMPLE_WIDTH)
    observed_reduced = _reduce_profile(obs, intrinsics, reduced)
    predicted_reduced = profile.rows_pinhole(
        reduced.width_px, reduced.height_px, hfov_deg, solve.yaw_deg, solve.pitch_deg
    )
    predicted_full = profile.rows_pinhole(width, height, hfov_deg, solve.yaw_deg, solve.pitch_deg)

    candidates = [
        PoseCandidate(extrinsics=extrinsics_at(yaw, solve.pitch_deg), score=float(chamfer))
        for chamfer, yaw, _dv in (solve.candidates or [(solve.chamfer_px, solve.yaw_deg, 0.0)])[:8]
    ]
    frame = SolveFrame(
        east_m=position.east_m,
        north_m=position.north_m,
        up_m=position.up_m,
        yaw_deg=estimate_extrinsics.yaw_deg,
        pitch_deg=solve.pitch_deg,
        score=solve.chamfer_px,
        evaluations=int(len(solve.yaw_profile_deg)),
        profile=_jsonable_profile(predicted_reduced),
    )
    return PoseSolveResult(
        strategy="horizon",
        estimate=estimate,
        evaluations=int(len(solve.yaw_profile_deg)),
        sample_width=reduced.width_px,
        sample_height=reduced.height_px,
        observed_profile=_jsonable_profile(observed_reduced),
        predicted_profile=_jsonable_profile(predicted_reduced),
        predicted_profile_full=_jsonable_profile(predicted_full),
        candidates=candidates,
        trace=[frame],
        diagnostics={
            "verdict": solve.verdict,
            "chamfer_px": round(solve.chamfer_px, 2),
            "alias_ratio": round(solve.alias_ratio, 2),
            "snr": round(solve.snr, 1),
            "well_width_deg": round(solve.well_width_deg, 1),
            "coverage": round(solve.coverage, 2),
        },
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
        # Multiplies the final confidence: <1 when the global search found a
        # well-separated alternative pose that fits nearly as well (ambiguous).
        self.localization_factor = 1.0
        # Distinct plausible (score, theta) poses from a prior-free search.
        self.candidates: list[tuple[float, NDArray[np.float64]]] = []

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


def _run_global(traced: _TracedSolve) -> None:
    """Prior-free localization: horizon-correlation seeds, then local refine.

    Finds the best whole-map (position, yaw) candidates by correlating the
    observed outline against each candidate's 360 deg terrain horizon, then
    refines the top seeds with Powell on the full 5-DOF objective.
    """

    objective = traced.objective
    prior = objective.prior
    terrain = objective.terrain
    eye_height_m = float(
        np.clip(prior.position.up_m - terrain.elevation_at(prior.position.east_m, prior.position.north_m), 2.0, 4000.0)
    )
    seeds = horizon_seeds(
        terrain,
        objective.observed_profile,
        objective.intrinsics,
        pitch_deg=prior.pitch_deg,
        eye_height_m=eye_height_m,
    )
    bounds = objective.bounds()
    lows = np.array([low for low, _high in bounds], dtype=np.float64)
    highs = np.array([high for _low, high in bounds], dtype=np.float64)

    # Stage 1: cheaply polish every seed so the true basin (often outranked by
    # ambiguous skyline matches in the coarse score) gets a fair full-objective
    # evaluation. Stage 2: deep-refine only the few best by full-objective score.
    polished: list[tuple[float, NDArray[np.float64]]] = []
    for seed in seeds:
        theta0 = np.clip(
            np.array(
                [seed.position.east_m, seed.position.north_m, seed.position.up_m, seed.yaw_deg, seed.pitch_deg],
                dtype=np.float64,
            ),
            lows,
            highs,
        )
        traced.cost(theta0)
        result = minimize(
            traced.cost,
            theta0,
            method="Powell",
            bounds=bounds,
            options={"maxiter": 20, "xtol": 5e-2, "ftol": 5e-2, "disp": False},
        )
        theta = np.asarray(result.x, dtype=np.float64)
        polished.append((float(traced.objective.score(theta)), theta))
        traced.record()

    polished.sort(key=lambda item: item[0])
    refined: list[tuple[float, NDArray[np.float64]]] = []
    for _score, theta in polished[:8]:
        result = minimize(
            traced.cost,
            theta,
            method="Powell",
            bounds=bounds,
            callback=traced.record,
            options={"maxiter": 80, "xtol": 1e-2, "ftol": 1e-2, "disp": False},
        )
        refined.append((float(traced.objective.score(result.x)), np.asarray(result.x, dtype=np.float64)))
    traced.record()

    # Collect every well-separated pose that fits, best-first: a single outline is
    # often ambiguous, so all of them are real candidate locations.
    separation_m = 0.05 * float(traced.terrain.x_m[-1] - traced.terrain.x_m[0])
    scored = sorted(refined + polished, key=lambda item: item[0])
    traced.candidates = _distinct_candidates(scored, separation_m, max_count=10)
    traced.localization_factor = _candidate_margin(traced.candidates)


def _distinct_candidates(
    scored: list[tuple[float, NDArray[np.float64]]], separation_m: float, max_count: int
) -> list[tuple[float, NDArray[np.float64]]]:
    """Greedily keep best-scoring poses that are spatially well separated."""

    kept: list[tuple[float, NDArray[np.float64]]] = []
    for score, theta in scored:
        if all(np.hypot(theta[0] - other[0], theta[1] - other[1]) > separation_m for _s, other in kept):
            kept.append((score, theta))
            if len(kept) >= max_count:
                break
    return kept


def _candidate_margin(candidates: list[tuple[float, NDArray[np.float64]]]) -> float:
    """Confidence factor in [0, 1]: how much better the winner is than the runner-up.

    A small gap means several far-apart poses fit nearly as well (ambiguous),
    so a residual-only confidence would be misleadingly high.
    """

    if len(candidates) < 2:
        return 1.0
    best, runner_up = candidates[0][0], candidates[1][0]
    relative_gap = (runner_up - best) / max(abs(best), 1e-6)
    return float(np.clip(relative_gap / 0.5, 0.0, 1.0))


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
    # Keep the yaw scan ~8deg-spaced so a wide (compass-distrusting) yaw prior is
    # still covered finely enough to land in the right basin.
    yaw_span = prior.yaw_sigma_deg * 3.6
    yaw_count = int(np.clip(round(yaw_span / 8.0), 11, 60))
    yaw_values = np.linspace(prior.yaw_deg - yaw_span / 2.0, prior.yaw_deg + yaw_span / 2.0, yaw_count)
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
    observed_profile = contour.to_profile()
    predicted = renderer.skyline_profile(terrain, intrinsics, extrinsics, stride=1)
    residuals = robust_contour_residuals(predicted, observed_profile)
    mae, p95, valid_columns = residual_summary(residuals)
    observed_valid = int(np.sum(np.isfinite(observed_profile)))
    confidence, _is_match = match_confidence(p95, valid_columns / max(1, observed_valid), intrinsics.height_px)
    # Discount confidence when the global search found an ambiguous (multi-modal) fit.
    confidence *= traced.localization_factor
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
        confidence=confidence,
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

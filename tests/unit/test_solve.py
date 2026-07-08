"""Pluggable full-pose solver tests."""

from __future__ import annotations

import math

import pytest

from peakle.optimization.solve import STRATEGIES
from peakle.scene.scene import Scene

# `global` is prior-free and intentionally searches the whole map; on a tiny test
# scene its skyline match is ambiguous, so it is covered by a looser smoke test.
PRIOR_FREE_STRATEGIES = {"global"}
PRIOR_STRATEGIES = tuple(strategy for strategy in STRATEGIES if strategy not in PRIOR_FREE_STRATEGIES)


def _place_near_prominent_peak(scene: Scene):
    peak = max(scene.peaks, key=lambda candidate: candidate.prominence_m)
    east_m = peak.local_position.east_m - 2000.0
    north_m = peak.local_position.north_m - 3000.0
    yaw_deg = math.degrees(math.atan2(peak.local_position.east_m - east_m, peak.local_position.north_m - north_m))
    return scene.create_view(east_m, north_m, yaw_deg=yaw_deg, pitch_deg=3.0)


@pytest.mark.parametrize("strategy", PRIOR_STRATEGIES)
def test_solver_recovers_heading(scene: Scene, strategy: str) -> None:
    view = _place_near_prominent_peak(scene)

    solve = scene.run_solve(view.id, strategy, {"seed": 1})
    result = solve.result
    metrics = result.estimate.metrics

    assert result.strategy == strategy
    assert result.evaluations > 0
    assert result.trace, "expected a non-empty convergence trace"
    assert len(result.trace) <= 60
    # Yaw is well observed from the skyline; position is weakly observed.
    assert metrics.yaw_error_deg is not None
    assert metrics.yaw_error_deg < 6.0
    assert math.isfinite(metrics.contour_mae_px)
    # A good fit reports a meaningful confidence in [0, 1].
    assert metrics.confidence is not None
    assert 0.0 <= metrics.confidence <= 1.0


def test_global_strategy_runs_prior_free(scene: Scene) -> None:
    view = _place_near_prominent_peak(scene)
    solve = scene.run_solve(view.id, "global", {})
    result = solve.result
    assert result.strategy == "global"
    assert result.trace, "expected a non-empty convergence trace"
    assert math.isfinite(result.estimate.metrics.contour_mae_px)
    assert result.estimate.metrics.yaw_error_deg is not None


def test_contour_database_strategy_returns_ranked_candidates(scene: Scene) -> None:
    view = _place_near_prominent_peak(scene)
    solve = scene.run_solve(view.id, "contourdb", {})
    result = solve.result
    assert result.strategy == "contourdb"
    assert result.trace, "expected a non-empty convergence trace"
    assert result.candidates, "expected ranked database candidates"
    assert math.isfinite(result.estimate.metrics.contour_mae_px)
    assert result.estimate.metrics.yaw_error_deg is not None


def test_solves_accumulate_on_a_view(scene: Scene) -> None:
    view = _place_near_prominent_peak(scene)
    first = scene.run_solve(view.id, "nelder", {"seed": 1})
    second = scene.run_solve(view.id, "powell", {"seed": 1})

    assert first.id != second.id
    assert set(scene.views[view.id].solves) == {first.id, second.id}

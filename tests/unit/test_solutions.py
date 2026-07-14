"""Persistent pose solution store tests."""

from __future__ import annotations

import pytest

from peakle.scene.scene import Scene
from peakle.web.solutions import SolutionStore, solution_key


def test_solution_store_persists_gt_view_solves_across_runtime_ids(scene: Scene, tmp_path) -> None:
    view = scene.create_view(0.0, -2000.0, 0.0, 2.0)
    view.source = "gt"
    view.gt_name = "sample-a"
    solve = scene.run_solve(view.id, "horizon", {"seed": 1})

    store = SolutionStore(tmp_path)
    store.save(view, solve)

    reopened = view.model_copy(update={"id": "view-99", "solves": {}})
    loaded = SolutionStore(tmp_path).load(reopened)

    assert solution_key(reopened) == "gt:sample-a"
    assert len(loaded) == 1
    assert loaded[0].id == solve.id
    assert loaded[0].strategy == "horizon"


def test_solution_store_remove_deletes_one_solve(scene: Scene, tmp_path) -> None:
    view = scene.create_view(0.0, -2000.0, 0.0, 2.0)
    view.source = "gt"
    view.gt_name = "sample-a"
    solve = scene.run_solve(view.id, "horizon", {"seed": 1})
    store = SolutionStore(tmp_path)
    store.save(view, solve)

    store.remove(view, solve.id)

    assert store.load(view) == []


def test_solution_store_rejects_results_for_changed_solver_inputs(scene: Scene, tmp_path) -> None:
    view = scene.create_view(0.0, -2000.0, 0.0, 2.0)
    view.source = "gt"
    view.gt_name = "sample-a"
    solve = scene.run_solve(view.id, "horizon", {"seed": 1})
    store = SolutionStore(tmp_path)
    store.save(view, solve)

    changed_prior = view.prior.model_copy(update={"yaw_deg": view.prior.yaw_deg + 3.0})
    changed = view.model_copy(update={"prior": changed_prior, "solves": {}})

    assert store.load(changed) == []

    changed_terrain = view.model_copy(update={"terrain_fingerprint": "different-terrain", "solves": {}})

    assert store.load(changed_terrain) == []

    replacement = solve.model_copy(update={"id": "solve-replacement", "prior": changed_prior})
    store.save(changed, replacement)

    assert [row.id for row in store.load(changed)] == ["solve-replacement"]


def test_gt_reference_pose_cannot_be_used_as_solver_prior(scene: Scene) -> None:
    view = scene.create_view(0.0, -2000.0, 0.0, 2.0)
    view.source = "gt"

    with pytest.raises(ValueError, match="evaluation reference"):
        scene.run_solve(view.id, "horizon", {"prior_source": "pose:truth"})


def test_gt_solve_requires_explicit_available_evidence_and_hides_crop_pitch_error(scene: Scene) -> None:
    view = scene.create_view(0.0, -2000.0, 0.0, 2.0)
    view.source = "gt"
    view.default_evidence_source = "photo_auto"
    view.evidence_contours = {"pfm_oracle": view.contour}
    view.evidence_metadata = {
        "photo_auto": {"available": False, "reason": "photo extraction rejected"},
        "pfm_oracle": {"available": True, "diagnostic": True},
    }
    view.pitch_comparable = False

    with pytest.raises(ValueError, match="photo_auto.*unavailable"):
        scene.run_solve(view.id, "horizon")

    solve = scene.run_solve(view.id, "horizon", {"evidence_source": "pfm_oracle"})

    assert solve.params["evidence_source"] == "pfm_oracle"
    assert solve.params["evidence_provenance"]["diagnostic"] is True
    assert len(solve.params["evidence_contour_sha256"]) == 64
    assert solve.params["pitch_comparable"] is False
    assert solve.result.estimate.metrics.yaw_error_deg is not None
    assert solve.result.estimate.metrics.pitch_error_deg is None

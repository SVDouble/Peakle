"""Persistent pose solution store tests."""

from __future__ import annotations

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

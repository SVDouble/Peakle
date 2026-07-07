"""Geometry checks for weighted-explanation scoring: weights, tolerance, and unexplained noise."""

import types

import numpy as np

from peakle.localize import explanation as expl
from peakle.localize.explanation import arbitrate_by_explanation, explanation_score

H, W = 100, 160


def _line(row, c0=10, c1=150):
    m = np.zeros((H, W), bool)
    m[row, c0:c1] = True
    return m


def test_weights_and_unexplained_noise():
    # photo contours: one on the DEM skyline, one on a DEM rib, one pure noise
    photo = _line(20) | _line(50) | _line(80)
    dem = {"sky": _line(22), "occ": np.zeros((H, W), bool), "rib": _line(52), "cou": np.zeros((H, W), bool)}
    rep = explanation_score(photo, dem)
    # thirds: 1.0 (sky) + 0.3 (rib) + 0.0 (noise) -> mean ~0.433
    assert abs(rep["score"] - (1.0 + 0.3 + 0.0) / 3.0) < 0.02, rep
    assert rep["explained"]["sky"] > 0.3 and rep["explained"]["rib"] > 0.3


def test_best_family_wins_overlap():
    photo = _line(30)
    dem = {"sky": _line(33), "occ": _line(28), "rib": _line(30), "cou": np.zeros((H, W), bool)}
    rep = explanation_score(photo, dem)
    assert abs(rep["score"] - 1.0) < 1e-6, "overlapping families must credit the highest weight"


def test_out_of_tolerance_explains_nothing():
    photo = _line(20)
    dem = {
        "sky": _line(60),
        "occ": np.zeros((H, W), bool),
        "rib": np.zeros((H, W), bool),
        "cou": np.zeros((H, W), bool),
    }
    assert explanation_score(photo, dem)["score"] == 0.0


def test_right_pose_beats_wrong_pose():
    # the wrong pose fits the skyline but leaves internal structure unexplained
    photo = _line(20) | _line(55) | _line(75)
    right = {"sky": _line(21), "occ": _line(56), "rib": _line(74), "cou": np.zeros((H, W), bool)}
    wrong = {"sky": _line(21), "occ": _line(95), "rib": np.zeros((H, W), bool), "cou": np.zeros((H, W), bool)}
    assert explanation_score(photo, right)["score"] > explanation_score(photo, wrong)["score"] + 0.25


def _fake_solve(chamfer, yaw):
    return types.SimpleNamespace(chamfer_px=chamfer, yaw_deg=yaw, candidates=[(chamfer, yaw, 0.0)])


def test_arbitration_prefers_better_explanation_within_slack(monkeypatch):
    # "good" (yaw 10) has slightly HIGHER chamfer but explains the photo's ribs;
    # "lake" (yaw 200) wins on chamfer but explains nothing internal.
    photo = _line(20) | _line(55) | _line(75)
    masks = {
        10: {"sky": _line(21), "occ": _line(56), "rib": _line(74), "cou": np.zeros((H, W), bool)},
        200: {"sky": _line(21), "occ": _line(95), "rib": np.zeros((H, W), bool), "cou": np.zeros((H, W), bool)},
    }
    monkeypatch.setattr(expl, "dem_typed_masks", lambda *a, **k: masks[round(a[5])])
    solved = {"lake": (None, _fake_solve(10.0, 200)), "good": (None, _fake_solve(11.0, 10))}
    win, scores = arbitrate_by_explanation(solved, None, 0.0, W, H, 50.0, photo)
    assert win == "good", scores


def test_arbitration_slack_excludes_far_chamfer(monkeypatch):
    # same masks, but now "good" chamfer is 1.5x the best -> outside the 1.2 slack,
    # so the chamfer winner "lake" must stand even though it explains less.
    photo = _line(20) | _line(55) | _line(75)
    masks = {
        10: {"sky": _line(21), "occ": _line(56), "rib": _line(74), "cou": np.zeros((H, W), bool)},
        200: {"sky": _line(21), "occ": _line(95), "rib": np.zeros((H, W), bool), "cou": np.zeros((H, W), bool)},
    }
    monkeypatch.setattr(expl, "dem_typed_masks", lambda *a, **k: masks[round(a[5])])
    solved = {"lake": (None, _fake_solve(10.0, 200)), "good": (None, _fake_solve(15.0, 10))}
    win, _ = arbitrate_by_explanation(solved, None, 0.0, W, H, 50.0, photo)
    assert win == "lake"

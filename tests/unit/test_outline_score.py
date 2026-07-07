"""Validation of the outline-scoring API: known geometric cases with exact expected metrics."""

import numpy as np
import pytest

from peakle.localize.outline_score import rows_to_mask, score_outlines


@pytest.fixture()
def gt_npz(tmp_path):
    h, w = 120, 200
    sky = np.full(w, 30.0)  # flat skyline at row 30
    contours = np.zeros((h, w), bool)
    contours[70, 20:180] = True  # one internal contour at row 70
    p = tmp_path / "gt.npz"
    np.savez_compressed(
        p,
        gt_skyline=sky.astype(np.float32),
        dem_skyline=sky.astype(np.float32),
        gt_contours=np.packbits(contours),
        dem_contours=np.packbits(contours),
        shape=np.array([h, w]),
    )
    return p


def test_perfect_prediction_scores_one(gt_npz):
    pred = rows_to_mask(np.full(200, 30.0), 120)
    pred[70, 20:180] = True
    s = score_outlines(pred, gt_npz)
    assert s.precision == pytest.approx(1.0)
    assert s.recall_skyline == pytest.approx(1.0)
    assert s.recall_internal == pytest.approx(1.0)
    assert s.f1 == pytest.approx(1.0)


def test_skyline_only_prediction_shows_internal_miss(gt_npz):
    pred = rows_to_mask(np.full(200, 31.0), 120)  # skyline 1px off, no internal lines
    s = score_outlines(pred, gt_npz)
    assert s.precision == pytest.approx(1.0)
    assert s.recall_skyline == pytest.approx(1.0)
    assert s.recall_internal == 0.0, "missing every internal contour must be visible per-family"
    assert s.f1 < 0.9


def test_garbage_prediction_scores_low(gt_npz):
    pred = np.zeros((120, 200), bool)
    pred[100:118, :] = np.random.default_rng(0).random((18, 200)) > 0.5  # noise far from GT
    s = score_outlines(pred, gt_npz)
    assert s.precision < 0.05
    assert s.recall_skyline == 0.0


def test_shape_mismatch_raises(gt_npz):
    with pytest.raises(ValueError):
        score_outlines(np.zeros((50, 50), bool), gt_npz)

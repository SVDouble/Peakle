"""Tests for curvature-based DEM ridge/valley extraction."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from peakle.dem_ridges import _trace_grid, ridge_valley_masks


def test_ridge_mask_detects_crest() -> None:
    height, width = 40, 40
    yy, _ = np.mgrid[0:height, 0:width]
    elevation = 100.0 - np.abs(yy - 20) * 8.0  # a steep ridge crest along row 20
    terrain = SimpleNamespace(elevation_m=elevation)
    ridge, valley = ridge_valley_masks(terrain, prominence_m=3.0, smooth=0.5)
    assert ridge[20].sum() > width * 0.5      # crest row is a ridge
    assert ridge[5].sum() < ridge[20].sum()   # flat slope is not
    assert valley.sum() < ridge.sum()         # a pure ridge has few troughs


def test_valley_mask_detects_trough() -> None:
    height, width = 40, 40
    yy, _ = np.mgrid[0:height, 0:width]
    terrain = SimpleNamespace(elevation_m=np.abs(yy - 20) * 8.0)  # steep trough along row 20
    _, valley = ridge_valley_masks(terrain, prominence_m=3.0, smooth=0.5)
    assert valley[20].sum() > width * 0.4


def test_trace_grid_returns_polyline() -> None:
    mask = np.zeros((30, 30), dtype=bool)
    mask[15, 5:25] = True
    polylines = _trace_grid(mask, min_len=5)
    assert polylines and len(polylines[0]) >= 10

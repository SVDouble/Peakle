"""Geometry checks for the photo-edge-support metric (no learned model needed)."""

import numpy as np

from peakle.localize.photo_support import family_support


def test_supported_line_scores_one():
    line = np.zeros((100, 120), bool)
    line[40, 10:110] = True
    edges = np.zeros_like(line)
    edges[43, 10:110] = True                     # 3 px away, within tolerance
    assert family_support(line, edges) == 1.0


def test_displaced_line_scores_zero():
    line = np.zeros((100, 120), bool)
    line[40, 10:110] = True
    edges = np.zeros_like(line)
    edges[80, 10:110] = True                     # 40 px away — the photo shows the edge elsewhere
    assert family_support(line, edges) == 0.0


def test_partial_support_is_fractional():
    line = np.zeros((100, 120), bool)
    line[40, 10:110] = True
    edges = np.zeros_like(line)
    edges[41, 10:60] = True                      # half the line has photo evidence
    s = family_support(line, edges)
    assert 0.45 <= s <= 0.55, s


def test_empty_line_is_none_and_no_edges_is_zero():
    line = np.zeros((50, 50), bool)
    edges = np.zeros((50, 50), bool)
    assert family_support(line, edges) is None
    line[10, 5:45] = True
    assert family_support(line, edges) == 0.0

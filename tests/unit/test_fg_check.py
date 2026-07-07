"""Geometric checks for the missing-foreground comparator (no learned model needed)."""

import numpy as np

from peakle.localize.fg_check import foreground_report

H, W = 100, 160


def test_photo_foreground_absent_from_dem_is_flagged():
    mono = np.full((H, W), 0.8)
    mono[40:, 20:90] = 0.05  # a big near tower in the photo
    dem = np.full((H, W), 6000.0)  # DEM renders only distant terrain
    rep = foreground_report(mono, dem)
    assert rep["missing_foreground"], rep
    assert rep["photo_fg_frac"] > 0.2 and rep["dem_fg_frac"] == 0.0


def test_matching_foreground_is_not_flagged():
    mono = np.full((H, W), 0.8)
    mono[40:, 20:90] = 0.05
    dem = np.full((H, W), 6000.0)
    dem[40:, 20:90] = 400.0  # the DEM has the near layer too
    rep = foreground_report(mono, dem)
    assert not rep["missing_foreground"], rep


def test_distant_only_scene_is_not_flagged():
    mono = np.full((H, W), 0.7)  # no near layer in the photo at all
    dem = np.full((H, W), 8000.0)
    rep = foreground_report(mono, dem)
    assert not rep["missing_foreground"], rep
    assert rep["photo_fg_frac"] == 0.0


def test_sky_excluded_via_skyline():
    mono = np.full((H, W), 0.9)
    mono[:30, :] = 0.1  # weird near-ish clouds ABOVE the skyline
    mono[60:, :] = 0.05  # real near terrain below
    dem = np.full((H, W), 7000.0)
    sky = np.full(W, 35.0)
    rep = foreground_report(mono, dem, sky_rows=sky)
    assert rep["missing_foreground"], rep
    # cloud pixels must not have inflated the fraction beyond the true below-sky share
    assert rep["photo_fg_frac"] <= 0.65

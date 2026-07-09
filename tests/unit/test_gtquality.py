"""GT alignment audit helpers."""

from __future__ import annotations

import math

import numpy as np

from peakle.localize.gtquality import alignment_audit, alignment_record, skyline_vertical_error_stats_m


def test_alignment_record_classifies_metric_failures() -> None:
    row = alignment_record(
        {
            "name": "matterhorn",
            "quality": "SUSPECT",
            "sky_cons_px": 32.0,
            "pfm_cons_px": 12.0,
            "contour_cons_px": 41.0,
            "pfm_offset_px": 22.0,
            "sky_error_m": 300.0,
            "pfm_error_m": 10.0,
            "sky_support": 0.3,
            "pfm_support": 0.9,
            "dyaw_deg": 7.0,
            "de_m": 100.0,
            "dn_m": 200.0,
            "obs_source": "photo",
            "reasons": ["manual check"],
        }
    )

    codes = {mode["code"] for mode in row["failure_modes"]}
    assert "dem_observed_skyline_mismatch" in codes
    assert "dem_outline_mismatch" in codes
    assert "photo_pfm_registration_mismatch" in codes
    assert "dem_observed_skyline_meter_mismatch" in codes
    assert "weak_photo_skyline_support" in codes
    assert "dem_pfm_skyline_mismatch" not in codes
    assert row["severity"] > 1.0


def test_alignment_audit_ranks_worst_records_and_counts_modes() -> None:
    audit = alignment_audit(
        [
            {"name": "clean", "quality": "CLEAN", "sky_cons_px": 2.0, "pfm_cons_px": 3.0},
            {"name": "bad", "quality": "SUSPECT", "sky_cons_px": 60.0, "pfm_cons_px": 80.0},
        ],
        include_clean=False,
    )

    assert audit["total"] == 2
    assert audit["clean"] == 1
    assert audit["suspect"] == 1
    assert [row["name"] for row in audit["rows"]] == ["bad"]
    assert audit["mode_counts"]["dem_observed_skyline_mismatch"] == 1
    assert audit["mode_counts"]["dem_pfm_skyline_mismatch"] == 1


def test_skyline_vertical_error_stats_converts_rows_to_meters() -> None:
    width, height, fov = 100, 50, 60.0
    focal = width / math.radians(fov)
    dem_rows = np.asarray([24.5, 24.5])
    # One column is 0.01 tangent-elevation above the DEM, the other 0.02 below it.
    observed_rows = np.asarray([24.5 - 0.01 * focal, 24.5 + 0.02 * focal])
    ranges = np.asarray([1000.0, 2000.0])

    stats = skyline_vertical_error_stats_m(observed_rows, dem_rows, ranges, width, height, fov)

    assert stats["mean_m"] == 25.0
    assert stats["median_m"] == 25.0
    assert stats["p90_m"] == 37.0
    assert stats["range_median_m"] == 1500.0

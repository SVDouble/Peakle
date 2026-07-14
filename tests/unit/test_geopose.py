from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from peakle.localize import geopose
from peakle.localize.geopose import load_sample


def _write_info(root: Path, lines: list[str]) -> None:
    root.mkdir()
    (root / "info.txt").write_text("\n".join(lines) + "\n")


def test_load_sample_retains_original_metadata_but_keeps_refined_primary(tmp_path: Path) -> None:
    root = tmp_path / "sample"
    _write_info(
        root,
        [
            "MANUAL",
            "0 0 0",
            "46.4894",
            "7.62461",
            "2320.0",
            "0.796052",
            "46.471001",
            "7.6139998",
            "2838.5",
            "0.59904179082",
        ],
    )

    sample = load_sample(root)

    assert sample.lat == pytest.approx(46.4894)
    assert sample.lon == pytest.approx(7.62461)
    assert sample.elev_m == pytest.approx(2320.0)
    assert sample.fov_deg == pytest.approx(math.degrees(0.796052))
    assert sample.original_metadata is not None
    assert sample.original_metadata.lat == pytest.approx(46.471001)
    assert sample.original_metadata.lon == pytest.approx(7.6139998)
    assert sample.original_metadata.elev_m == pytest.approx(2838.5)
    assert sample.original_metadata.fov_deg == pytest.approx(math.degrees(0.59904179082))


def test_load_sample_accepts_legacy_six_line_metadata(tmp_path: Path) -> None:
    root = tmp_path / "legacy"
    _write_info(
        root,
        [
            "AUTO",
            "0 0 0",
            "46.0",
            "8.0",
            "1500.0",
            "1.0471975512",
        ],
    )

    sample = load_sample(root)

    assert sample.manual is False
    assert sample.fov_deg == pytest.approx(60.0)
    assert sample.original_metadata is None


def test_resampled_oracle_skyline_scales_depth_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    depth = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 2.0],
            [1.0, 3.0, 0.0, 2.0],
            [1.0, 3.0, 4.0, 2.0],
        ],
        dtype=float,
    )
    monkeypatch.setattr(geopose, "read_pfm", lambda _path: depth)

    rows = geopose.resampled_oracle_skyline("dummy.pfm", width=4, height=8)

    assert rows.tolist() == [2.0, 4.0, 6.0, 2.0]

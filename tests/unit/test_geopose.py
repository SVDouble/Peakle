import numpy as np

from peakle.localize import geopose


def test_resampled_oracle_skyline_scales_depth_rows(monkeypatch) -> None:
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

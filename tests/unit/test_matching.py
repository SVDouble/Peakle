"""Outline fit + match-decision tests."""

from __future__ import annotations

import math

import numpy as np

from peakle.matching import fit_outline, outline_match, rerank_by_ridges
from peakle.rendering.rasterizer import SyntheticRenderer
from peakle.scene.scene import Scene
from peakle.segmentation import Ridge, RidgeField, extract_dp


def _place(scene: Scene):
    peak = max(scene.peaks, key=lambda candidate: candidate.prominence_m)
    east_m = peak.local_position.east_m - 2000.0
    north_m = peak.local_position.north_m - 3000.0
    yaw_deg = math.degrees(math.atan2(peak.local_position.east_m - east_m, peak.local_position.north_m - north_m))
    return scene.create_view(east_m, north_m, yaw_deg=yaw_deg, pitch_deg=3.0)


def test_extracted_outline_fits_and_matches(scene: Scene) -> None:
    view = _place(scene)
    image = np.asarray(view.render_arrays.image, dtype=np.float64) / 255.0
    outline = extract_dp(image)

    result, report = fit_outline(
        scene.terrain,
        outline,
        view.intrinsics.height_px,
        view.intrinsics,
        view.prior,
        strategy="powell",
        truth=view.true_extrinsics,
    )

    assert report.is_match
    assert report.p95_px < 15.0
    assert report.coverage > 0.8

    # A large vertical offset is a different skyline and must be rejected.
    shifted = np.asarray(result.predicted_profile, dtype=np.float64) + 60.0
    assert not outline_match(result.observed_profile, shifted, result.sample_height).is_match


def test_multi_ridge_rerank_picks_true_pose(scene: Scene) -> None:
    view = _place(scene)
    renderer = SyntheticRenderer()
    true_ext = view.true_extrinsics
    layers = renderer.ridge_layers(scene.terrain, view.intrinsics, true_ext, stride=2)

    def confidence(rows: np.ndarray) -> np.ndarray:
        return np.where(np.isfinite(rows), 1.0, 0.0)

    observed = RidgeField(
        skyline=Ridge(layers[0], confidence(layers[0]), "skyline"),
        ridges=[Ridge(layers[k], confidence(layers[k]), "ridge") for k in range(1, layers.shape[0])],
        response=np.zeros((1, 1)),
    )
    wrong = true_ext.model_copy(update={"yaw_deg": true_ext.yaw_deg + 25.0})
    ranked = rerank_by_ridges(scene.terrain, observed, view.intrinsics, [wrong, true_ext], stride=2)

    assert abs(ranked[0][1].yaw_deg - true_ext.yaw_deg) < 1.0  # true pose wins
    assert ranked[0][0] < ranked[1][0]  # and scores strictly better

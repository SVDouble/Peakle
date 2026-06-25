"""Photo-processing pipeline tests."""

from __future__ import annotations

import math

import numpy as np

from peakle.pipeline import Evidence, Pipeline, default_pipeline, localize
from peakle.pipeline.evidence import ExifData
from peakle.pipeline.exif import horizontal_fov_deg, intrinsics_from_exif
from peakle.scene.scene import Scene


def test_exif_focal_length_sets_fov() -> None:
    exif = ExifData(present=True, focal_length_35mm_mm=24.0)
    fov = horizontal_fov_deg(exif)
    assert fov is not None
    assert 70.0 < fov < 78.0
    intrinsics = intrinsics_from_exif(exif, 960, 540, 55.0)
    assert math.isclose(intrinsics.horizontal_fov_deg(), fov, abs_tol=1.0)


def test_no_exif_falls_back_to_default_fov() -> None:
    exif = ExifData(present=False)
    assert horizontal_fov_deg(exif) is None
    intrinsics = intrinsics_from_exif(exif, 960, 540, 55.0)
    assert math.isclose(intrinsics.horizontal_fov_deg(), 55.0, abs_tol=1.0)


def _view_image(scene: Scene):
    peak = max(scene.peaks, key=lambda candidate: candidate.prominence_m)
    east_m = peak.local_position.east_m - 2000.0
    north_m = peak.local_position.north_m - 3000.0
    yaw = math.degrees(math.atan2(peak.local_position.east_m - east_m, peak.local_position.north_m - north_m))
    view = scene.create_view(east_m, north_m, yaw_deg=yaw, pitch_deg=3.0)
    return np.asarray(view.render_arrays.image, dtype=np.float64) / 255.0


def test_pipeline_augments_and_localizes(scene: Scene) -> None:
    evidence = default_pipeline(scene.terrain).run_sync(Evidence.from_array(_view_image(scene), source="test-view"))

    # Each stage augmented the record.
    assert evidence.intrinsics is not None
    assert evidence.contour is not None and len(evidence.contour.points) > 1
    assert evidence.depth is not None and evidence.depth.shape == (evidence.image_height_px, evidence.image_width_px)
    assert evidence.edges is None  # no learned edge detector supplied -> classical response
    assert evidence.ridges is not None and evidence.ridges.skyline.rows.size == evidence.image_width_px
    assert evidence.prior is None  # no EXIF on a synthetic render
    assert len(evidence.provenance) == 7

    result = localize(evidence, scene.terrain)
    assert result.candidates  # prior-free search returns plausible poses


def test_parallel_matches_sequential(scene: Scene) -> None:
    image = _view_image(scene)
    parallel = Pipeline(default_pipeline(scene.terrain).stages, parallel=True).run_sync(Evidence.from_array(image))
    sequential = Pipeline(default_pipeline(scene.terrain).stages, parallel=False).run_sync(Evidence.from_array(image))

    assert math.isclose(parallel.intrinsics.horizontal_fov_deg(), sequential.intrinsics.horizontal_fov_deg())
    assert len(parallel.contour.points) == len(sequential.contour.points)
    assert sorted(parallel.provenance) == sorted(sequential.provenance)

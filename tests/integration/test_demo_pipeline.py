"""Synthetic demo pipeline integration tests."""

import json
from pathlib import Path

from peakle.config import load_settings
from peakle.demo.pipeline import DemoOptions, run_demo


def test_demo_pipeline_writes_scene_artifacts(tmp_path: Path) -> None:
    settings = load_settings()
    result = run_demo(
        DemoOptions.from_settings(
            settings,
            output_dir=tmp_path,
            seed=42,
            grid_width=96,
            grid_height=72,
            image_width=480,
            image_height=270,
            optimization_max_iterations=40,
        )
    )

    for filename in ("terrain.npz", "terrain.json", "peaks.json", "scene.json"):
        assert (tmp_path / filename).exists()

    # Views are computed on the fly by the live server; the demo no longer
    # precomputes per-view artifacts or static viewer assets.
    assert not (tmp_path / "viewer-data.json").exists()
    assert not (tmp_path / "app.js").exists()
    assert not (tmp_path / "views").exists()

    scene = json.loads((tmp_path / "scene.json").read_text(encoding="utf-8"))
    assert {"terrain_spec", "intrinsics", "true_camera", "pose_prior", "peaks"} <= set(scene)

    assert result.visible_labels >= 1
    assert result.position_error_m is not None
    assert result.position_error_m < 180.0
    assert result.yaw_error_deg is not None
    assert result.yaw_error_deg < 4.0
    assert result.contour_mae_px < 25.0

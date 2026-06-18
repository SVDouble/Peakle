"""Synthetic demo integration tests."""

import json
from pathlib import Path

from peakle.config import load_settings
from peakle.demo.pipeline import DemoOptions, run_demo


def test_demo_pipeline_writes_browser_artifacts(tmp_path: Path) -> None:
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

    expected = [
        "terrain.npz",
        "terrain.json",
        "peaks.json",
        "scene.json",
        "viewer-data.json",
        "index.html",
        "app.js",
        "styles.css",
        "views/view-01/render.png",
        "views/view-01/terrain_mask.png",
        "views/view-01/contour.json",
        "views/view-01/pose_estimate.json",
        "views/view-01/annotations.json",
        "views/view-01/annotated.png",
    ]
    for filename in expected:
        assert (tmp_path / filename).exists()

    viewer_data = json.loads((tmp_path / "viewer-data.json").read_text(encoding="utf-8"))
    assert set(viewer_data) == {"peaks", "scene", "terrain", "views"}
    assert viewer_data["views"]
    assert not (tmp_path / "render.png").exists()
    assert not (tmp_path / "annotated.png").exists()

    assert result.visible_labels >= 1
    assert result.position_error_m is not None
    assert result.position_error_m < 180.0
    assert result.yaw_error_deg is not None
    assert result.yaw_error_deg < 4.0
    assert result.contour_mae_px < 25.0

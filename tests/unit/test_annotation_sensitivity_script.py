from __future__ import annotations

import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from peakle.config import AppSettings, settings_payload
from peakle.scripts import bench_annotation_sensitivity as module


def _args(output: Path, **updates) -> Namespace:
    values = vars(module._parser().parse_args(["--output", str(output)]))
    values.update(updates)
    return Namespace(**values)


def test_effective_settings_apply_small_square_reproducible_overrides(
    tmp_path: Path,
    small_settings: AppSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "settings.yaml"
    loaded: list[Path | None] = []
    monkeypatch.setattr(module, "load_settings", lambda path: loaded.append(path) or small_settings)

    effective = module._effective_settings(
        _args(
            tmp_path / "out",
            config=config,
            seed=73,
            terrain_extent_m=12_000.0,
            terrain_grid=64,
            image_width=200,
            image_height=120,
            max_views=2,
            camera_ground_clearance_m=2.5,
        )
    )

    assert loaded == [config]
    assert effective.random_seed == effective.terrain.seed == 73
    assert (effective.terrain.width_m, effective.terrain.height_m) == (12_000.0, 12_000.0)
    assert (effective.terrain.grid_width, effective.terrain.grid_height) == (64, 64)
    assert (effective.render.image_width, effective.render.image_height) == (200, 120)
    assert effective.camera.view_count == 2
    assert effective.camera.overlook_height_m == 2.5


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("terrain_extent_m", 1_999.0, "terrain-extent-m"),
        ("terrain_extent_m", float("inf"), "terrain-extent-m"),
        ("terrain_grid", 63, "terrain-grid"),
        ("image_width", 159, "image-width"),
        ("image_height", 119, "image-height"),
        ("max_views", 0, "max-views"),
        ("terrain_stride", 0, "terrain-stride"),
        ("max_labels", 0, "max-labels"),
        ("camera_ground_clearance_m", 0.0, "camera-ground-clearance"),
        ("camera_ground_clearance_m", float("nan"), "camera-ground-clearance"),
        ("seed", -1, "seed"),
    ],
)
def test_invalid_overrides_fail_before_scene_build(tmp_path: Path, name: str, value: object, message: str) -> None:
    with pytest.raises(SystemExit, match=message):
        module._validate_args(_args(tmp_path / "out", **{name: value}))


def test_main_publishes_typed_results_and_complete_hash_linked_run(
    tmp_path: Path,
    small_settings: AppSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "gate-1"
    state = SimpleNamespace(true_cameras=(object(), object()), terrain=object())
    study = SimpleNamespace(cases=(object(), object(), object()))
    results = {
        "schema": "peakle_annotation_sensitivity_suite_v1",
        "truth_contract": {"truth_class": "diagnostic_oracle", "estimator_present": False},
        "view_ids": ["view-01", "view-02"],
        "studies": [],
        "aggregates": [],
    }

    class Suite:
        studies = (study, study)

        @staticmethod
        def model_dump(*, mode: str, by_alias: bool) -> dict:
            assert (mode, by_alias) == ("json", True)
            return results

    selected: dict[str, object] = {}

    def run_suite(received_state, *, camera_indices, terrain_stride, max_labels):
        selected.update(
            state=received_state,
            camera_indices=tuple(camera_indices),
            terrain_stride=terrain_stride,
            max_labels=max_labels,
        )
        return Suite()

    monkeypatch.setattr(module, "_effective_settings", lambda _args: small_settings)
    monkeypatch.setattr(module.SceneState, "from_settings", lambda settings: state)
    monkeypatch.setattr(module, "_run_suite", run_suite)
    monkeypatch.setattr(module, "terrain_fingerprint", lambda _terrain: {"sha256": "a" * 64})
    monkeypatch.setattr(
        module,
        "whole_worktree_provenance",
        lambda: {"git_sha": "abc", "scope": "whole_worktree", "dirty": False},
    )
    monkeypatch.setattr(sys, "argv", ["bench", "--output", str(output), "--terrain-stride", "3"])

    module.main()

    results_bytes = (output / "results.json").read_bytes()
    run_bytes = (output / "run.json").read_bytes()
    run = json.loads(run_bytes)
    assert results_bytes == module.canonical_json_bytes(results)
    assert run_bytes == module.canonical_json_bytes(run)
    assert run["schema"] == module.RUN_SCHEMA == "peakle_annotation_sensitivity_run_v2"
    assert run["status"] == "complete"
    assert run["run_id"] == output.name
    assert (run["view_count"], run["case_count"]) == (2, 6)
    assert run["results_sha256"] == hashlib.sha256(results_bytes).hexdigest()
    assert run["effective_settings"] == settings_payload(small_settings)
    assert run["benchmark_args"]["seed"] == 13
    assert run["benchmark_args"]["terrain_extent_m"] == 16_000.0
    assert run["benchmark_args"]["terrain_grid"] == 256
    assert run["benchmark_args"]["terrain_stride"] == 3
    assert run["terrain"] == {"sha256": "a" * 64}
    assert selected == {"state": state, "camera_indices": (0, 1), "terrain_stride": 3, "max_labels": 8}
    assert "estimator_archive" not in results_bytes.decode()


def test_existing_output_fails_before_loading_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    monkeypatch.setattr(module, "_effective_settings", lambda _args: pytest.fail("settings should not be loaded"))
    monkeypatch.setattr(sys, "argv", ["bench", "--output", str(output)])

    with pytest.raises(SystemExit, match="refusing to overwrite"):
        module.main()

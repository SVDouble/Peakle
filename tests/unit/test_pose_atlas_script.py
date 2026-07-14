from __future__ import annotations

from pathlib import Path

import pytest

from peakle.scripts import bench_pose_atlas as module


def test_cli_helpers_reject_ambiguous_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    samples = [Path("/data/alpha"), Path("/data/beta")]
    monkeypatch.setattr(module, "find_sample_dirs", lambda: samples)

    assert module._selected_samples("alpha,beta") == samples
    assert module._csv_tracks("photo_auto,pfm_oracle,photo_auto") == (
        "photo_auto",
        "pfm_oracle",
    )
    with pytest.raises(SystemExit, match="duplicate"):
        module._selected_samples("alpha,alpha")
    with pytest.raises(SystemExit, match="unknown"):
        module._selected_samples("missing")
    with pytest.raises(SystemExit, match="unknown evidence"):
        module._csv_tracks("made_up")
    with pytest.raises(SystemExit):
        module._parser().parse_args(["--replicate", "-1", "--output", "/tmp/new"])


def test_aggregate_and_summary_use_full_lattice_oracle() -> None:
    error = {
        "horizontal_position_m": 25.0,
        "vertical_m": 1.0,
        "position_3d_m": 25.02,
        "yaw_deg": 1.0,
        "pitch_deg": None,
    }
    evaluated = {
        "candidate_id": "candidate",
        "estimator_rank": 3,
        "estimator_rank_scope": "full_score_lattice",
        "errors": error,
        "normalized_joint_error": 0.25,
        "reaches_target": True,
    }
    top_k = [
        {
            "requested_k": 100,
            "evaluated_k": 3,
            "reaches_target": True,
            "recall": 1.0,
            "best_candidate": evaluated,
        }
    ]
    samples = [
        {
            "name": "alpha",
            "tracks": {
                "pfm_oracle": {
                    "status": "ok",
                    "runtime_s": 2.5,
                    "evaluation": {
                        "winner_errors": evaluated,
                        "full_lattice_gt_oracle": evaluated,
                        "shortlist_top_k": top_k,
                    },
                }
            },
        }
    ]

    aggregate = module._aggregates(samples, ("pfm_oracle",))[0]
    summary = module._summary_markdown({"samples": samples})

    assert aggregate["full_lattice_oracle_successes"] == 1
    assert aggregate["median_full_lattice_oracle_horizontal_m"] == 25.0
    assert "full-lattice oracle" in summary
    assert "25.0 m / 1.0°" in summary


def test_cache_inventory_is_compact_and_content_attested() -> None:
    inventory = {
        "terrain": {
            "file_count": 2,
            "files": [
                {"name": "a.tif", "size": 10, "mtime_ns": 1},
                {"name": "b.tif", "size": 20, "mtime_ns": 2},
            ],
        }
    }

    compact = module._compact_cache_inventory(inventory)

    assert compact["scope"] == "available_cache_inventory_not_exact_consumption"
    assert compact["caches"]["terrain"]["file_count"] == 2
    assert compact["caches"]["terrain"]["total_size_bytes"] == 30
    assert "files" not in compact["caches"]["terrain"]

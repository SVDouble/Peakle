"""Workbench API integration tests."""

from __future__ import annotations

import hashlib
import json
import time
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from peakle.domain.camera import CameraExtrinsics
from peakle.domain.contours import ImagePoint, SkylineContour
from peakle.domain.coordinates import LocalPoint
from peakle.localize.atlas_dashboard import (
    ATLAS_DASHBOARD_FILENAME,
    ATLAS_DASHBOARD_SCHEMA,
    build_atlas_dashboard,
    canonical_json_bytes,
)
from peakle.scene.scene import Scene
from peakle.scene.state import build_intrinsics
from peakle.web.api import benchmarks as benchmarks_api
from peakle.web.api import gtlab as gtlab_api
from peakle.web.api import views as views_api
from peakle.web.app import create_app


def _wait_job(client: TestClient, job_id: str, timeout_s: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"completed", "failed", "cancelled"}:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish")


@pytest.fixture
def client(scene: Scene, tmp_path) -> TestClient:
    return TestClient(create_app(scene, job_store_dir=tmp_path / "jobs", solution_store_dir=tmp_path / "solutions"))


def test_scene_terrain_peaks_endpoints(client: TestClient) -> None:
    scene = client.get("/api/scene").json()
    assert scene["providers"] == ["demo", "srtm"]
    assert {strategy["name"] for strategy in scene["strategies"]} == {
        "powell",
        "cmaes",
        "contourdb",
        "nelder",
        "evolution",
        "global",
        "horizon",
    }
    assert client.get("/api/terrain").json()["grid_width"] == 96
    assert len(client.get("/api/peaks").json()) >= 1


def test_view_lifecycle_and_solve(client: TestClient) -> None:
    terrain = client.get("/api/terrain").json()
    east = (terrain["x_min_m"] + terrain["x_max_m"]) / 2
    north = terrain["y_min_m"] + 0.05 * (terrain["y_max_m"] - terrain["y_min_m"])

    created = client.post(
        "/api/views",
        json={"east_m": east, "north_m": north, "yaw_deg": 0.0, "pitch_deg": 2.0, "eye_height_m": 150.0},
    )
    assert created.status_code == 201
    view = created.json()
    view_id = view["id"]
    assert view["contour"]["points"]
    assert view["prior"] is not None
    assert view["image_camera"]["projection"] == "pinhole"
    assert view["image_camera"]["width_px"] == view["intrinsics"]["width_px"]
    assert view["default_evidence_source"] == "rendered_skyline"
    assert view["pitch_comparable"] is True
    assert view["evidence_sources"][0]["id"] == "rendered_skyline"

    image = client.get(f"/api/views/{view_id}/image")
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"

    patched = client.patch(f"/api/views/{view_id}", json={"label": "Ridge"})
    assert patched.json()["label"] == "Ridge"

    solved = client.post(
        f"/api/views/{view_id}/solves",
        json={"strategy": "nelder", "params": {"seed": 1, "prior_source": "pose:truth"}},
    )
    assert solved.status_code == 201
    solve = solved.json()
    assert solve["strategy"] == "nelder"
    assert solve["params"]["prior_source"] == "pose:truth"
    assert solve["params"]["evidence_source"] == "rendered_skyline"
    assert solve["prior"]["position"]["east_m"] == pytest.approx(view["true_extrinsics"]["position"]["east_m"])
    assert solve["result"]["trace"]
    # JSON-safe: no NaN/Infinity leaked into the payload.
    json.loads(solved.text)

    solve_id = solve["id"]
    assert client.get(f"/api/views/{view_id}/solves/{solve_id}").status_code == 200
    truth_layer = client.get(f"/api/views/{view_id}/poses/truth/layers/sky.png")
    assert truth_layer.status_code == 200
    assert truth_layer.headers["content-type"] == "image/png"
    solve_layer = client.get(f"/api/views/{view_id}/poses/solve:{solve_id}/layers/occ.png")
    assert solve_layer.status_code == 200
    assert solve_layer.headers["content-type"] == "image/png"
    assert len(client.get(f"/api/views/{view_id}/solves").json()) == 1
    persisted = client.app.state.solution_store.load(client.app.state.scene.views[view_id])
    assert [row.id for row in persisted] == [solve_id]
    assert persisted[0].params["prior_source"] == "pose:truth"


def test_solve_views_job_adds_solves(client: TestClient) -> None:
    terrain = client.get("/api/terrain").json()
    east = (terrain["x_min_m"] + terrain["x_max_m"]) / 2
    north = terrain["y_min_m"] + 0.05 * (terrain["y_max_m"] - terrain["y_min_m"])
    view = client.post(
        "/api/views",
        json={"east_m": east, "north_m": north, "yaw_deg": 0.0, "pitch_deg": 2.0, "eye_height_m": 150.0},
    ).json()

    created = client.post(
        "/api/jobs",
        json={
            "view_ids": [view["id"]],
            "strategy": "horizon",
            "params": {"seed": 1},
            "max_workers": 1,
        },
    )

    assert created.status_code == 202
    job = _wait_job(client, created.json()["id"])
    assert job["status"] == "completed"
    assert job["done"] == 1
    solves = client.get(f"/api/views/{view['id']}/solves").json()
    assert len(solves) == 1
    assert solves[0]["strategy"] == "horizon"


def test_solve_api_rejects_unavailable_default_evidence_without_oracle_fallback(
    client: TestClient, scene: Scene
) -> None:
    view = scene.create_view(0.0, -2000.0, 0.0, 2.0)
    view.default_evidence_source = "photo_auto"
    view.evidence_contours = {"pfm_oracle": view.contour}
    view.evidence_metadata = {
        "photo_auto": {"available": False, "reason": "photo extraction rejected"},
        "pfm_oracle": {"available": True, "diagnostic": True},
    }

    response = client.post(f"/api/views/{view.id}/solves", json={"strategy": "horizon", "params": {}})

    assert response.status_code == 400
    assert "photo_auto" in response.json()["detail"]
    assert "pfm_oracle" in response.json()["detail"]


def test_solve_views_job_rejects_catalogue_pseudo_solves(client: TestClient) -> None:
    created = client.post("/api/jobs", json={"view_ids": ["sample-a"], "max_workers": 1})

    assert created.status_code == 400
    assert "benchmark" in created.json()["detail"]


def test_gt_alignment_audit_route(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gtlab_api,
        "_index",
        lambda: {
            "ok": {"name": "ok", "quality": "CLEAN", "sky_cons_px": 2.0, "pfm_cons_px": 2.0},
            "bad": {"name": "bad", "quality": "SUSPECT", "sky_cons_px": 40.0, "pfm_cons_px": 5.0},
        },
    )

    report = client.get("/api/gt/alignment-audit?limit=10").json()

    assert report["total"] == 2
    assert report["rows"][0]["name"] == "bad"


def test_pose_benchmark_dashboard_api(client: TestClient, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "output"
    run_dir = output / "20260713-120000-geopose-bench"
    run_dir.mkdir(parents=True)
    (run_dir / "results.json").write_text(
        json.dumps(
            [
                {
                    "name": "good",
                    "manual": True,
                    "gt_consistency_px": 3.0,
                    "oracle": {"correct": True, "yaw_err": 0.4, "chamfer_px": 2.0, "verdict": "CONFIRMED"},
                    "extracted": {"correct": True, "yaw_err": 1.2, "chamfer_px": 5.0, "verdict": "CONFIRMED"},
                },
                {
                    "name": "bad",
                    "manual": False,
                    "gt_consistency_px": 20.0,
                    "oracle": {"correct": False, "yaw_err": 40.0, "chamfer_px": 12.0, "verdict": "AMBIGUOUS"},
                    "extracted": {"correct": False, "yaw_err": -80.0, "chamfer_px": 20.0, "verdict": "REJECTED"},
                },
            ]
        )
    )
    monkeypatch.setattr(benchmarks_api, "OUTPUT", output)

    runs = client.get("/api/bench/runs").json()
    assert runs[0]["sample_count"] == 2
    summary = client.get(f"/api/bench/runs/{runs[0]['id']}/summary").json()
    assert summary["subsets"]["all"]["pfm_oracle"]["success_rate"] == pytest.approx(0.5)
    assert summary["subsets"]["map_proxy_5px"]["photo_auto"]["success_rate"] == pytest.approx(1.0)
    cases = client.get(f"/api/bench/runs/{runs[0]['id']}/cases?subset=manual").json()
    assert [row["name"] for row in cases["rows"]] == ["good"]
    policy = client.get("/api/bench/compatibility").json()
    assert "solver_output" in policy["forbidden_inputs"]
    assert policy["tiers"]["MAP_A"]["p90_deg_lte"] == pytest.approx(0.75)
    assert client.get("/bench", follow_redirects=False).headers["location"] == "/benchmark.html"


def test_pose_benchmark_dashboard_surfaces_strategy_matrix(
    client: TestClient, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    run_dir = output / "20260713-130000-matrix-geopose-bench"
    run_dir.mkdir(parents=True)
    baseline_errors = {"horizontal_position_m": 200.0, "yaw_deg": 15.0, "pitch_deg": None}
    shared = {
        "name": "clean",
        "manual": True,
        "evidence_track": "pfm_oracle",
        "prior_regime": "perturbed_metadata",
        "status": "ok",
        "runtime_s": 1.0,
        "ranking_eligible": True,
        "ranking_exclusions": [],
        "compatibility_tier": "MAP_A",
        "photo_edge_supported": True,
        "original_metadata_diagnostic": {
            "available": True,
            "used_by_estimator": False,
            "used_for_ranking": False,
            "refined_minus_original": {
                "horizontal_position_m": 412.0,
                "vertical_m": -18.0,
                "fov_deg": 2.5,
            },
        },
    }
    cases = [
        {
            **shared,
            "id": "clean:keep",
            "algorithm": "keep-prior",
            "errors": baseline_errors,
            "success": {"value": False},
            "baseline": None,
        },
        {
            **shared,
            "id": "clean:cmaes",
            "algorithm": "cmaes",
            "errors": {"horizontal_position_m": 40.0, "yaw_deg": 2.0, "pitch_deg": None},
            "success": {"value": True},
            "baseline": {"errors": baseline_errors, "success": {"value": False}},
            "result": {
                "diagnostics": {
                    "candidate_validation": {
                        "schema": "peakle_candidate_pose_holdout_validation_v1",
                        "enabled": True,
                        "passed": True,
                        "failures": [],
                    }
                }
            },
        },
    ]
    payload = {
        "schema_version": 2,
        "run_id": run_dir.name,
        "rows": [
            {
                "name": "clean",
                "manual": True,
                "gt_dem_compatibility": {
                    "tier": "MAP_A",
                    "p90_deg": 0.2,
                    "height": {"tier": "HEIGHT_A", "raw_camera_clearance_m": 2.0},
                },
            }
        ],
        "matrix_cases": cases,
        "aggregates": [],
    }
    (run_dir / "results.json").write_text(json.dumps(payload))
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "created_at": "2026-07-13T13:00:00+00:00",
                "status": "complete",
                "matrix": {
                    "algorithms": ["keep-prior", "cmaes"],
                    "evidence_tracks": ["pfm_oracle"],
                    "prior_regimes": ["perturbed_metadata"],
                },
                "render_matching": {
                    "candidate_validation": {
                        "enabled": True,
                        "query_grid_columns": 8,
                        "query_grid_rows": 6,
                        "folds": 4,
                    }
                },
            }
        )
    )
    monkeypatch.setattr(benchmarks_api, "OUTPUT", output)

    runs = client.get("/api/bench/runs").json()
    assert runs[0]["kind"] == "strategy_matrix"
    assert runs[0]["algorithm_count"] == 2
    assert runs[0]["recommended"] is True
    assert runs[0]["candidate_validation"]["enabled"] is True
    summary = client.get(f"/api/bench/runs/{runs[0]['id']}/summary").json()
    assert summary["mode"] == "matrix"
    assert summary["default_subset"] == "primary_height_a"
    aggregates = summary["subsets"]["primary_height_a"]["aggregates"]
    cmaes = next(row for row in aggregates if row["algorithm"] == "cmaes")
    assert cmaes["success_rate"] == 1.0
    assert cmaes["median_position_delta_vs_prior_m"] == -160.0
    result = client.get(f"/api/bench/runs/{runs[0]['id']}/cases?algorithm=cmaes&subset=primary").json()
    assert result["mode"] == "matrix"
    assert result["rows"][0]["delta_vs_prior"]["yaw_deg"] == -13.0
    ambiguity = result["rows"][0]["original_metadata_diagnostic"]
    assert ambiguity["used_by_estimator"] is False
    assert ambiguity["used_for_ranking"] is False
    assert ambiguity["refined_minus_original"]["horizontal_position_m"] == 412.0
    assert result["rows"][0]["candidate_validation"]["passed"] is True


def test_pose_benchmark_dashboard_surfaces_compact_pose_atlas_study(
    client: TestClient, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    run_dir = output / "20260714-three-photo-native-skyline-pose-atlas-v2"
    run_dir.mkdir(parents=True)

    def evaluated(position_m: float, yaw_deg: float, rank: int, *, reaches: bool, scope: str) -> dict:
        return {
            "candidate_id": f"candidate-{rank}",
            "estimator_rank": rank,
            "estimator_rank_scope": scope,
            "hypothesis": {"estimator_score": rank / 1000.0, "pose": {"yaw_deg": 10.0}},
            "errors": {
                "horizontal_position_m": position_m,
                "vertical_m": 1.0,
                "position_3d_m": position_m + 0.1,
                "yaw_deg": yaw_deg,
                "pitch_deg": None,
            },
            "normalized_joint_error": max(position_m / 100.0, yaw_deg / 5.0),
            "reaches_target": reaches,
        }

    def complete_track(
        position_m: float,
        oracle_position_m: float,
        oracle_rank: int,
        *,
        winner_reaches: bool,
        top_100_reaches: bool,
    ) -> dict:
        winner = evaluated(
            position_m,
            3.0,
            1,
            reaches=winner_reaches,
            scope="spatially_diverse_shortlist",
        )
        oracle = evaluated(
            oracle_position_m,
            1.0,
            oracle_rank,
            reaches=True,
            scope="full_score_lattice",
        )
        return {
            "status": "ok",
            "runtime_s": 12.5,
            "evidence": {"available": True},
            "estimator_archive": {
                "archive_sha256": "archive-sha",
                "candidate_count": 1323,
                "selected_candidate_id": "candidate-1",
                "grid": {"position_count": 441},
                "query_geometry": {"width_px": 1024},
                "native_patch_supplied": True,
                "full_score_lattice": {
                    "yaw_count": 360,
                    "hypothesis_count": 158760,
                    "positions": [{"yaw_scores": [0.1, 0.2], "private_lattice_sentinel": True}],
                },
                "candidates": [{"private_candidate_sentinel": True}],
            },
            "evaluation": {
                "archive_sha256": "archive-sha",
                "reference_data_used": True,
                "used_by_estimator": False,
                "target": {
                    "horizontal_position_m_lte": 100.0,
                    "absolute_yaw_deg_lte": 5.0,
                    "pitch_used_by_oracle": False,
                },
                "winner_errors": winner,
                "full_lattice_gt_oracle": oracle,
                "selection_regret": {
                    "horizontal_position_m": position_m - oracle_position_m,
                    "yaw_deg": 2.0,
                },
                "shortlist_top_k": [
                    {
                        "candidate_pool": "spatially_diverse_shortlist",
                        "requested_k": 100,
                        "evaluated_k": 100,
                        "reaches_target": top_100_reaches,
                        "recall": 1.0 if top_100_reaches else 0.0,
                        "best_candidate": oracle,
                    }
                ],
                "evaluation_only_reference_position_probe": {
                    "used_by_estimator": False,
                    "included_in_estimator_archive": False,
                    "errors": oracle["errors"],
                    "score_delta_reference_minus_blind_winner": 0.5,
                    "best_yaw_mode": {"private_probe_sentinel": True},
                },
            },
        }

    compatibility = {
        "policy": "gt_dem_compat_v1",
        "tier": "MAP_B",
        "p90_deg": 0.75,
        "height": {"tier": "HEIGHT_A", "raw_camera_clearance_m": 2.5},
    }
    results = {
        "schema": benchmarks_api.ATLAS_STUDY_SCHEMA,
        "run_id": run_dir.name,
        "config": {
            "tracks": ["pfm_oracle", "photo_auto"],
            "atlas": {"radius_m": 500.0, "spacing_m": 50.0, "yaw_step_deg": 1.0},
        },
        "samples": [
            {
                "name": "manual-b",
                "manual": True,
                "compatibility": compatibility,
                "photo_edge_support": {"usable": True},
                "prior": {
                    "regime": "perturbed_metadata",
                    "constructed_from_reference_for_controlled_perturbation": True,
                    "errors": {"horizontal_position_m": 200.0, "yaw_deg": 15.0},
                },
                "tracks": {
                    "pfm_oracle": complete_track(200.0, 25.0, 900, winner_reaches=False, top_100_reaches=False),
                    "photo_auto": complete_track(80.0, 20.0, 80, winner_reaches=True, top_100_reaches=True),
                },
            },
            {
                "name": "automatic-b",
                "manual": False,
                "compatibility": compatibility,
                "photo_edge_support": {"usable": False},
                "prior": {
                    "regime": "perturbed_metadata",
                    "constructed_from_reference_for_controlled_perturbation": True,
                    "errors": {"horizontal_position_m": 200.0, "yaw_deg": 15.0},
                },
                "tracks": {
                    "pfm_oracle": complete_track(220.0, 50.0, 400, winner_reaches=False, top_100_reaches=True),
                    "photo_auto": {
                        "status": "evidence_rejected",
                        "evidence": {"available": False, "reason": "no_candidate"},
                        "estimator_archive": None,
                        "evaluation": None,
                    },
                },
            },
        ],
        "aggregates": [{"sentinel": "API recomputes these"}],
    }
    encoded = (json.dumps(results, sort_keys=True) + "\n").encode()
    results_sha256 = hashlib.sha256(encoded).hexdigest()
    dashboard_encoded = canonical_json_bytes(build_atlas_dashboard(results, results_sha256))
    (run_dir / "results.json").write_bytes(encoded)
    (run_dir / ATLAS_DASHBOARD_FILENAME).write_bytes(dashboard_encoded)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "schema": benchmarks_api.ATLAS_STUDY_SCHEMA,
                "run_id": run_dir.name,
                "status": "complete",
                "created_at": "2026-07-14T08:00:00+00:00",
                "results_sha256": results_sha256,
                "dashboard": {
                    "schema": ATLAS_DASHBOARD_SCHEMA,
                    "path": ATLAS_DASHBOARD_FILENAME,
                    "sha256": hashlib.sha256(dashboard_encoded).hexdigest(),
                    "source_results_sha256": results_sha256,
                    "size_bytes": len(dashboard_encoded),
                },
                "implementation": {
                    "git_revision": "0123456789abcdef",
                    "aggregate_sha256": "implementation-sha",
                    "tracked_source_diff_sha256": "diff-sha",
                    "source_worktree_status": [" M src/peakle/localize/skyline_atlas.py"],
                },
            }
        )
    )
    monkeypatch.setattr(benchmarks_api, "OUTPUT", output)
    monkeypatch.setattr(benchmarks_api, "ATLAS_CACHE_ROOT", tmp_path / "atlas-cache")

    def fail_dense_parse(_path: Path) -> dict:
        raise AssertionError("a declared compact sidecar must avoid parsing dense results")

    monkeypatch.setattr(benchmarks_api, "_parse_dense_atlas_results", fail_dense_parse)

    runs = client.get("/api/bench/runs").json()
    assert runs[0]["kind"] == "pose_atlas"
    assert runs[0]["artifact_schema"] == benchmarks_api.ATLAS_STUDY_SCHEMA
    assert runs[0]["track_count"] == 2
    assert runs[0]["completed_track_count"] == 3
    assert runs[0]["evidence_rejected_count"] == 1
    assert runs[0]["hash_verified"] is True
    assert runs[0]["dashboard_hash_verified"] is True
    assert runs[0]["compact_source"] == "artifact_sidecar"
    assert runs[0]["git_sha"] == "0123456789abcdef"
    assert runs[0]["implementation_sha256"] == "implementation-sha"
    assert runs[0]["dirty_code"] is True
    assert "recommended" not in runs[0]

    summary = client.get(f"/api/bench/runs/{run_dir.name}/summary").json()
    assert summary["mode"] == "atlas"
    assert summary["default_subset"] == "map_ab_height_a"
    assert summary["subsets"]["all"]["requested_track_count"] == 4
    assert summary["subsets"]["all"]["completed_track_count"] == 3
    pfm = next(row for row in summary["subsets"]["all"]["aggregates"] if row["track"] == "pfm_oracle")
    assert pfm["blind_winner_success_rate"] == 0.0
    assert pfm["full_lattice_oracle_success_rate"] == 1.0
    assert pfm["median_blind_winner_horizontal_m"] == 210.0
    assert pfm["median_full_lattice_oracle_horizontal_m"] == 37.5
    assert pfm["median_full_lattice_oracle_estimator_rank"] == 650.0
    photo = next(row for row in summary["subsets"]["all"]["aggregates"] if row["track"] == "photo_auto")
    assert photo["blind_winner_success_rate"] == 0.5
    assert photo["evidence_rejected"] == 1

    response = client.get(f"/api/bench/runs/{run_dir.name}/cases?subset=manual&evidence=pfm_oracle")
    cases = response.json()
    assert cases["mode"] == "atlas"
    assert cases["total"] == 1
    case = cases["rows"][0]
    assert case["blind_winner"]["errors"]["horizontal_position_m"] == 200.0
    assert case["prior_regime"] == benchmarks_api.ATLAS_PRIOR_REGIME
    assert case["prior_context"]["constructed_from_reference_for_controlled_perturbation"] is True
    assert case["prior_context"]["recorded_yaw_pitch_altitude_used_by_atlas"] is False
    assert case["full_lattice_oracle"]["errors"]["horizontal_position_m"] == 25.0
    assert case["full_lattice_oracle"]["estimator_rank"] == 900
    assert case["archive"]["full_lattice"]["hypothesis_count"] == 158760
    assert case["shortlist_top_100"]["reaches_target"] is False
    assert case["shortlist_first_reach"] is None
    public_json = json.dumps({"runs": runs, "summary": summary, "cases": cases})
    assert "yaw_scores" not in public_json
    assert "private_lattice_sentinel" not in public_json
    assert "private_candidate_sentinel" not in public_json
    assert "private_probe_sentinel" not in public_json
    assert client.get(f"/api/bench/runs/{run_dir.name}/cases?subset=primary").status_code == 400

    # The sidecar is part of the immutable run commit.  Once its bytes no
    # longer match run.json, discovery fails closed instead of parsing the
    # 111 MB dense result as a silent fallback.
    (run_dir / ATLAS_DASHBOARD_FILENAME).write_bytes(dashboard_encoded + b" ")
    assert client.get("/api/bench/runs").json() == []
    assert client.get(f"/api/bench/runs/{run_dir.name}").status_code == 404


def test_pose_atlas_legacy_projection_cache_is_content_addressed(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "legacy-pose-atlas"
    run_dir.mkdir()
    results = {
        "schema": benchmarks_api.ATLAS_STUDY_SCHEMA,
        "run_id": run_dir.name,
        "config": {"tracks": []},
        "samples": [],
    }
    encoded = canonical_json_bytes(results)
    results_path = run_dir / "results.json"
    results_path.write_bytes(encoded)
    metadata = {
        "schema": benchmarks_api.ATLAS_STUDY_SCHEMA,
        "run_id": run_dir.name,
        "results_sha256": hashlib.sha256(encoded).hexdigest(),
    }
    cache_root = tmp_path / "cache"
    monkeypatch.setattr(benchmarks_api, "ATLAS_CACHE_ROOT", cache_root)
    benchmarks_api._read_atlas_artifact_cached.cache_clear()

    generated = benchmarks_api._read_atlas_artifact(results_path, metadata)

    assert generated is not None
    assert generated.compact_source == "results_fallback"
    cache_path = benchmarks_api._atlas_cache_path(cache_root, generated.results_sha256)
    assert cache_path.is_file()

    benchmarks_api._read_atlas_artifact_cached.cache_clear()

    def fail_dense_parse(_path: Path) -> dict:
        raise AssertionError("the content-addressed compact cache should be reused")

    monkeypatch.setattr(benchmarks_api, "_parse_dense_atlas_results", fail_dense_parse)
    cached = benchmarks_api._read_atlas_artifact(results_path, metadata)

    assert cached is not None
    assert cached.compact_source == "content_cache"
    assert cached.payload == generated.payload

    # Cache keys are not allowed to override the digest committed by run.json.
    results_path.write_bytes(encoded + b" ")
    benchmarks_api._read_atlas_artifact_cached.cache_clear()
    assert benchmarks_api._read_atlas_artifact(results_path, metadata) is None


def test_matrix_default_prefers_primary_over_excluded_map_diagnostic() -> None:
    base = {
        "manual": True,
        "evidence_track": "pfm_oracle",
        "prior_regime": "perturbed_metadata",
        "status": "ok",
        "runtime_s": 1.0,
        "errors": {"horizontal_position_m": 200.0, "yaw_deg": 15.0, "pitch_deg": None},
        "success": {"value": False},
        "baseline": None,
        "photo_edge_supported": True,
    }
    payload = {
        "rows": [
            {
                "name": "diagnostic-a",
                "gt_dem_compatibility": {"tier": "MAP_A", "height": {"tier": "HEIGHT_A"}},
            },
            {
                "name": "primary-b",
                "gt_dem_compatibility": {"tier": "MAP_B", "height": {"tier": "HEIGHT_B"}},
            },
        ],
        "matrix_cases": [
            {
                **base,
                "id": "diagnostic-a:global",
                "name": "diagnostic-a",
                "algorithm": "global",
                "compatibility_tier": "MAP_A",
                "ranking_eligible": False,
                "ranking_exclusions": ["regional_global_is_unvalidated"],
            },
            {
                **base,
                "id": "primary-b:cmaes",
                "name": "primary-b",
                "algorithm": "cmaes",
                "compatibility_tier": "MAP_B",
                "ranking_eligible": True,
                "ranking_exclusions": [],
            },
        ],
    }
    run = {"kind": "strategy_matrix", "dirty_code": False, "primary_case_count": 1}

    summary = benchmarks_api._matrix_summary(run, payload)

    assert summary["subsets"]["map_a_height_a"]["attempted_case_count"] == 1
    assert summary["subsets"]["primary"]["attempted_case_count"] == 1
    assert summary["default_subset"] == "primary"


def test_photo_upload_creates_photo_backed_view(
    client: TestClient, scene: Scene, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_create_photo_view(
        scene: Scene, image_bytes: bytes, lat: float, lon: float, fov: float, eye: float, label: str | None
    ):
        photo = Image.open(BytesIO(image_bytes)).convert("RGB")
        intrinsics = build_intrinsics(photo.width, photo.height, fov)
        extrinsics = CameraExtrinsics(
            position=LocalPoint(east_m=0.0, north_m=0.0, up_m=scene.terrain.elevation_at(0.0, 0.0) + eye),
            yaw_deg=0.0,
            pitch_deg=0.0,
            roll_deg=0.0,
        )
        contour = SkylineContour(
            image_width_px=photo.width,
            image_height_px=photo.height,
            points=[ImagePoint(x_px=float(x), y_px=4.0) for x in range(photo.width)],
            source="photo",
        )
        return scene.add_backed_view(intrinsics, extrinsics, contour, photo, source="photo", label=label)

    monkeypatch.setattr(views_api, "_create_photo_view", fake_create_photo_view)
    buffer = BytesIO()
    Image.new("RGB", (16, 8), (150, 190, 220)).save(buffer, format="PNG")

    created = client.post(
        "/api/views/from-photo?lat_deg=46.1&lon_deg=7.8&horizontal_fov_deg=55&eye_height_m=2&label=Phone",
        content=buffer.getvalue(),
        headers={"content-type": "image/png"},
    )

    assert created.status_code == 201
    view = created.json()
    assert view["source"] == "photo"
    assert view["label"] == "Phone"
    assert view["photo_url"] == f"/api/views/{view['id']}/photo"
    assert view["prior"]["yaw_sigma_deg"] == 120.0
    assert view["image_camera"]["width_px"] == 16
    assert view["image_camera"]["height_px"] == 8
    assert view["image_camera"]["horizontal_fov_deg"] == pytest.approx(55.0)
    assert view["image_camera"]["projection"] == "pinhole"


def test_photo_upload_rejects_oversized_body(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_create_photo_view(*args: object) -> None:
        pytest.fail("_create_photo_view should not run for oversized uploads")

    monkeypatch.setattr(views_api, "_create_photo_view", fail_create_photo_view)

    response = client.post(
        "/api/views/from-photo?lat_deg=46.1&lon_deg=7.8&horizontal_fov_deg=55",
        content=b"x" * (views_api.PHOTO_MAX_UPLOAD_BYTES + 1),
        headers={"content-type": "image/jpeg"},
    )

    assert response.status_code == 413
    assert "12 MB" in response.json()["detail"]


def test_config_rebuild_clears_views(client: TestClient) -> None:
    terrain = client.get("/api/terrain").json()
    east = (terrain["x_min_m"] + terrain["x_max_m"]) / 2
    client.post(
        "/api/views", json={"east_m": east, "north_m": terrain["y_min_m"] + 100, "yaw_deg": 0.0, "pitch_deg": 2.0}
    )
    assert len(client.get("/api/views").json()) == 1

    rebuilt = client.put(
        "/api/scene/config",
        json={
            "provider": "demo",
            "seed": 7,
            "image_width": 320,
            "image_height": 180,
            "horizontal_fov_deg": 55.0,
            "default_strategy": "powell",
        },
    )
    assert rebuilt.status_code == 200
    assert rebuilt.json()["config"]["seed"] == 7
    assert client.get("/api/views").json() == []


def test_unknown_view_returns_404(client: TestClient) -> None:
    assert client.get("/api/views/view-99").status_code == 404
    assert client.post("/api/views/view-99/solves", json={"strategy": "powell"}).status_code == 404


def test_gt_rebuild_endpoint_is_not_registered(client: TestClient) -> None:
    assert client.get("/api/gt/rebuild").status_code == 404

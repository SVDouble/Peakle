"""Workbench API integration tests."""

from __future__ import annotations

import json
import time
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from peakle.domain.camera import CameraExtrinsics
from peakle.domain.contours import ImagePoint, SkylineContour
from peakle.domain.coordinates import LocalPoint
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
            }
        )
    )
    monkeypatch.setattr(benchmarks_api, "OUTPUT", output)

    runs = client.get("/api/bench/runs").json()
    assert runs[0]["kind"] == "strategy_matrix"
    assert runs[0]["algorithm_count"] == 2
    assert runs[0]["recommended"] is True
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

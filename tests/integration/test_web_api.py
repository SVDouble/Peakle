"""Workbench API integration tests."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from peakle.scene.scene import Scene
from peakle.web.app import create_app


@pytest.fixture
def client(scene: Scene) -> TestClient:
    return TestClient(create_app(scene))


def test_scene_terrain_peaks_endpoints(client: TestClient) -> None:
    scene = client.get("/api/scene").json()
    assert scene["providers"] == ["demo"]
    assert {strategy["name"] for strategy in scene["strategies"]} == {"powell", "nelder", "evolution"}
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

    image = client.get(f"/api/views/{view_id}/image")
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"

    patched = client.patch(f"/api/views/{view_id}", json={"label": "Ridge"})
    assert patched.json()["label"] == "Ridge"

    solved = client.post(f"/api/views/{view_id}/solves", json={"strategy": "nelder", "params": {"seed": 1}})
    assert solved.status_code == 201
    solve = solved.json()
    assert solve["strategy"] == "nelder"
    assert solve["result"]["trace"]
    # JSON-safe: no NaN/Infinity leaked into the payload.
    json.loads(solved.text)

    solve_id = solve["id"]
    assert client.get(f"/api/views/{view_id}/solves/{solve_id}").status_code == 200
    assert len(client.get(f"/api/views/{view_id}/solves").json()) == 1


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

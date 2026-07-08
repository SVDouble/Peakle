"""Workbench API integration tests."""

from __future__ import annotations

import json
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from peakle.domain.camera import CameraExtrinsics
from peakle.domain.contours import ImagePoint, SkylineContour
from peakle.domain.coordinates import LocalPoint
from peakle.scene.scene import Scene
from peakle.scene.state import build_intrinsics
from peakle.web.api import views as views_api
from peakle.web.app import create_app


@pytest.fixture
def client(scene: Scene) -> TestClient:
    return TestClient(create_app(scene))


def test_scene_terrain_peaks_endpoints(client: TestClient) -> None:
    scene = client.get("/api/scene").json()
    assert scene["providers"] == ["demo", "srtm"]
    assert {strategy["name"] for strategy in scene["strategies"]} == {
        "powell",
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

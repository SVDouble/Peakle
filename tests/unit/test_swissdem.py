from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
from PIL import Image

from peakle.localize import swissdem


class _JsonResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> _JsonResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def _item(*filenames: str, updated: str = "2024-01-01T00:00:00Z") -> dict[str, object]:
    return {
        "properties": {"updated": updated},
        "assets": {
            f"asset-{index}": {"href": f"https://download.example.test/{filename}"}
            for index, filename in enumerate(filenames)
        },
    }


def _request_url(request: str | urllib.request.Request) -> str:
    return request if isinstance(request, str) else request.full_url


def test_ensure_swiss_tiles_follows_next_pages_and_fetches_only_newest_editions(tmp_path: Path, monkeypatch) -> None:
    next_url = f"{swissdem.STAC_ITEMS}?cursor=page-2"
    pages: list[dict[str, object]] = [
        {
            "features": [
                _item("swissalti3d_2019_2615-1090_2_2056_5728.tif"),
                _item("swissalti3d_2022_2616-1090_2_2056_5728.tif"),
            ],
            "links": [{"rel": "next", "href": "?cursor=page-2"}],
        },
        {
            "features": [
                _item("swissalti3d_2024_2615-1090_2_2056_5728.tif"),
                _item("swissalti3d_2020_2617-1090_2_2056_5728.tif"),
                _item("swissalti3d_2024_2618-1090_0.5_2056_5728.tif"),
            ],
            "links": [],
        },
    ]
    opened: list[str] = []
    downloads: list[str] = []

    def fake_urlopen(request: str | urllib.request.Request, *, timeout: int) -> _JsonResponse:
        assert timeout == 30
        opened.append(_request_url(request))
        return _JsonResponse(pages[len(opened) - 1])

    def fake_urlretrieve(href: str, destination: str | Path):
        downloads.append(href)
        Path(destination).write_bytes(b"mock GeoTIFF")
        return str(destination), None

    monkeypatch.setattr(swissdem.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(swissdem.urllib.request, "urlretrieve", fake_urlretrieve)

    present = swissdem.ensure_swiss_tiles(tmp_path, lat=46.0, lon=8.0, radius_m=4500.0)

    initial = urlsplit(opened[0])
    assert initial.path.endswith("/api/stac/v1/collections/ch.swisstopo.swissalti3d/items")
    assert parse_qs(initial.query)["limit"] == ["100"]
    assert opened[1] == next_url
    assert present == 3
    assert {Path(urlsplit(href).path).name for href in downloads} == {
        "swissalti3d_2024_2615-1090_2_2056_5728.tif",
        "swissalti3d_2022_2616-1090_2_2056_5728.tif",
        "swissalti3d_2020_2617-1090_2_2056_5728.tif",
    }
    assert not (tmp_path / "swissalti3d_2019_2615-1090_2_2056_5728.tif").exists()


def test_ensure_swiss_tiles_reuses_any_cached_edition_for_a_coordinate(tmp_path: Path, monkeypatch) -> None:
    cached = tmp_path / "swissalti3d_2019_2615-1090_2_2056_5728.tif"
    cached.write_bytes(b"existing cache remains compatible")

    def fake_urlopen(request: str | urllib.request.Request, *, timeout: int) -> _JsonResponse:
        return _JsonResponse(
            {
                "features": [_item("swissalti3d_2024_2615-1090_2_2056_5728.tif")],
                "links": [],
            }
        )

    def fail_urlretrieve(_href: str, _destination: str | Path):
        raise AssertionError("an already cached LV95 coordinate must not be downloaded again")

    monkeypatch.setattr(swissdem.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(swissdem.urllib.request, "urlretrieve", fail_urlretrieve)

    assert swissdem.ensure_swiss_tiles(tmp_path, lat=46.0, lon=8.0) == 1
    assert cached.exists()


def test_load_swiss_patch_uses_newest_cached_edition_deterministically(tmp_path: Path) -> None:
    lat, lon = 46.0, 8.0
    east_m, north_m = swissdem._lv95(lat, lon)
    east_km, north_km = int(east_m // 1000), int(north_m // 1000)
    newest = tmp_path / f"swissalti3d_2024_{east_km}-{north_km}_2_2056_5728.tif"
    oldest = tmp_path / f"swissalti3d_2019_{east_km}-{north_km}_2_2056_5728.tif"

    # Create the newest file first and the stale duplicate second: filesystem iteration order
    # must not decide which edition overwrites the mosaic.
    Image.fromarray(np.full((500, 500), 2024.0, dtype=np.float32)).save(newest)
    Image.fromarray(np.full((500, 500), 2019.0, dtype=np.float32)).save(oldest)

    patch = swissdem.load_swiss_patch(tmp_path, lat, lon, res=100.0, radius_m=100.0)

    assert patch is not None
    finite = patch.elevation_m[np.isfinite(patch.elevation_m)]
    assert finite.size > 0
    assert np.allclose(finite, 2024.0)


def test_bilinear_resampling_never_blends_nodata_into_a_finite_elevation() -> None:
    mosaic = np.array([[np.nan, 1000.0], [1000.0, 1000.0]], dtype=np.float32)
    coordinates = [np.array([0.75]), np.array([0.75])]

    result = swissdem._resample_with_full_finite_support(mosaic, coordinates, (1,))

    # Directly interpolating a -9999 sentinel would produce a superficially valid ~312 m value.
    assert np.isnan(result[0])


def test_bilinear_resampling_preserves_fully_supported_values() -> None:
    mosaic = np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float32)
    coordinates = [np.array([0.5]), np.array([0.5])]

    result = swissdem._resample_with_full_finite_support(mosaic, coordinates, (1,))

    assert result[0] == 250.0

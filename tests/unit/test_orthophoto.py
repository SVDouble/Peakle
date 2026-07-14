from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from peakle.domain.coordinates import GeoPoint, LocalFrame
from peakle.rendering.orthophoto import (
    DEFAULT_SWISSIMAGE_URL,
    LocalRasterTexture,
    MissingOrthophotoTiles,
    OrthophotoMosaic,
    SwissImageProvider,
)


def test_local_raster_texture_samples_local_enu_and_masks_outside() -> None:
    x_m = np.asarray([-10.0, 0.0, 10.0])
    y_m = np.asarray([-10.0, 0.0, 10.0])
    x_grid, y_grid = np.meshgrid(x_m, y_m)
    rgb = np.stack((x_grid + 20.0, y_grid + 20.0, np.full_like(x_grid, 50.0)), axis=-1).astype(np.uint8)
    texture = LocalRasterTexture(x_m=x_m, y_m=y_m, rgb=rgb, source="unit-test")

    sampled, valid = texture.sample_local(
        np.asarray([0.0, 5.0, 50.0]),
        np.asarray([0.0, -5.0, 0.0]),
    )

    assert valid.tolist() == [True, True, False]
    assert sampled[0].tolist() == [20, 20, 50]
    assert sampled[1].tolist() == [25, 15, 50]
    assert sampled[2].tolist() == [0, 0, 0]
    assert texture.provenance()["source"] == "unit-test"


def test_local_raster_rejects_nodata_interpolation_and_hashes_georeferencing() -> None:
    rgb = np.full((2, 2, 3), 120, dtype=np.uint8)
    valid_mask = np.asarray([[True, False], [True, False]])
    texture = LocalRasterTexture(
        x_m=np.asarray([0.0, 1.0]),
        y_m=np.asarray([0.0, 1.0]),
        rgb=rgb,
        valid_mask=valid_mask,
        content_sha256="A" * 64,
    )

    sampled, valid = texture.sample_local(
        np.asarray([0.0, 0.25, np.nan]),
        np.asarray([0.5, 0.5, 0.5]),
    )
    shifted_axis = LocalRasterTexture(
        x_m=np.asarray([0.0, 2.0]),
        y_m=np.asarray([0.0, 1.0]),
        rgb=rgb,
        valid_mask=valid_mask,
    )
    no_mask = LocalRasterTexture(
        x_m=np.asarray([0.0, 1.0]),
        y_m=np.asarray([0.0, 1.0]),
        rgb=rgb,
    )
    provenance = texture.provenance()

    assert valid.tolist() == [True, False, False]
    assert sampled[0].tolist() == [120, 120, 120]
    assert sampled[1:].tolist() == [[0, 0, 0], [0, 0, 0]]
    assert provenance["source_content_sha256"] == "a" * 64
    assert provenance["content_sha256"] != shifted_axis.provenance()["content_sha256"]
    assert provenance["content_sha256"] != no_mask.provenance()["content_sha256"]


def test_xyz_sampling_converts_global_pixel_edges_to_array_centres() -> None:
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    rgb[:, 128, 0] = 200
    rgb[127, :, 1] = 20
    rgb[128, :, 1] = 220
    rgb[..., 2] = 40
    mosaic = OrthophotoMosaic(
        frame=LocalFrame(origin=GeoPoint(latitude_deg=0.0, longitude_deg=0.0, elevation_m=0.0)),
        rgb=rgb,
        zoom=0,
        origin_global_x_px=0,
        origin_global_y_px=0,
        layer="test-layer",
        time="test-time",
        tile_records=(),
        service_url_template=DEFAULT_SWISSIMAGE_URL,
    )

    sampled, valid = mosaic.sample_local(np.asarray([0.0]), np.asarray([0.0]))
    provenance = mosaic.provenance()

    # lon=0, lat=0 is global edge coordinate (128, 128), halfway between
    # ndarray sample centres 127 and 128 on each axis.
    assert valid.tolist() == [True]
    assert sampled[0].tolist() == [100, 120, 40]
    assert provenance["mosaic_origin_global_pixel_edge"] == [0, 0]
    assert len(provenance["mosaic_rgb_sha256"]) == 64


def test_swissimage_is_offline_by_default_and_content_addresses_cached_tiles(tmp_path) -> None:
    frame = LocalFrame(origin=GeoPoint(latitude_deg=46.0, longitude_deg=8.0, elevation_m=0.0))
    provider = SwissImageProvider(cache_dir=tmp_path, zoom=15, download_missing=False, max_tiles=16)

    with pytest.raises(MissingOrthophotoTiles) as caught:
        provider.mosaic_for_local_bounds(
            frame,
            east_min_m=-100.0,
            east_max_m=100.0,
            north_min_m=-100.0,
            north_max_m=100.0,
        )

    assert caught.value.missing
    for index, path in enumerate(caught.value.missing):
        path.parent.mkdir(parents=True, exist_ok=True)
        tile = np.full((256, 256, 3), (30 + index, 80, 120), dtype=np.uint8)
        Image.fromarray(tile, mode="RGB").save(path, format="JPEG", quality=95)

    mosaic = provider.mosaic_for_local_bounds(
        frame,
        east_min_m=-100.0,
        east_max_m=100.0,
        north_min_m=-100.0,
        north_max_m=100.0,
    )
    sampled, valid = mosaic.sample_local(np.asarray([0.0]), np.asarray([0.0]))
    provenance = mosaic.provenance()

    assert valid.tolist() == [True]
    assert sampled.shape == (1, 3)
    assert provenance["tile_count"] == caught.value.requested
    assert len(provenance["tile_manifest_sha256"]) == 64

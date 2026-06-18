"""Synthetic rasterizer tests."""

import numpy as np

from peakle.rendering.rasterizer import SyntheticRenderer, _ScreenVertex


def test_rasterizer_prefers_nearer_triangle_for_overlapping_pixels() -> None:
    """A hidden projected triangle must not overwrite nearer visible terrain."""

    renderer = SyntheticRenderer()
    inverse_depth_buffer = np.full((12, 12), -np.inf, dtype=np.float64)
    far_triangle = (
        _ScreenVertex(u_px=2.0, v_px=2.0, inverse_depth=0.1, valid=True),
        _ScreenVertex(u_px=10.0, v_px=2.0, inverse_depth=0.1, valid=True),
        _ScreenVertex(u_px=6.0, v_px=10.0, inverse_depth=0.1, valid=True),
    )
    near_triangle = (
        _ScreenVertex(u_px=2.0, v_px=2.0, inverse_depth=0.4, valid=True),
        _ScreenVertex(u_px=10.0, v_px=2.0, inverse_depth=0.4, valid=True),
        _ScreenVertex(u_px=6.0, v_px=10.0, inverse_depth=0.4, valid=True),
    )

    renderer._rasterize_triangle(inverse_depth_buffer, far_triangle)
    renderer._rasterize_triangle(inverse_depth_buffer, near_triangle)

    assert np.isclose(inverse_depth_buffer[5, 6], 0.4)

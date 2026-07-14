"""Photo skyline extraction regressions."""

from __future__ import annotations

import numpy as np

from peakle.localize import extract
from peakle.localize.extract import best_skyline_candidate, extract_candidates, extract_skyline


def test_title_text_in_black_border_does_not_seed_skyline() -> None:
    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    rgb[30:100, :] = (90, 150, 220)  # real sky in the cylindrical crop
    rgb[70:100, :] = (120, 120, 120)  # mountain/terrain below the skyline
    rgb[5:12, 20:80] = (80, 130, 220)  # blue title text in the black top border

    rows = extract_skyline(rgb).rows

    assert np.nanmedian(rows[25:75]) >= 68.0


def test_extract_candidates_uses_segmenter_before_contours(monkeypatch) -> None:
    class FakeSegmenter:
        name = "sam3"

        def sky_mask(self, rgb):
            mask = np.zeros(rgb.shape[:2], dtype=bool)
            mask[:70, :] = True
            return mask

    monkeypatch.setattr(extract, "_available_segmenter", lambda _kind: FakeSegmenter())
    rgb = np.full((120, 160, 3), (110, 145, 180), dtype=np.uint8)
    rgb[70:, :] = (100, 100, 100)

    candidates = extract_candidates(rgb, backend="sam3")
    selected = best_skyline_candidate(candidates)

    assert "sam3" in candidates
    assert selected is not None
    assert selected[0] == "sam3"
    assert np.nanmedian(selected[1].rows) == 70.0


def test_color_candidates_do_not_load_optional_models(monkeypatch) -> None:
    def fail_if_loaded(_kind):
        raise AssertionError("the hermetic color backend loaded an optional model")

    monkeypatch.setattr(extract, "_available_segmenter", fail_if_loaded)
    monkeypatch.setattr(extract, "learned_skyline", lambda _rgb: (_ for _ in ()).throw(AssertionError))
    rgb = np.full((80, 120, 3), (90, 140, 190), dtype=np.uint8)
    rgb[45:, :] = (100, 100, 100)

    candidates = extract_candidates(rgb, backend="color")

    assert set(candidates) == {"color", "blue", "bright"}

"""Synthetic peak detection and naming."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter, percentile_filter

from peakle.domain.peaks import Peak, PeakDetectionSpec
from peakle.domain.terrain import TerrainMap

_ADJECTIVES = [
    "Aster",
    "Brisk",
    "Clear",
    "Dawn",
    "Elder",
    "Frost",
    "Granite",
    "High",
    "Ivory",
    "Juniper",
    "Keen",
    "Lumen",
    "Mica",
    "North",
    "Onyx",
    "Pale",
    "Quartz",
    "Raven",
]

_NOUNS = [
    "Horn",
    "Spire",
    "Ridge",
    "Crown",
    "Needle",
    "Summit",
    "Pillar",
    "Dome",
    "Pointe",
    "Crag",
    "Fang",
    "Sattel",
    "Kopf",
    "Tower",
    "Gipfel",
    "Peak",
    "Col",
    "Hoch",
]


class PeakDetector:
    """Detects synthetic peaks from a generated elevation grid.

    Args:
        spec: Peak detection parameters.
    """

    def __init__(self, spec: PeakDetectionSpec) -> None:
        self.spec = spec

    def detect(self, terrain: TerrainMap) -> list[Peak]:
        """Finds and names prominent local maxima.

        Args:
            terrain: Generated terrain map.

        Returns:
            Peaks sorted by synthetic prominence and elevation.
        """

        window = self.spec.neighborhood_px
        if window % 2 == 0:
            window += 1

        elevation = terrain.elevation_m
        local_max = elevation == maximum_filter(elevation, size=window, mode="nearest")
        border = max(window // 2, self.spec.prominence_radius_px)
        local_max[:border, :] = False
        local_max[-border:, :] = False
        local_max[:, :border] = False
        local_max[:, -border:] = False

        radius = self.spec.prominence_radius_px
        local_floor = percentile_filter(
            elevation,
            percentile=35.0,
            size=radius * 2 + 1,
            mode="nearest",
        )
        prominence = elevation - local_floor
        candidate_mask = local_max & (prominence >= self.spec.min_prominence_m)
        rows, cols = np.where(candidate_mask)
        if rows.size == 0:
            return []

        candidate_prominence = prominence[rows, cols]
        candidate_elevation = elevation[rows, cols]
        order = np.lexsort((cols, rows, -candidate_elevation, -candidate_prominence))

        peaks: list[Peak] = []
        for index, candidate_index in enumerate(
            order[: self.spec.max_peaks],
            start=1,
        ):
            row = int(rows[candidate_index])
            col = int(cols[candidate_index])
            peaks.append(
                Peak(
                    id=f"peak-{index:03d}",
                    name=self._name_for(index),
                    local_position=terrain.point_at_index(row, col),
                    geo_position=terrain.geo_at_index(row, col),
                    elevation_m=float(candidate_elevation[candidate_index]),
                    prominence_m=float(candidate_prominence[candidate_index]),
                )
            )
        return peaks

    def _name_for(self, index: int) -> str:
        adjective = _ADJECTIVES[(index - 1) % len(_ADJECTIVES)]
        noun = _NOUNS[((index - 1) * 7) % len(_NOUNS)]
        return f"{adjective} {noun}"

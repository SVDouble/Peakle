"""Map providers.

A `MapProvider` supplies the elevation map a scene is built on. Only the
synthetic `DemoMapProvider` exists today; a real provider (DEM tiles, etc.) can
implement the same protocol later without touching the scene or solver code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from peakle.domain.terrain import TerrainMap, TerrainSpec
from peakle.terrain.dem import DEFAULT_DEM_DIR, find_hgt_tile, load_dem_terrain
from peakle.terrain.generator import TerrainGenerator

ProviderKind = Literal["demo", "srtm"]
PROVIDER_KINDS: tuple[ProviderKind, ...] = ("demo", "srtm")


@runtime_checkable
class MapProvider(Protocol):
    """Supplies an elevation map for a scene."""

    kind: str

    def generate(self) -> TerrainMap:
        """Builds the elevation map."""
        ...


class DemoMapProvider:
    """Synthetic terrain provider driven by a `TerrainSpec` (seed selectable).

    Args:
        spec: Terrain generation specification; `spec.seed` chooses the map.
    """

    kind = "demo"

    def __init__(self, spec: TerrainSpec) -> None:
        self.spec = spec

    def generate(self) -> TerrainMap:
        """Generates synthetic terrain for the configured spec."""

        return TerrainGenerator(self.spec).generate()


class SrtmMapProvider:
    """Real-world terrain from a downloaded `.hgt` DEM tile.

    Crops a `spec`-sized window (centred on the tile's highest point) from the
    first `.hgt` tile found in `dem_dir`. The tile's geographic location, not the
    spec origin, determines where on Earth the scene sits.

    Args:
        spec: Terrain spec providing the crop extent and grid resolution.
        dem_dir: Directory holding `.hgt` tiles (default: ``data/dem_samples``).
    """

    kind = "srtm"

    def __init__(self, spec: TerrainSpec, dem_dir: Path = DEFAULT_DEM_DIR) -> None:
        self.spec = spec
        self.dem_dir = Path(dem_dir)

    def generate(self) -> TerrainMap:
        """Loads and crops real elevations from the available DEM tile."""

        return load_dem_terrain(self.spec, find_hgt_tile(self.dem_dir))


def build_provider(kind: ProviderKind, spec: TerrainSpec) -> MapProvider:
    """Builds a map provider by kind.

    Args:
        kind: Provider identifier.
        spec: Terrain specification for synthetic providers.

    Returns:
        A map provider instance.
    """

    if kind == "demo":
        return DemoMapProvider(spec)
    if kind == "srtm":
        return SrtmMapProvider(spec)
    msg = f"unknown map provider {kind!r}; expected one of {PROVIDER_KINDS}"
    raise ValueError(msg)

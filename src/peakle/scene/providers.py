"""Map providers.

A `MapProvider` supplies the elevation map a scene is built on. Only the
synthetic `DemoMapProvider` exists today; a real provider (DEM tiles, etc.) can
implement the same protocol later without touching the scene or solver code.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from peakle.domain.terrain import TerrainMap, TerrainSpec
from peakle.terrain.generator import TerrainGenerator

ProviderKind = Literal["demo"]
PROVIDER_KINDS: tuple[ProviderKind, ...] = ("demo",)


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
    msg = f"unknown map provider {kind!r}; expected one of {PROVIDER_KINDS}"
    raise ValueError(msg)

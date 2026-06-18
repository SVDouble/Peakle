"""Request schemas for the workbench API."""

from __future__ import annotations

from typing import Any, Literal, get_args

from pydantic import BaseModel, Field

from peakle.optimization.solve import STRATEGIES
from peakle.scene.providers import PROVIDER_KINDS

StrategyName = Literal["powell", "nelder", "evolution"]
ProviderName = Literal["demo"]
assert set(STRATEGIES) == set(get_args(StrategyName))  # noqa: S101 - keep schema and solver list in sync
assert set(PROVIDER_KINDS) == set(get_args(ProviderName))  # noqa: S101 - keep schema and provider list in sync


class SceneConfigRequest(BaseModel):
    """Config-panel request to (re)build the scene.

    Attributes:
        provider: Map provider kind.
        seed: Map seed.
        image_width: Render width in pixels.
        image_height: Render height in pixels.
        horizontal_fov_deg: Camera horizontal field of view.
        default_strategy: Default solver strategy.
    """

    provider: ProviderName = "demo"
    seed: int = Field(ge=0)
    image_width: int = Field(ge=160, le=4096)
    image_height: int = Field(ge=120, le=4096)
    horizontal_fov_deg: float = Field(gt=1.0, lt=179.0)
    default_strategy: StrategyName = "powell"


class ViewCreateRequest(BaseModel):
    """Request to place a camera and create a view.

    Attributes:
        east_m: Camera easting in meters.
        north_m: Camera northing in meters.
        yaw_deg: Heading in degrees.
        pitch_deg: Tilt in degrees.
        eye_height_m: Camera height above the terrain surface.
        label: Optional display label.
    """

    east_m: float
    north_m: float
    yaw_deg: float = Field(ge=-360.0, le=360.0)
    pitch_deg: float = Field(ge=-89.0, le=89.0)
    eye_height_m: float = Field(ge=0.0, le=5000.0, default=150.0)
    label: str | None = None


class ViewPatchRequest(BaseModel):
    """Request to edit a view's pose or label (all fields optional)."""

    east_m: float | None = None
    north_m: float | None = None
    yaw_deg: float | None = Field(default=None, ge=-360.0, le=360.0)
    pitch_deg: float | None = Field(default=None, ge=-89.0, le=89.0)
    eye_height_m: float | None = Field(default=None, ge=0.0, le=5000.0)
    label: str | None = None


class SolveRequest(BaseModel):
    """Request to run a solver against a view.

    Attributes:
        strategy: Solver strategy to run.
        params: Optional solver parameters (e.g. `seed`).
    """

    strategy: StrategyName
    params: dict[str, Any] = Field(default_factory=dict)

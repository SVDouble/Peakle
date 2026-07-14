"""Request schemas for the workbench API."""

from __future__ import annotations

from typing import Any, Literal, get_args

from pydantic import BaseModel, Field

from peakle.optimization.solve import STRATEGIES
from peakle.scene.providers import PROVIDER_KINDS

StrategyName = Literal["horizon", "contourdb", "cmaes", "powell", "nelder", "evolution", "global"]
ProviderName = Literal["demo", "srtm"]
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
    default_strategy: StrategyName = "horizon"


class SceneFocusRequest(BaseModel):
    """Request to recenter the map on a geographic point (clears views).

    Attributes:
        lat_deg: Latitude of the new map centre.
        lon_deg: Longitude of the new map centre.
        extent_m: Square window size in meters.
    """

    lat_deg: float = Field(ge=-90.0, le=90.0)
    lon_deg: float = Field(ge=-180.0, le=180.0)
    extent_m: float = Field(default=40000.0, ge=2000.0, le=120000.0)


class ViewCreateRequest(BaseModel):
    """Request to place a baseline pose and create a view.

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
        params: Optional solver parameters (e.g. `seed`, `position_prior`,
            `orientation_prior`, `prior_source`, `evidence_source`). `prior_source`
            may be `metadata`, `pose:truth`, or `pose:solve:<id>`. Evidence choices
            are exposed by the view payload; GT views default to `photo_auto`, while
            `pfm_oracle` must be selected explicitly.
    """

    strategy: StrategyName
    params: dict[str, Any] = Field(default_factory=dict)


class JobCreateRequest(BaseModel):
    """Request to enqueue background solving for loaded or catalogue views."""

    view_ids: list[str] = Field(default_factory=list, max_length=2000)
    strategy: StrategyName | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    max_workers: int = Field(default=2, ge=1, le=8)

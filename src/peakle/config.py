"""Application settings loaded from YAML and environment overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict, YamlConfigSettingsSource

from peakle.domain.peaks import PeakDetectionSpec
from peakle.domain.terrain import TerrainSpec

DEFAULT_SETTINGS_FILE = Path(__file__).with_name("default_settings.yaml")


class RenderSettings(BaseModel):
    """Synthetic render settings."""

    image_width: int = Field(ge=160)
    image_height: int = Field(ge=120)
    horizontal_fov_deg: float = Field(gt=1.0, lt=179.0)


class OptimizationSettings(BaseModel):
    """Pose optimization settings."""

    max_iterations: int = Field(ge=1)
    objective_terrain_stride: int = Field(ge=1)


class DemoCameraSettings(BaseModel):
    """Synthetic demo camera placement settings."""

    east_offset_fraction: float
    north_offset_fraction: float
    overlook_height_m: float = Field(gt=0.0)
    view_count: int = Field(ge=1)
    view_spacing_fraction: float = Field(ge=0.0)


class PoseNoiseSettings(BaseModel):
    """Synthetic pose-prior noise and uncertainty settings."""

    horizontal_noise_m: float = Field(ge=0.0)
    vertical_noise_m: float = Field(ge=0.0)
    yaw_noise_deg: float = Field(ge=0.0)
    pitch_noise_deg: float = Field(ge=0.0)
    horizontal_sigma_m: float = Field(gt=0.0)
    vertical_sigma_m: float = Field(gt=0.0)
    yaw_sigma_deg: float = Field(gt=0.0)
    pitch_sigma_deg: float = Field(gt=0.0)


class WebSettings(BaseModel):
    """Local static viewer server settings."""

    host: str
    port: int = Field(ge=1, le=65535)


class AppSettings(BaseSettings):
    """Application settings with YAML defaults and environment overrides.

    `src/peakle/default_settings.yaml` is the baseline source. Set
    `PEAKLE_CONFIG_FILE` or call `load_settings(config_file=...)` to use another
    YAML file. Nested values can be overridden with environment variables such as
    `PEAKLE_RENDER__IMAGE_WIDTH=1280`.
    """

    model_config = SettingsConfigDict(
        env_prefix="PEAKLE_",
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    _config_file_override: ClassVar[Path | None] = None

    artifact_dir: Path
    random_seed: int
    terrain: TerrainSpec
    peak_detection: PeakDetectionSpec
    render: RenderSettings
    optimization: OptimizationSettings
    camera: DemoCameraSettings
    pose_noise: PoseNoiseSettings
    web: WebSettings
    log_level: str

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Adds the YAML file as the baseline source."""

        yaml_settings = YamlConfigSettingsSource(
            settings_cls,
            yaml_file=_settings_file_path(),
        )
        return init_settings, env_settings, dotenv_settings, yaml_settings, file_secret_settings

    @classmethod
    def with_config_file(cls, config_file: Path | None) -> AppSettings:
        """Loads settings with an optional YAML file override."""

        cls._config_file_override = config_file
        try:
            return cls(**{})
        finally:
            cls._config_file_override = None


def load_settings(config_file: Path | None = None) -> AppSettings:
    """Loads application settings.

    Args:
        config_file: Optional YAML file that replaces the packaged default.

    Returns:
        Validated application settings.
    """

    return AppSettings.with_config_file(config_file)


def _settings_file_path() -> Path:
    override = AppSettings._config_file_override
    if override is not None:
        return override
    configured = os.environ.get("PEAKLE_CONFIG_FILE")
    if configured:
        return Path(configured)
    return DEFAULT_SETTINGS_FILE


def settings_payload(settings: AppSettings) -> dict[str, Any]:
    """Returns a JSON-compatible settings snapshot."""

    return settings.model_dump(mode="json")

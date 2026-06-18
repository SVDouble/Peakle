"""Coordinate models and local Earth approximation helpers."""

from __future__ import annotations

import math

from pydantic import BaseModel, Field

EARTH_RADIUS_M = 6_371_008.8


class GeoPoint(BaseModel):
    """Geodetic coordinate with elevation.

    Attributes:
        latitude_deg: Latitude in degrees.
        longitude_deg: Longitude in degrees.
        elevation_m: Elevation above sea level in meters.
    """

    latitude_deg: float = Field(ge=-90.0, le=90.0)
    longitude_deg: float = Field(ge=-180.0, le=180.0)
    elevation_m: float


class LocalPoint(BaseModel):
    """Point in a local tangent-plane frame.

    Attributes:
        east_m: Meters east from the local origin.
        north_m: Meters north from the local origin.
        up_m: Meters above sea level.
    """

    east_m: float
    north_m: float
    up_m: float

    def as_tuple(self) -> tuple[float, float, float]:
        """Returns the point as an `(east, north, up)` tuple."""

        return (self.east_m, self.north_m, self.up_m)


class LocalFrame(BaseModel):
    """Small-area equirectangular coordinate converter.

    Args:
        origin: Geodetic origin of the local frame.
    """

    origin: GeoPoint

    def geo_to_local(self, point: GeoPoint) -> LocalPoint:
        """Converts a geodetic point to local meters.

        Args:
            point: Geodetic point to convert.

        Returns:
            Local tangent-plane point in meters.
        """

        lat0 = math.radians(self.origin.latitude_deg)
        d_lat = math.radians(point.latitude_deg - self.origin.latitude_deg)
        d_lon = math.radians(point.longitude_deg - self.origin.longitude_deg)
        return LocalPoint(
            east_m=d_lon * EARTH_RADIUS_M * math.cos(lat0),
            north_m=d_lat * EARTH_RADIUS_M,
            up_m=point.elevation_m,
        )

    def local_to_geo(self, point: LocalPoint) -> GeoPoint:
        """Converts local meters to a geodetic point.

        Args:
            point: Local tangent-plane point.

        Returns:
            Geodetic point using the local equirectangular approximation.
        """

        lat0 = math.radians(self.origin.latitude_deg)
        latitude = self.origin.latitude_deg + math.degrees(point.north_m / EARTH_RADIUS_M)
        longitude = self.origin.longitude_deg + math.degrees(point.east_m / (EARTH_RADIUS_M * math.cos(lat0)))
        return GeoPoint(
            latitude_deg=latitude,
            longitude_deg=longitude,
            elevation_m=point.up_m,
        )

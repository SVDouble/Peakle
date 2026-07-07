"""EXIF reading and the camera/geo facts derived from it.

Real photos carry the unknowns we otherwise have to guess: focal length (=> field
of view, which we measured to be decisive for localization), GPS position, capture
time, and compass heading. This module reads them and converts focal length to a
horizontal FOV.
"""

from __future__ import annotations

import math

from PIL import ExifTags, Image

from peakle.domain.camera import CameraIntrinsics
from peakle.pipeline.evidence import ExifData

# A 35mm full-frame is 36mm wide; the 35mm-equivalent focal length maps straight
# to a horizontal FOV without needing a per-model sensor-size database.
FULL_FRAME_WIDTH_MM = 36.0
_GPS_TAGS = ExifTags.GPSTAGS


def read_exif(image: Image.Image | None) -> ExifData:
    """Reads EXIF camera + GPS fields from a PIL image."""

    if image is None:
        return ExifData(present=False)
    exif = image.getexif()
    if not exif:
        return ExifData(present=False)
    tags = {ExifTags.TAGS.get(key, key): value for key, value in exif.items()}
    gps = _gps_tags(exif)
    return ExifData(
        present=True,
        focal_length_mm=_to_float(tags.get("FocalLength")),
        focal_length_35mm_mm=_to_float(tags.get("FocalLengthIn35mmFilm")),
        datetime=tags.get("DateTimeOriginal") or tags.get("DateTime"),
        make=_to_str(tags.get("Make")),
        model=_to_str(tags.get("Model")),
        gps_lat_deg=_gps_coordinate(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef")),
        gps_lon_deg=_gps_coordinate(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef")),
        gps_alt_m=_to_float(gps.get("GPSAltitude")),
        heading_deg=_to_float(gps.get("GPSImgDirection")),
    )


def horizontal_fov_deg(exif: ExifData) -> float | None:
    """Horizontal FOV from the 35mm-equivalent focal length, if available."""

    if exif.focal_length_35mm_mm and exif.focal_length_35mm_mm > 0:
        return math.degrees(2.0 * math.atan(FULL_FRAME_WIDTH_MM / (2.0 * exif.focal_length_35mm_mm)))
    return None


def intrinsics_from_exif(exif: ExifData, width_px: int, height_px: int, default_fov_deg: float) -> CameraIntrinsics:
    """Builds intrinsics from EXIF focal length, falling back to a default FOV."""

    fov = horizontal_fov_deg(exif) or default_fov_deg
    return CameraIntrinsics.from_horizontal_fov(width_px=width_px, height_px=height_px, horizontal_fov_deg=fov)


def _gps_tags(exif: object) -> dict:
    try:
        ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)  # type: ignore[attr-defined]
    except AttributeError, KeyError, ValueError:
        return {}
    return {_GPS_TAGS.get(key, key): value for key, value in (ifd or {}).items()}


def _gps_coordinate(dms: object, ref: object) -> float | None:
    if not dms:
        return None
    try:
        degrees, minutes, seconds = (_to_float(part) or 0.0 for part in dms)
    except TypeError, ValueError:
        return None
    value = degrees + minutes / 60.0 + seconds / 3600.0
    if str(ref).upper() in {"S", "W"}:
        value = -value
    return value


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except TypeError, ValueError:
        return None


def _to_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value).strip("\x00 ").strip() or None

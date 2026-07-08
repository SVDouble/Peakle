"""Angle helpers shared by pose, GT, and UI-facing metrics."""

from __future__ import annotations


def angle_delta_deg(a_deg: float, b_deg: float) -> float:
    """Returns the absolute shortest angular distance in degrees."""

    return abs(float((a_deg - b_deg + 180.0) % 360.0 - 180.0))

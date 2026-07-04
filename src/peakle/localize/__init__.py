"""Skyline-based camera orientation recovery, promoted from the validated local/ research scripts.

The package answers two questions for a photo with a known viewpoint (GPS):
  1. What camera orientation (yaw, pitch, and optionally FOV) best explains the skyline?
  2. Should the answer be TRUSTED?  Every solve carries diagnostics (yaw-profile sharpness,
     alias margin, coverage) and a conservative verdict instead of a bare residual.
"""

from peakle.localize.raycast import skyline_cyl, skyline_pinhole
from peakle.localize.solve import OrientationSolve, solve_orientation

__all__ = [
    "OrientationSolve",
    "skyline_cyl",
    "skyline_pinhole",
    "solve_orientation",
]

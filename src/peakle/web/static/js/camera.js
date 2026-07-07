"use strict";

// One camera abstraction for the whole UI. A placed View and a GT sample are both "cameras" —
// a pose you can look through, a photo/render you can inspect, and outlines you can overlay — so
// the map POV, the inspector, and the map markers consume this single shape instead of branching
// on view-vs-GT everywhere. The two data sources differ only inside these adapters.

import { verticalFovDeg } from "./geometry.js";

// --- placed View (synthetic camera on the current terrain, optionally solved) ---
export function viewCamera(view, solve) {
  const vfov = verticalFovDeg(view.intrinsics);
  const aspect = view.intrinsics.width_px / view.intrinsics.height_px;
  return {
    kind: "view",
    id: view.id,
    label: view.label,
    latlon: null, // a view lives in the current local frame, not at a global lat/lon
    aspect,
    view,
    solve,
    // Pose in the LOCAL scene frame for a POV mode ("true" | "predicted"); null if unavailable.
    scenePose(_terrain, mode) {
      if (mode === "true" && view.true_extrinsics) {
        const e = view.true_extrinsics;
        return { position: e.position, yaw_deg: e.yaw_deg, pitch_deg: e.pitch_deg, vfovDeg: vfov, aspect };
      }
      if (mode === "predicted" && solve) {
        const e = solve.result.estimate.extrinsics;
        return { position: e.position, yaw_deg: e.yaw_deg, pitch_deg: e.pitch_deg, vfovDeg: vfov, aspect };
      }
      return null;
    },
  };
}

// --- GT sample (real photo with a refined pose at a global lat/lon + precomputed outline layers) ---
export function gtCamera(sample, geoToLocal, terrainElevationAt) {
  const hfovRad = (sample.fov_deg * Math.PI) / 180;
  const f = sample.width / hfovRad; // cyltan focal length, px/rad, same both axes
  const pitchDeg = (Math.atan((sample.dv_px ?? 0) / f) * 180) / Math.PI;
  const vfovDeg = (2 * Math.atan(sample.height / (2 * f)) * 180) / Math.PI;
  const aspect = sample.width / sample.height;
  return {
    kind: "gt",
    id: sample.name,
    label: sample.name,
    latlon: { lat: sample.lat, lon: sample.lon },
    aspect,
    sample,
    // A GT sample only has a True pose; it resolves against the terrain window (null if outside).
    scenePose(terrain, mode) {
      if (mode !== "true" || !terrain?.lat_min_deg) {
        return null;
      }
      const local = geoToLocal(terrain, sample.lat, sample.lon);
      if (!local) {
        return null;
      }
      const up = terrainElevationAt ? terrainElevationAt(terrain, local.east_m, local.north_m) : sample.cam_z_m;
      return {
        position: {
          east_m: local.east_m + (sample.de_m ?? 0),
          north_m: local.north_m + (sample.dn_m ?? 0),
          up_m: sample.cam_z_m ?? up,
        },
        yaw_deg: sample.yaw_deg,
        pitch_deg: pitchDeg,
        vfovDeg,
        aspect,
      };
    },
  };
}

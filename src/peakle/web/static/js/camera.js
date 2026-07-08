"use strict";

// One camera type for the whole UI. A placed view and a GT sample are the SAME thing — a camera
// with a pose you look through, an image you inspect, and outlines you overlay. They differ only
// in two data facts the user named: a GT sample's image is IMMUTABLE (a photograph, not a
// re-render) and it carries a PRIOR pose (the dataset label). Everything behavioural —
// resolving the pose into the scene, POV, inspection — is shared here; the two builders at the
// bottom just extract source-specific fields into the same shape.

import { verticalFovDeg } from "./geometry.js";
import { geoToLocal } from "./panels/map/gt-spots.js";

// A pose descriptor is frame-tagged: "local" (already in the scene's east/north/up frame, a placed
// view) or "geo" (a global lat/lon + refined offset, a GT sample). resolvePose is the single place
// that difference lives — every consumer just asks the camera for scenePose(terrain, mode).
function resolvePose(desc, terrain) {
  if (!desc) {
    return null;
  }
  let position;
  if (desc.frame === "geo") {
    if (!terrain?.lat_min_deg) {
      return null;
    }
    const local = geoToLocal(terrain, desc.lat, desc.lon);
    if (!local) {
      return null; // sample is outside the current terrain window
    }
    position = { east_m: local.east_m + (desc.de_m ?? 0), north_m: local.north_m + (desc.dn_m ?? 0), up_m: desc.up_m };
  } else {
    position = desc.position;
  }
  return { position, yaw_deg: desc.yaw_deg, pitch_deg: desc.pitch_deg, vfovDeg: desc.vfovDeg, aspect: desc.aspect };
}

class Camera {
  constructor(fields) {
    Object.assign(this, fields); // kind, id, label, aspect, poses, imageImmutable, hasLayers, view, solve, sample
  }

  // The one POV path: resolve this camera's pose for a mode ("true" | "predicted") into the scene.
  scenePose(terrain, mode) {
    return resolvePose(this.poses[mode], terrain);
  }

  hasPose(mode) {
    return !!this.poses[mode];
  }
}

// --- placed view: pose in the local frame, image is a live DEM render (mutable) ---
export function viewCamera(view, solve) {
  const vfovDeg = verticalFovDeg(view.intrinsics);
  const aspect = view.intrinsics.width_px / view.intrinsics.height_px;
  const local = (e) => ({ frame: "local", position: e.position, yaw_deg: e.yaw_deg, pitch_deg: e.pitch_deg, vfovDeg, aspect });
  return new Camera({
    kind: "view",
    id: view.id,
    label: view.label,
    aspect,
    poses: {
      true: view.true_extrinsics ? local(view.true_extrinsics) : null,
      predicted: solve ? local(solve.result.estimate.extrinsics) : null,
    },
    imageImmutable: false, // rendered from the current pose
    hasLayers: false, // shows its skyline via the inspector SVG, not precomputed layers
    photoSrc: view.photo_url ?? null, // a GT-derived view has a reference photograph to overlay
    view,
    solve,
  });
}

// --- GT sample: an immutable photo + a prior (refined dataset) pose in the geo frame ---
// `adjust` (dyaw/de/dn/dv) is the live inspector adjustment applied on top of the refined pose, so
// the True-POV camera moves as you drag the sliders.
export function gtCamera(sample, adjust = { dyaw: 0, de: 0, dn: 0, dv: 0 }) {
  const f = sample.width / ((sample.fov_deg * Math.PI) / 180); // cyltan focal length, px/rad
  const prior = {
    frame: "geo",
    lat: sample.lat,
    lon: sample.lon,
    de_m: (sample.de_m ?? 0) + (adjust.de ?? 0),
    dn_m: (sample.dn_m ?? 0) + (adjust.dn ?? 0),
    up_m: sample.cam_z_m,
    yaw_deg: sample.yaw_deg + (adjust.dyaw ?? 0),
    pitch_deg: (Math.atan(((sample.dv_px ?? 0) + (adjust.dv ?? 0)) / f) * 180) / Math.PI,
    vfovDeg: (2 * Math.atan(sample.height / (2 * f)) * 180) / Math.PI,
    aspect: sample.width / sample.height,
  };
  return new Camera({
    kind: "gt",
    id: sample.name,
    label: sample.name,
    aspect: sample.width / sample.height,
    poses: { true: prior }, // the refined dataset pose — a prior you can adjust and solve from
    imageImmutable: true, // the photograph
    hasLayers: true, // precomputed outline layer PNGs
    sample,
    photoSrc: `/api/gt/samples/${encodeURIComponent(sample.name)}/layers/photo.png`,
  });
}

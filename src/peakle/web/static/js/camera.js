"use strict";

// One camera type for the whole UI. A placed view and a GT sample are the SAME thing — a camera
// with a pose you look through, an image you inspect, and outlines you overlay. They differ only
// in two data facts the user named: a GT sample's image is IMMUTABLE (a photograph, not a
// re-render) and it carries a PRIOR pose (the dataset label). Everything behavioural —
// resolving the pose into the scene, POV, inspection — is shared here; the two builders at the
// bottom just extract source-specific fields into the same shape.

import { geoToLocal } from "./geometry.js";

const DEG = Math.PI / 180;
const RAD = 180 / Math.PI;

export class ImageCamera {
  constructor({ widthPx, heightPx, horizontalFovDeg, projection = "pinhole" }) {
    this.widthPx = widthPx;
    this.heightPx = heightPx;
    this.horizontalFovDeg = horizontalFovDeg;
    this.projection = projection;
  }

  static fromPayload(payload, fallbackIntrinsics = null, fallbackProjection = "pinhole") {
    if (payload) {
      return new ImageCamera({
        widthPx: payload.width_px,
        heightPx: payload.height_px,
        horizontalFovDeg: payload.horizontal_fov_deg,
        projection: payload.projection ?? fallbackProjection,
      });
    }
    if (!fallbackIntrinsics) {
      return null;
    }
    return new ImageCamera({
      widthPx: fallbackIntrinsics.width_px,
      heightPx: fallbackIntrinsics.height_px,
      horizontalFovDeg: horizontalFovFromIntrinsics(fallbackIntrinsics),
      projection: fallbackProjection,
    });
  }

  static fromGtSample(sample) {
    return new ImageCamera({
      widthPx: sample.width,
      heightPx: sample.height,
      horizontalFovDeg: sample.fov_deg,
      projection: "cyltan",
    });
  }

  get aspect() {
    return this.widthPx / this.heightPx;
  }

  get focalLengthPx() {
    const hfovRad = this.horizontalFovDeg * DEG;
    return this.projection === "cyltan" ? this.widthPx / hfovRad : this.widthPx / (2 * Math.tan(hfovRad / 2));
  }

  get verticalFovDeg() {
    return 2 * Math.atan(this.heightPx / (2 * this.focalLengthPx)) * RAD;
  }

  pitchDegFromVerticalShiftPx(shiftPx) {
    return Math.atan(shiftPx / this.focalLengthPx) * RAD;
  }

  poseDescriptor(fields) {
    return { ...fields, imageCamera: this };
  }

  scenePose(fields) {
    return {
      ...fields,
      vfovDeg: this.verticalFovDeg,
      aspect: this.aspect,
      imageCamera: this,
    };
  }
}

function horizontalFovFromIntrinsics(intrinsics) {
  return 2 * Math.atan(intrinsics.width_px / (2 * intrinsics.focal_length_px)) * RAD;
}

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
  return desc.imageCamera.scenePose({ position, yaw_deg: desc.yaw_deg, pitch_deg: desc.pitch_deg });
}

class Camera {
  constructor(fields) {
    Object.assign(this, fields); // kind, id, label, imageCamera, poses, imageImmutable, hasLayers, view, solve, sample
  }

  // The one POV path: resolve this camera's pose for a mode ("true" | "predicted") into the scene.
  scenePose(terrain, mode) {
    return resolvePose(this.poses[mode], terrain);
  }

  scenePoseFromLocal(extrinsics, terrain) {
    return resolvePose(this.imageCamera.poseDescriptor({
      frame: "local",
      position: extrinsics.position,
      yaw_deg: extrinsics.yaw_deg,
      pitch_deg: extrinsics.pitch_deg,
    }), terrain);
  }

  hasPose(mode) {
    return !!this.poses[mode];
  }
}

export function viewImageCamera(view) {
  return ImageCamera.fromPayload(view.image_camera, view.intrinsics, view.source === "gt" ? "cyltan" : "pinhole");
}

// --- placed view: pose in the local frame, image is a live DEM render (mutable) ---
export function viewCamera(view, solve) {
  const imageCamera = viewImageCamera(view);
  const local = (e) => imageCamera.poseDescriptor({ frame: "local", position: e.position, yaw_deg: e.yaw_deg, pitch_deg: e.pitch_deg });
  return new Camera({
    kind: "view",
    id: view.id,
    label: view.label,
    imageCamera,
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

// --- GT sample: an immutable photo + the refined dataset pose in the geo frame ---
export function gtCamera(sample) {
  const imageCamera = ImageCamera.fromGtSample(sample);
  const prior = imageCamera.poseDescriptor({
    frame: "geo",
    lat: sample.lat,
    lon: sample.lon,
    de_m: sample.de_m ?? 0,
    dn_m: sample.dn_m ?? 0,
    up_m: sample.cam_z_m,
    yaw_deg: sample.yaw_deg,
    pitch_deg: imageCamera.pitchDegFromVerticalShiftPx(sample.dv_px ?? 0),
  });
  return new Camera({
    kind: "gt",
    id: sample.name,
    label: sample.name,
    imageCamera,
    poses: { true: prior }, // the refined dataset pose: a baseline candidate you can open and solve from
    imageImmutable: true, // the photograph
    hasLayers: true, // precomputed outline layer PNGs
    sample,
    photoSrc: `/api/gt/samples/${encodeURIComponent(sample.name)}/layers/photo.png`,
  });
}

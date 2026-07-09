"use strict";

// One selected-view descriptor for the whole UI. A placed view and a GT sample are the same
// left-list entity: a view (image/crop/photo) with a camera model and one or more poses. GT
// samples differ only in two data facts: their view image is immutable and their baseline pose
// comes from the dataset/refinement pipeline.

import { geoToLocal } from "./geometry.js";

const DEG = Math.PI / 180;
const RAD = 180 / Math.PI;

// Browser-side mirror of peakle.domain.camera/projection for interactive Three.js rendering.
// Solver, audit, and skyline math should stay in Python and come through API payloads.
export class CameraModel {
  constructor({ widthPx, heightPx, horizontalFovDeg, projection = "pinhole" }) {
    this.widthPx = widthPx;
    this.heightPx = heightPx;
    this.horizontalFovDeg = horizontalFovDeg;
    this.projection = projection;
  }

  static fromPayload(payload, fallbackIntrinsics = null, fallbackProjection = "pinhole") {
    if (payload) {
      return new CameraModel({
        widthPx: payload.width_px,
        heightPx: payload.height_px,
        horizontalFovDeg: payload.horizontal_fov_deg,
        projection: payload.projection ?? fallbackProjection,
      });
    }
    if (!fallbackIntrinsics) {
      return null;
    }
    return new CameraModel({
      widthPx: fallbackIntrinsics.width_px,
      heightPx: fallbackIntrinsics.height_px,
      horizontalFovDeg: horizontalFovFromIntrinsics(fallbackIntrinsics),
      projection: fallbackProjection,
    });
  }

  static fromGtSample(sample) {
    return new CameraModel({
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
    return { ...fields, cameraModel: this };
  }

  scenePose(fields) {
    return {
      ...fields,
      vfovDeg: this.verticalFovDeg,
      aspect: this.aspect,
      cameraModel: this,
    };
  }
}

function horizontalFovFromIntrinsics(intrinsics) {
  return 2 * Math.atan(intrinsics.width_px / (2 * intrinsics.focal_length_px)) * RAD;
}

// A pose descriptor is frame-tagged: "local" (already in the scene's east/north/up frame, a placed
// view) or "geo" (a global lat/lon + refined offset, a GT sample). resolvePose is the single place
// that difference lives — every consumer just asks the selected view for scenePose(terrain, mode).
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
  return desc.cameraModel.scenePose({ position, yaw_deg: desc.yaw_deg, pitch_deg: desc.pitch_deg });
}

class SelectedView {
  constructor(fields) {
    Object.assign(this, fields); // kind, id, label, cameraModel, poses, imageImmutable, hasLayers, view, solve, sample
  }

  // The one POV path: resolve this view's pose for a mode ("true" | "predicted") into the scene.
  scenePose(terrain, mode) {
    return resolvePose(this.poses[mode], terrain);
  }

  scenePoseFromLocal(extrinsics, terrain) {
    return resolvePose(this.cameraModel.poseDescriptor({
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

export function viewCameraModel(view) {
  return CameraModel.fromPayload(view.image_camera, view.intrinsics, view.source === "gt" ? "cyltan" : "pinhole");
}

// --- placed view: pose in the local frame, image is a live DEM render (mutable) ---
export function viewCamera(view, solve) {
  const cameraModel = viewCameraModel(view);
  const local = (e) => cameraModel.poseDescriptor({ frame: "local", position: e.position, yaw_deg: e.yaw_deg, pitch_deg: e.pitch_deg });
  return new SelectedView({
    kind: "view",
    id: view.id,
    label: view.label,
    cameraModel,
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
  const cameraModel = CameraModel.fromGtSample(sample);
  const gtDepth = cameraModel.poseDescriptor({
    frame: "geo",
    lat: sample.lat,
    lon: sample.lon,
    up_m: sample.gt_elev_m ?? sample.cam_z_m,
    yaw_deg: sample.gt_yaw_deg ?? sample.yaw_deg,
    pitch_deg: sample.gt_pitch_deg ?? cameraModel.pitchDegFromVerticalShiftPx(sample.dv_px ?? 0),
  });
  const refined = cameraModel.poseDescriptor({
    frame: "geo",
    lat: sample.lat,
    lon: sample.lon,
    de_m: sample.de_m ?? 0,
    dn_m: sample.dn_m ?? 0,
    up_m: sample.cam_z_m,
    yaw_deg: sample.yaw_deg,
    pitch_deg: cameraModel.pitchDegFromVerticalShiftPx(sample.dv_px ?? 0),
  });
  return new SelectedView({
    kind: "gt",
    id: sample.name,
    label: sample.name,
    cameraModel,
    poses: {
      gtDepth,
      true: refined,
    },
    imageImmutable: true, // the photograph
    hasLayers: true, // precomputed outline layer PNGs
    sample,
    photoSrc: `/api/gt/samples/${encodeURIComponent(sample.name)}/layers/photo.png`,
  });
}

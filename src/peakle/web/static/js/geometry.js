"use strict";

// Pure scene geometry and coordinate math shared by every panel. No DOM, no
// network. `localToScenePoint` maps local east/north/up metres into the
// normalized three.js scene box used by the map viewer.

import * as THREE from "three";

export const TERRAIN_WIDTH = 2.35;
export const TERRAIN_DEPTH = 1.72;
export const TERRAIN_HEIGHT = 0.66;
export const RASTER_EPSILON = 1e-8;
const DEG = Math.PI / 180;

// The scene box is derived from the terrain's REAL proportions (with mild vertical
// emphasis), not fixed constants: squashing a square 24 km focus window into a
// 2.35x1.72 box while stretching its relief to a fixed height made the mountains
// read as rugged/exaggerated (user-reported).
const SCENE_MAX_EXTENT = 3.0;
export const SCENE_Z_EMPHASIS = 1.35;
export const PHYSICAL_TERRAIN_VERTICAL_SCALE = 1 / SCENE_Z_EMPHASIS;

export function terrainFrame(terrain) {
  const xExtent = Math.max(terrain.x_max_m - terrain.x_min_m, 1);
  const yExtent = Math.max(terrain.y_max_m - terrain.y_min_m, 1);
  const zExtent = Math.max(terrain.elevation_max_m - terrain.elevation_min_m, 1);
  const maxExtent = Math.max(xExtent, yExtent);
  return {
    xMin: terrain.x_min_m,
    xMax: terrain.x_max_m,
    yMin: terrain.y_min_m,
    yMax: terrain.y_max_m,
    zMin: terrain.elevation_min_m,
    zMax: terrain.elevation_max_m,
    sceneW: (SCENE_MAX_EXTENT * xExtent) / maxExtent,
    sceneD: (SCENE_MAX_EXTENT * yExtent) / maxExtent,
    sceneH: (SCENE_MAX_EXTENT * (zExtent / maxExtent)) * SCENE_Z_EMPHASIS,
  };
}

export function scaledTerrainFrame(frame, verticalScale) {
  return { ...frame, sceneH: (frame.sceneH ?? TERRAIN_HEIGHT) * verticalScale };
}

export function terrainLocalGridPoint(terrain, frame, row, col) {
  return {
    east_m: interpolate(frame.xMin, frame.xMax, col / (terrain.grid_width - 1)),
    north_m: interpolate(frame.yMin, frame.yMax, row / (terrain.grid_height - 1)),
    up_m: terrain.elevation_m[row][col],
  };
}

export function localToScenePoint(localPoint, frame) {
  const east = normalize(localPoint.east_m, frame.xMin, frame.xMax) - 0.5;
  const north = normalize(localPoint.north_m, frame.yMin, frame.yMax) - 0.5;
  const elevation = normalize(localPoint.up_m, frame.zMin, frame.zMax);
  return new THREE.Vector3(
    east * (frame.sceneW ?? TERRAIN_WIDTH),
    elevation * (frame.sceneH ?? TERRAIN_HEIGHT),
    -north * (frame.sceneD ?? TERRAIN_DEPTH),
  );
}

// Inverse of localToScenePoint for the east/north plane (used when the user
// clicks the terrain to place a view).
export function sceneToLocalEastNorth(point, frame) {
  const east = (point.x / (frame.sceneW ?? TERRAIN_WIDTH) + 0.5) * (frame.xMax - frame.xMin) + frame.xMin;
  const north = (-point.z / (frame.sceneD ?? TERRAIN_DEPTH) + 0.5) * (frame.yMax - frame.yMin) + frame.yMin;
  return { east_m: east, north_m: north };
}

// Linear geographic <-> local mapping across the focused terrain window. This is accurate enough
// at the app's map scale and keeps every panel using the same convention.
export function geoToLocal(terrain, latDeg, lonDeg) {
  if (
    !Number.isFinite(latDeg) ||
    !Number.isFinite(lonDeg) ||
    terrain.lat_min_deg === undefined ||
    latDeg < terrain.lat_min_deg ||
    latDeg > terrain.lat_max_deg ||
    lonDeg < terrain.lon_min_deg ||
    lonDeg > terrain.lon_max_deg
  ) {
    return null;
  }
  const lonT = (lonDeg - terrain.lon_min_deg) / (terrain.lon_max_deg - terrain.lon_min_deg);
  const latT = (latDeg - terrain.lat_min_deg) / (terrain.lat_max_deg - terrain.lat_min_deg);
  return {
    east_m: interpolate(terrain.x_min_m, terrain.x_max_m, lonT),
    north_m: interpolate(terrain.y_min_m, terrain.y_max_m, latT),
  };
}

export function localTerrainToGeo(terrain, local) {
  return {
    lon: terrain.lon_min_deg + ((local.east_m - terrain.x_min_m) / (terrain.x_max_m - terrain.x_min_m)) * (terrain.lon_max_deg - terrain.lon_min_deg),
    lat: terrain.lat_min_deg + ((local.north_m - terrain.y_min_m) / (terrain.y_max_m - terrain.y_min_m)) * (terrain.lat_max_deg - terrain.lat_min_deg),
  };
}

export function terrainElevationAt(terrain, eastM, northM) {
  const col = normalize(eastM, terrain.x_min_m, terrain.x_max_m) * (terrain.grid_width - 1);
  const row = normalize(northM, terrain.y_min_m, terrain.y_max_m) * (terrain.grid_height - 1);
  const c0 = Math.max(0, Math.min(terrain.grid_width - 2, Math.floor(col)));
  const r0 = Math.max(0, Math.min(terrain.grid_height - 2, Math.floor(row)));
  const tc = Math.max(0, Math.min(1, col - c0));
  const tr = Math.max(0, Math.min(1, row - r0));
  const elevation = terrain.elevation_m;
  const top = interpolate(elevation[r0][c0], elevation[r0][c0 + 1], tc);
  const bottom = interpolate(elevation[r0 + 1][c0], elevation[r0 + 1][c0 + 1], tc);
  return interpolate(top, bottom, tr);
}

export function geoDistanceMeters(lat1Deg, lon1Deg, lat2Deg, lon2Deg) {
  const metersPerDegLat = 111_320;
  const metersPerDegLon = metersPerDegLat * Math.cos(((lat1Deg + lat2Deg) / 2) * DEG);
  return Math.hypot((lon2Deg - lon1Deg) * metersPerDegLon, (lat2Deg - lat1Deg) * metersPerDegLat);
}

export function cameraTargetScenePoint(extrinsics, frame, lookDistanceM = 3600) {
  const yaw = THREE.MathUtils.degToRad(extrinsics.yaw_deg);
  const pitch = THREE.MathUtils.degToRad(extrinsics.pitch_deg);
  const origin = extrinsics.position;
  return localToScenePoint(
    {
      east_m: origin.east_m + Math.sin(yaw) * Math.cos(pitch) * lookDistanceM,
      north_m: origin.north_m + Math.cos(yaw) * Math.cos(pitch) * lookDistanceM,
      up_m: origin.up_m + Math.sin(pitch) * lookDistanceM,
    },
    frame,
  );
}

export function cameraAxes(extrinsics) {
  const yaw = THREE.MathUtils.degToRad(extrinsics.yaw_deg);
  const pitch = THREE.MathUtils.degToRad(extrinsics.pitch_deg);
  const forward = [Math.sin(yaw) * Math.cos(pitch), Math.cos(yaw) * Math.cos(pitch), Math.sin(pitch)];
  const right = [Math.cos(yaw), -Math.sin(yaw), 0];
  const down = cross(forward, right);
  return { right: unit(right), down: unit(down), forward: unit(forward) };
}

export function localToCameraPoint(point, extrinsics, axes) {
  const vector = [
    point.east_m - extrinsics.position.east_m,
    point.north_m - extrinsics.position.north_m,
    point.up_m - extrinsics.position.up_m,
  ];
  return { depth: dot(vector, axes.forward), x: dot(vector, axes.right), y: dot(vector, axes.down) };
}

export function projectCameraPoint(point, intrinsics) {
  const depth = point.depth;
  if (depth <= 1) {
    return { depth, inverseDepth: Number.NEGATIVE_INFINITY, u: Number.NaN, v: Number.NaN, valid: false };
  }
  return {
    depth,
    inverseDepth: 1 / depth,
    valid: true,
    u: intrinsics.focal_length_px * (point.x / depth) + intrinsics.principal_x_px,
    v: intrinsics.focal_length_px * (point.y / depth) + intrinsics.principal_y_px,
  };
}

export function verticalFovDeg(intrinsics) {
  return THREE.MathUtils.radToDeg(2 * Math.atan(intrinsics.height_px / 2 / intrinsics.focal_length_px));
}

export function terrainVertexIndex(terrain, row, col) {
  return row * terrain.grid_width + col;
}

export function terrainTriangleId(terrain, row, col, triangleOffset) {
  return (row * (terrain.grid_width - 1) + col) * 2 + triangleOffset;
}

export function distanceMeters(a, b) {
  return Math.hypot(a.east_m - b.east_m, a.north_m - b.north_m, a.up_m - b.up_m);
}

export function angleDeltaDeg(a, b) {
  return Math.abs(((((a - b + 180) % 360) + 360) % 360) - 180);
}

export function wrapDeg(value) {
  return ((((value + 180) % 360) + 360) % 360) - 180;
}

export function normalize(value, min, max) {
  return (value - min) / Math.max(max - min, 1);
}

export function interpolate(min, max, t) {
  return min + (max - min) * t;
}

export function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

export function unit(vector) {
  const norm = Math.hypot(vector[0], vector[1], vector[2]);
  return [vector[0] / norm, vector[1] / norm, vector[2] / norm];
}

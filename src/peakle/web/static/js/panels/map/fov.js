"use strict";

// Camera coverage overlay: an additive glow fill over the terrain a camera
// actually sees, plus a crisp outline of the coverage boundary. Ported from the
// original viewer and parameterized by a single pose + intrinsics.

import * as THREE from "three";

import {
  RASTER_EPSILON,
  cameraAxes,
  cross,
  dot,
  interpolate,
  localToCameraPoint,
  localToScenePoint,
  projectCameraPoint,
  terrainLocalGridPoint,
  terrainTriangleId,
  terrainVertexIndex,
} from "../../geometry.js";

// Keep the coverage fill essentially on the terrain surface (was 0.011, which
// floated visibly above the mountain); polygon offset handles z-fighting.
const FOV_OVERLAY_VERTICAL_OFFSET = 0.0015;
const VISIBILITY_RASTER_MAX_WIDTH = 820;

export function createFovOverlay(terrain, frame, extrinsics, intrinsics, color) {
  const projection = {
    axes: cameraAxes(extrinsics),
    extrinsics,
    frustumPlanes: cameraFrustumPlanes(intrinsics),
    intrinsics,
  };
  const visibleTriangleIds = visibleTerrainTriangleIds(terrain, frame, projection);
  const positions = [];
  const boundaryEdges = new Map();
  for (let row = 0; row < terrain.grid_height - 1; row += 1) {
    for (let col = 0; col < terrain.grid_width - 1; col += 1) {
      const a = terrainLocalGridPoint(terrain, frame, row, col);
      const b = terrainLocalGridPoint(terrain, frame, row, col + 1);
      const c = terrainLocalGridPoint(terrain, frame, row + 1, col);
      const d = terrainLocalGridPoint(terrain, frame, row + 1, col + 1);
      addOverlayTriangle(positions, boundaryEdges, frame, projection, visibleTriangleIds, terrainTriangleId(terrain, row, col, 0), [a, b, c]);
      addOverlayTriangle(positions, boundaryEdges, frame, projection, visibleTriangleIds, terrainTriangleId(terrain, row, col, 1), [b, d, c]);
    }
  }
  const group = new THREE.Group();
  group.add(createFovFill(positions, color));
  group.add(createFovOutline(boundaryEdges, color));
  return group;
}

function createFovFill(positions, color) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  const mesh = new THREE.Mesh(
    geometry,
    new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: 0.3,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      polygonOffset: true,
      polygonOffsetFactor: -4,
      polygonOffsetUnits: -4,
      side: THREE.DoubleSide,
    }),
  );
  mesh.renderOrder = 1;
  return mesh;
}

function createFovOutline(boundaryEdges, color) {
  const positions = [];
  for (const edge of boundaryEdges.values()) {
    if (edge.count === 1) {
      positions.push(edge.a.x, edge.a.y, edge.a.z, edge.b.x, edge.b.y, edge.b.z);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  const lines = new THREE.LineSegments(
    geometry,
    new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.95, depthWrite: false }),
  );
  lines.renderOrder = 2;
  return lines;
}

function addOverlayTriangle(positions, boundaryEdges, frame, projection, visibleTriangleIds, triangleId, localVertices) {
  if (!visibleTriangleIds.has(triangleId) || !surfaceFacesCamera(localVertices, projection.extrinsics.position)) {
    return;
  }
  const clipped = clipTriangleToCameraFrame(localVertices, frame, projection);
  if (clipped.length < 3) {
    return;
  }
  const origin = clipped[0].scene;
  for (let index = 1; index < clipped.length - 1; index += 1) {
    pushTriangle(positions, origin, clipped[index].scene, clipped[index + 1].scene);
  }
  for (let index = 0; index < clipped.length; index += 1) {
    accumulateBoundaryEdge(boundaryEdges, clipped[index].scene, clipped[(index + 1) % clipped.length].scene);
  }
}

function accumulateBoundaryEdge(boundaryEdges, a, b) {
  const keyA = `${a.x.toFixed(4)},${a.y.toFixed(4)},${a.z.toFixed(4)}`;
  const keyB = `${b.x.toFixed(4)},${b.y.toFixed(4)},${b.z.toFixed(4)}`;
  const key = keyA < keyB ? `${keyA}|${keyB}` : `${keyB}|${keyA}`;
  const existing = boundaryEdges.get(key);
  if (existing) {
    existing.count += 1;
  } else {
    boundaryEdges.set(key, { a, b, count: 1 });
  }
}

function cameraFrustumPlanes(intrinsics) {
  const focalLength = intrinsics.focal_length_px;
  return [
    (point) => point.depth - 1,
    (point) => focalLength * point.x + intrinsics.principal_x_px * point.depth,
    (point) => -focalLength * point.x + (intrinsics.width_px - intrinsics.principal_x_px) * point.depth,
    (point) => focalLength * point.y + intrinsics.principal_y_px * point.depth,
    (point) => -focalLength * point.y + (intrinsics.height_px - intrinsics.principal_y_px) * point.depth,
  ];
}

function clipTriangleToCameraFrame(localVertices, frame, projection) {
  let polygon = localVertices.map((local) => overlayVertex(local, frame, projection));
  for (const signedDistance of projection.frustumPlanes) {
    polygon = clipPolygonByPlane(polygon, signedDistance);
    if (polygon.length === 0) {
      return polygon;
    }
  }
  return polygon;
}

function clipPolygonByPlane(polygon, signedDistance) {
  const clipped = [];
  let previous = polygon[polygon.length - 1];
  let previousDistance = signedDistance(previous.camera);
  let previousInside = previousDistance >= -RASTER_EPSILON;
  for (const current of polygon) {
    const currentDistance = signedDistance(current.camera);
    const currentInside = currentDistance >= -RASTER_EPSILON;
    if (currentInside !== previousInside) {
      const denominator = previousDistance - currentDistance;
      const t = denominator === 0 ? 0 : previousDistance / denominator;
      clipped.push(interpolateOverlayVertex(previous, current, THREE.MathUtils.clamp(t, 0, 1)));
    }
    if (currentInside) {
      clipped.push(current);
    }
    previous = current;
    previousDistance = currentDistance;
    previousInside = currentInside;
  }
  return clipped;
}

function overlayVertex(local, frame, projection) {
  const scene = localToScenePoint(local, frame);
  scene.y += FOV_OVERLAY_VERTICAL_OFFSET;
  return { camera: localToCameraPoint(local, projection.extrinsics, projection.axes), local, scene };
}

function interpolateOverlayVertex(a, b, t) {
  return {
    camera: {
      depth: interpolate(a.camera.depth, b.camera.depth, t),
      x: interpolate(a.camera.x, b.camera.x, t),
      y: interpolate(a.camera.y, b.camera.y, t),
    },
    local: {
      east_m: interpolate(a.local.east_m, b.local.east_m, t),
      north_m: interpolate(a.local.north_m, b.local.north_m, t),
      up_m: interpolate(a.local.up_m, b.local.up_m, t),
    },
    scene: a.scene.clone().lerp(b.scene, t),
  };
}

function surfaceFacesCamera(vertices, cameraPosition) {
  const edgeA = localVector(vertices[0], vertices[1]);
  const edgeB = localVector(vertices[0], vertices[2]);
  const normal = cross(edgeA, edgeB);
  const centroid = {
    east_m: (vertices[0].east_m + vertices[1].east_m + vertices[2].east_m) / 3,
    north_m: (vertices[0].north_m + vertices[1].north_m + vertices[2].north_m) / 3,
    up_m: (vertices[0].up_m + vertices[1].up_m + vertices[2].up_m) / 3,
  };
  return dot(normal, localVector(centroid, cameraPosition)) > 0;
}

function visibleTerrainTriangleIds(terrain, frame, projection) {
  const vertices = projectedTerrainGrid(terrain, frame, projection);
  const raster = visibilityRaster(projection.intrinsics);
  const depthBuffer = new Float32Array(raster.width * raster.height).fill(Number.NEGATIVE_INFINITY);
  const ownerBuffer = new Int32Array(raster.width * raster.height).fill(-1);
  for (let row = 0; row < terrain.grid_height - 1; row += 1) {
    for (let col = 0; col < terrain.grid_width - 1; col += 1) {
      const topLeft = vertices[terrainVertexIndex(terrain, row, col)];
      const topRight = vertices[terrainVertexIndex(terrain, row, col + 1)];
      const bottomLeft = vertices[terrainVertexIndex(terrain, row + 1, col)];
      const bottomRight = vertices[terrainVertexIndex(terrain, row + 1, col + 1)];
      rasterizeVisibilityTriangle(raster, depthBuffer, ownerBuffer, terrainTriangleId(terrain, row, col, 0), topLeft, topRight, bottomLeft);
      rasterizeVisibilityTriangle(raster, depthBuffer, ownerBuffer, terrainTriangleId(terrain, row, col, 1), topRight, bottomRight, bottomLeft);
    }
  }
  const visible = new Set();
  for (const owner of ownerBuffer) {
    if (owner >= 0) {
      visible.add(owner);
    }
  }
  return visible;
}

function projectedTerrainGrid(terrain, frame, projection) {
  const vertices = new Array(terrain.grid_width * terrain.grid_height);
  for (let row = 0; row < terrain.grid_height; row += 1) {
    for (let col = 0; col < terrain.grid_width; col += 1) {
      const local = terrainLocalGridPoint(terrain, frame, row, col);
      const cameraPoint = localToCameraPoint(local, projection.extrinsics, projection.axes);
      vertices[terrainVertexIndex(terrain, row, col)] = projectCameraPoint(cameraPoint, projection.intrinsics);
    }
  }
  return vertices;
}

function visibilityRaster(intrinsics) {
  const scale = Math.min(1, VISIBILITY_RASTER_MAX_WIDTH / intrinsics.width_px);
  return {
    height: Math.max(1, Math.round(intrinsics.height_px * scale)),
    scale,
    width: Math.max(1, Math.round(intrinsics.width_px * scale)),
  };
}

function rasterizeVisibilityTriangle(raster, depthBuffer, ownerBuffer, triangleId, a, b, c) {
  if (!a.valid || !b.valid || !c.valid) {
    return;
  }
  const au = a.u * raster.scale;
  const av = a.v * raster.scale;
  const bu = b.u * raster.scale;
  const bv = b.v * raster.scale;
  const cu = c.u * raster.scale;
  const cv = c.v * raster.scale;
  const minX = Math.max(0, Math.floor(Math.min(au, bu, cu)));
  const maxX = Math.min(raster.width - 1, Math.ceil(Math.max(au, bu, cu)));
  const minY = Math.max(0, Math.floor(Math.min(av, bv, cv)));
  const maxY = Math.min(raster.height - 1, Math.ceil(Math.max(av, bv, cv)));
  if (minX > maxX || minY > maxY) {
    return;
  }
  const denominator = (bv - cv) * (au - cu) + (cu - bu) * (av - cv);
  if (Math.abs(denominator) <= RASTER_EPSILON) {
    return;
  }
  for (let y = minY; y <= maxY; y += 1) {
    const sampleY = y + 0.5;
    for (let x = minX; x <= maxX; x += 1) {
      const sampleX = x + 0.5;
      const weightA = ((bv - cv) * (sampleX - cu) + (cu - bu) * (sampleY - cv)) / denominator;
      const weightB = ((cv - av) * (sampleX - cu) + (au - cu) * (sampleY - cv)) / denominator;
      const weightC = 1 - weightA - weightB;
      if (weightA < -RASTER_EPSILON || weightB < -RASTER_EPSILON || weightC < -RASTER_EPSILON) {
        continue;
      }
      const inverseDepth = weightA * a.inverseDepth + weightB * b.inverseDepth + weightC * c.inverseDepth;
      const bufferIndex = y * raster.width + x;
      if (inverseDepth > depthBuffer[bufferIndex]) {
        depthBuffer[bufferIndex] = inverseDepth;
        ownerBuffer[bufferIndex] = triangleId;
      }
    }
  }
}

function pushTriangle(positions, a, b, c) {
  positions.push(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z);
}

function localVector(from, to) {
  return [to.east_m - from.east_m, to.north_m - from.north_m, to.up_m - from.up_m];
}

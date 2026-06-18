"use strict";

// Terrain rendering: hypsometric-shaded mesh, marching-squares isolines, a base
// plane, and cartographic peak markers with labels. Ported from the original
// single-file viewer and parameterized by terrain + frame.

import * as THREE from "three";
import { CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

import {
  TERRAIN_DEPTH,
  TERRAIN_HEIGHT,
  TERRAIN_WIDTH,
  interpolate,
  localToScenePoint,
  terrainLocalGridPoint,
} from "../../geometry.js";

const ISOLINE_INTERVAL_M = 250;
const ISOLINE_VERTICAL_OFFSET = 0.0014;
const MAX_PEAK_LABELS = 8;

let peakGlyphTexture = null;

export function createTerrainGroup(terrain, frame) {
  const group = new THREE.Group();
  const mesh = createTerrainMesh(terrain, frame);
  group.add(mesh);
  group.add(createIsolines(terrain, frame));
  group.add(createBasePlane());
  return { group, mesh };
}

function createTerrainMesh(terrain, frame) {
  const geometry = terrainGeometry(terrain, frame);
  geometry.computeVertexNormals();
  return new THREE.Mesh(geometry, createTerrainMaterial());
}

function terrainGeometry(terrain, frame) {
  const gridWidth = terrain.grid_width;
  const gridHeight = terrain.grid_height;
  const positions = new Float32Array(gridWidth * gridHeight * 3);
  const indices = [];
  for (let row = 0; row < gridHeight; row += 1) {
    for (let col = 0; col < gridWidth; col += 1) {
      const offset = (row * gridWidth + col) * 3;
      const point = localToScenePoint(terrainLocalGridPoint(terrain, frame, row, col), frame);
      positions[offset] = point.x;
      positions[offset + 1] = point.y;
      positions[offset + 2] = point.z;
    }
  }
  for (let row = 0; row < gridHeight - 1; row += 1) {
    for (let col = 0; col < gridWidth - 1; col += 1) {
      const a = row * gridWidth + col;
      const b = a + 1;
      const c = a + gridWidth;
      const d = c + 1;
      indices.push(a, b, c, b, d, c);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  return geometry;
}

// Hypsometric (elevation) gradient with soft contour banding for separable
// ridges, injected into a standard lit material.
function createTerrainMaterial() {
  const material = new THREE.MeshStandardMaterial({ roughness: 0.9, metalness: 0, side: THREE.FrontSide });
  material.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", "#include <common>\nvarying float vTerrainElevation;")
      .replace(
        "#include <begin_vertex>",
        `#include <begin_vertex>\nvTerrainElevation = position.y / float(${TERRAIN_HEIGHT.toFixed(4)});`,
      );
    shader.fragmentShader = shader.fragmentShader
      .replace(
        "#include <common>",
        `#include <common>
varying float vTerrainElevation;
vec3 peakleHypsometric(float t) {
  vec3 c = vec3(0.16, 0.30, 0.20);
  c = mix(c, vec3(0.24, 0.44, 0.25), smoothstep(0.0, 0.22, t));
  c = mix(c, vec3(0.46, 0.50, 0.28), smoothstep(0.22, 0.45, t));
  c = mix(c, vec3(0.64, 0.57, 0.37), smoothstep(0.45, 0.66, t));
  c = mix(c, vec3(0.55, 0.44, 0.35), smoothstep(0.66, 0.82, t));
  c = mix(c, vec3(0.84, 0.81, 0.72), smoothstep(0.82, 0.93, t));
  c = mix(c, vec3(0.97, 0.97, 0.95), smoothstep(0.93, 1.0, t));
  return c;
}`,
      )
      .replace(
        "#include <color_fragment>",
        `#include <color_fragment>
{
  float terrainT = clamp(vTerrainElevation, 0.0, 1.0);
  vec3 hypso = peakleHypsometric(terrainT);
  float bandEdge = fract(terrainT * 9.0);
  float band = smoothstep(0.0, 0.08, bandEdge) * smoothstep(1.0, 0.92, bandEdge);
  hypso *= mix(0.8, 1.0, band);
  diffuseColor.rgb = hypso;
}`,
      );
  };
  return material;
}

function createIsolines(terrain, frame) {
  const positions = [];
  const firstLevel = Math.ceil(frame.zMin / ISOLINE_INTERVAL_M) * ISOLINE_INTERVAL_M;
  for (let level = firstLevel; level <= frame.zMax; level += ISOLINE_INTERVAL_M) {
    addIsolineLevel(positions, terrain, frame, level);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  const lines = new THREE.LineSegments(
    geometry,
    new THREE.LineBasicMaterial({ color: 0xe7dab2, transparent: true, opacity: 0.26, depthWrite: false }),
  );
  lines.renderOrder = 1;
  return lines;
}

function addIsolineLevel(positions, terrain, frame, level) {
  for (let row = 0; row < terrain.grid_height - 1; row += 1) {
    for (let col = 0; col < terrain.grid_width - 1; col += 1) {
      const corners = [
        { row, col, elevation: terrain.elevation_m[row][col] },
        { row, col: col + 1, elevation: terrain.elevation_m[row][col + 1] },
        { row: row + 1, col: col + 1, elevation: terrain.elevation_m[row + 1][col + 1] },
        { row: row + 1, col, elevation: terrain.elevation_m[row + 1][col] },
      ];
      const intersections = [
        edgeLevelIntersection(corners[0], corners[1], level),
        edgeLevelIntersection(corners[1], corners[2], level),
        edgeLevelIntersection(corners[2], corners[3], level),
        edgeLevelIntersection(corners[3], corners[0], level),
      ].filter(Boolean);
      const unique = deduplicate(intersections);
      if (unique.length === 2) {
        pushSegment(positions, terrain, frame, unique[0], unique[1]);
      } else if (unique.length === 4) {
        pushSegment(positions, terrain, frame, unique[0], unique[1]);
        pushSegment(positions, terrain, frame, unique[2], unique[3]);
      }
    }
  }
}

function edgeLevelIntersection(a, b, level) {
  const aDelta = a.elevation - level;
  const bDelta = b.elevation - level;
  if ((aDelta === 0 && bDelta === 0) || aDelta * bDelta > 0) {
    return null;
  }
  const t = aDelta === bDelta ? 0 : aDelta / (aDelta - bDelta);
  if (t < 0 || t > 1) {
    return null;
  }
  return { row: interpolate(a.row, b.row, t), col: interpolate(a.col, b.col, t), elevation: level };
}

function deduplicate(points) {
  const unique = [];
  for (const point of points) {
    if (!unique.some((c) => Math.abs(c.row - point.row) < 1e-5 && Math.abs(c.col - point.col) < 1e-5)) {
      unique.push(point);
    }
  }
  return unique;
}

function pushSegment(positions, terrain, frame, a, b) {
  const aPoint = floatPoint(terrain, frame, a);
  const bPoint = floatPoint(terrain, frame, b);
  positions.push(aPoint.x, aPoint.y, aPoint.z, bPoint.x, bPoint.y, bPoint.z);
}

function floatPoint(terrain, frame, sample) {
  const local = {
    east_m: interpolate(frame.xMin, frame.xMax, sample.col / Math.max(1, terrain.grid_width - 1)),
    north_m: interpolate(frame.yMin, frame.yMax, sample.row / Math.max(1, terrain.grid_height - 1)),
    up_m: sample.elevation,
  };
  const point = localToScenePoint(local, frame);
  point.y += ISOLINE_VERTICAL_OFFSET;
  return point;
}

function createBasePlane() {
  const geometry = new THREE.PlaneGeometry(TERRAIN_WIDTH * 1.05, TERRAIN_DEPTH * 1.05);
  const material = new THREE.MeshStandardMaterial({ color: 0x1f2d22, roughness: 0.95, metalness: 0, side: THREE.FrontSide });
  const plane = new THREE.Mesh(geometry, material);
  plane.rotation.x = -Math.PI / 2;
  plane.position.y = -0.018;
  return plane;
}

export function addPeakLabels(scene, peaks, frame) {
  const labeled = [...peaks].sort((a, b) => b.prominence_m - a.prominence_m).slice(0, MAX_PEAK_LABELS);
  const labels = [];
  for (const peak of labeled) {
    const point = localToScenePoint(peak.local_position, frame);
    const marker = createPeakMarker();
    marker.position.copy(point);
    scene.add(marker);
    const label = createLabel(peak.name, "peak-label");
    label.position.copy(point);
    label.position.y += 0.078;
    label.userData.occlusionAnchor = marker;
    scene.add(label);
    labels.push(label);
  }
  return labels;
}

function createPeakMarker() {
  if (!peakGlyphTexture) {
    peakGlyphTexture = createPeakGlyphTexture();
  }
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: peakGlyphTexture, transparent: true, depthWrite: false }));
  sprite.scale.setScalar(0.06);
  sprite.center.set(0.5, 0);
  return sprite;
}

function createPeakGlyphTexture() {
  const size = 64;
  const element = document.createElement("canvas");
  element.width = size;
  element.height = size;
  const context = element.getContext("2d");
  context.beginPath();
  context.moveTo(size * 0.5, size * 0.12);
  context.lineTo(size * 0.86, size * 0.82);
  context.lineTo(size * 0.14, size * 0.82);
  context.closePath();
  context.fillStyle = "#e2483a";
  context.fill();
  context.lineWidth = size * 0.07;
  context.lineJoin = "round";
  context.strokeStyle = "#2a0f0c";
  context.stroke();
  const texture = new THREE.CanvasTexture(element);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

export function createLabel(text, className) {
  const element = document.createElement("div");
  element.className = `terrain-label ${className}`;
  element.textContent = text;
  return new CSS2DObject(element);
}

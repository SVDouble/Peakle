"use strict";

// Terrain rendering: a lit mesh with several selectable shading modes, crisp
// black contour lines drawn analytically in the fragment shader, a base plane,
// and cartographic peak markers with labels.

import * as THREE from "three";
import { CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

import { TERRAIN_DEPTH, TERRAIN_HEIGHT, TERRAIN_WIDTH, interpolate, localToScenePoint } from "../../geometry.js";

const CONTOUR_INTERVAL_M = 250;
// Hillshade direction; matches the scene's directional "sun" so baked relief and
// real lighting agree.
const RELIEF_LIGHT_DIR = new THREE.Vector3(-1.6, 2.4, 1.1).normalize();

// Selectable surface shading modes. `index` is the int handed to the shader; the
// list order is what the Setup panel offers, so the default sits first.
export const SHADING_MODES = [
  { id: "relief", label: "Relief shading", index: 1 },
  { id: "hypsometric", label: "Elevation tint", index: 0 },
  { id: "massif", label: "Color per massif", index: 2 },
  { id: "landcover", label: "Landcover", index: 3 },
];
export const DEFAULT_SHADING_MODE = SHADING_MODES[0].id;

const SHADING_INDEX = new Map(SHADING_MODES.map((mode) => [mode.id, mode.index]));
const PEAK_MARKER_SCALE = 0.034;
const PEAK_LABEL_OFFSET = 0.052;

let peakGlyphTexture = null;

export function createTerrainGroup(terrain, frame, peaks) {
  const group = new THREE.Group();
  const { mesh, setShadingMode, setContours } = createTerrainMesh(terrain, frame, peaks);
  group.add(mesh);
  group.add(createBasePlane(frame));
  return { group, mesh, setShadingMode, setContours };
}

function createTerrainMesh(terrain, frame, peaks) {
  const geometry = terrainGeometry(terrain, frame, peaks);
  geometry.computeVertexNormals();
  const { material, uniforms } = createTerrainMaterial(frame);
  const mesh = new THREE.Mesh(geometry, material);
  const setShadingMode = (id) => {
    uniforms.uShadingMode.value = SHADING_INDEX.get(id) ?? SHADING_INDEX.get(DEFAULT_SHADING_MODE);
  };
  const setContours = (enabled) => {
    uniforms.uContours.value = enabled ? 1 : 0;
  };
  return { mesh, setShadingMode, setContours };
}

function terrainGeometry(terrain, frame, peaks) {
  const gridWidth = terrain.grid_width;
  const gridHeight = terrain.grid_height;
  const positions = new Float32Array(gridWidth * gridHeight * 3);
  const regionHue = new Float32Array(gridWidth * gridHeight);
  const massifPeaks = (peaks || []).filter((peak) => peak.prominence_m > 0);
  const colorPeaks = massifPeaks.length ? massifPeaks : peaks;
  const indices = [];
  for (let row = 0; row < gridHeight; row += 1) {
    for (let col = 0; col < gridWidth; col += 1) {
      const index = row * gridWidth + col;
      const east = interpolate(frame.xMin, frame.xMax, col / Math.max(1, gridWidth - 1));
      const north = interpolate(frame.yMin, frame.yMax, row / Math.max(1, gridHeight - 1));
      const point = localToScenePoint({ east_m: east, north_m: north, up_m: terrain.elevation_m[row][col] }, frame);
      positions[index * 3] = point.x;
      positions[index * 3 + 1] = point.y;
      positions[index * 3 + 2] = point.z;
      regionHue[index] = nearestPeakHue(east, north, colorPeaks);
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
  geometry.setAttribute("regionHue", new THREE.BufferAttribute(regionHue, 1));
  geometry.setIndex(indices);
  return geometry;
}

// Voronoi-style region tint: each vertex takes the hue of its nearest peak, so
// the "color per massif" mode paints every mountain a distinct colour. The
// golden-ratio step keeps neighbouring peaks well separated on the hue wheel.
function nearestPeakHue(east, north, peaks) {
  if (!peaks || !peaks.length) {
    return 0;
  }
  let bestIndex = 0;
  let bestDistance = Infinity;
  for (let i = 0; i < peaks.length; i += 1) {
    const position = peaks[i].local_position;
    const distance = (position.east_m - east) ** 2 + (position.north_m - north) ** 2;
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = i;
    }
  }
  return (bestIndex * 0.61803398875) % 1;
}

// One standard lit material whose albedo is computed by a small shading kernel
// selected at runtime via the uShadingMode uniform, with analytic contour lines
// composited on top. Returns the captured uniforms so the mode can be toggled
// live without rebuilding the scene.
function createTerrainMaterial(frame) {
  const uniforms = {
    uShadingMode: { value: SHADING_INDEX.get(DEFAULT_SHADING_MODE) },
    uContours: { value: 1 },
    uElevMin: { value: frame.zMin },
    uElevRange: { value: Math.max(1, frame.zMax - frame.zMin) },
    uContourInterval: { value: CONTOUR_INTERVAL_M },
    uReliefLightDir: { value: RELIEF_LIGHT_DIR.clone() },
  };
  const material = new THREE.MeshStandardMaterial({ roughness: 0.92, metalness: 0, side: THREE.FrontSide });
  material.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace(
        "#include <common>",
        `#include <common>
attribute float regionHue;
varying float vTerrainElevation;
varying float vRegionHue;
varying vec3 vObjNormal;`,
      )
      .replace(
        "#include <begin_vertex>",
        `#include <begin_vertex>
vTerrainElevation = position.y / float(${(frame.sceneH ?? TERRAIN_HEIGHT).toFixed(4)});
vRegionHue = regionHue;
vObjNormal = normalize(normal);`,
      );
    shader.fragmentShader = shader.fragmentShader
      .replace(
        "#include <common>",
        `#include <common>
uniform int uShadingMode;
uniform float uContours;
uniform float uElevMin;
uniform float uElevRange;
uniform float uContourInterval;
uniform vec3 uReliefLightDir;
varying float vTerrainElevation;
varying float vRegionHue;
varying vec3 vObjNormal;

vec3 peakleHypsometric(float t) {
  vec3 c = vec3(0.16, 0.30, 0.20);
  c = mix(c, vec3(0.24, 0.44, 0.25), smoothstep(0.0, 0.22, t));
  c = mix(c, vec3(0.46, 0.50, 0.28), smoothstep(0.22, 0.45, t));
  c = mix(c, vec3(0.64, 0.57, 0.37), smoothstep(0.45, 0.66, t));
  c = mix(c, vec3(0.55, 0.44, 0.35), smoothstep(0.66, 0.82, t));
  c = mix(c, vec3(0.84, 0.81, 0.72), smoothstep(0.82, 0.93, t));
  c = mix(c, vec3(0.97, 0.97, 0.95), smoothstep(0.93, 1.0, t));
  return c;
}

vec3 peakleHsv2rgb(vec3 c) {
  vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
  return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}

// Anti-aliased, screen-space-constant-width contour lines that fade out where
// the slope packs them tighter than the screen can resolve (steep faces),
// avoiding the black smears the old floating line geometry produced.
float peakleContour(float elevM, float interval) {
  float f = elevM / interval;
  float width = fwidth(f);
  float dist = abs(fract(f - 0.5) - 0.5);
  float line = 1.0 - smoothstep(0.0, width * 1.3, dist);
  return line * (1.0 - smoothstep(0.45, 1.0, width));
}`,
      )
      .replace(
        "#include <color_fragment>",
        `#include <color_fragment>
{
  float terrainT = clamp(vTerrainElevation, 0.0, 1.0);
  vec3 surfaceNormal = normalize(vObjNormal);
  float slope = clamp(1.0 - surfaceNormal.y, 0.0, 1.0);
  float relief = clamp(dot(surfaceNormal, normalize(uReliefLightDir)), 0.0, 1.0);
  relief = 0.35 + 0.65 * relief;

  vec3 base;
  if (uShadingMode == 0) {
    base = peakleHypsometric(terrainT);
    float bandEdge = fract(terrainT * 9.0);
    float band = smoothstep(0.0, 0.08, bandEdge) * smoothstep(1.0, 0.92, bandEdge);
    base *= mix(0.8, 1.0, band);
  } else if (uShadingMode == 2) {
    base = peakleHsv2rgb(vec3(vRegionHue, 0.5, 0.85)) * (0.55 + 0.45 * relief);
  } else if (uShadingMode == 3) {
    vec3 grass = vec3(0.26, 0.40, 0.22);
    vec3 rock = vec3(0.42, 0.38, 0.33);
    vec3 scree = vec3(0.56, 0.53, 0.47);
    vec3 snow = vec3(0.95, 0.96, 0.98);
    base = mix(grass, rock, smoothstep(0.30, 0.62, slope));
    base = mix(base, scree, smoothstep(0.45, 0.62, terrainT) * (1.0 - smoothstep(0.55, 0.82, slope)));
    base = mix(base, snow, smoothstep(0.78, 0.92, terrainT));
    base *= relief;
  } else {
    vec3 lit = vec3(0.80, 0.78, 0.72);
    vec3 shade = vec3(0.18, 0.21, 0.24);
    base = mix(shade, lit, relief);
    base = mix(base, base * 0.7, slope * 0.5);
  }

  float elevM = uElevMin + terrainT * uElevRange;
  float contour = peakleContour(elevM, uContourInterval) * uContours;
  base = mix(base, vec3(0.02), contour * 0.85);

  diffuseColor.rgb = base;
}`,
      );
  };
  return { material, uniforms };
}

function createBasePlane(frame) {
  const geometry = new THREE.PlaneGeometry((frame.sceneW ?? TERRAIN_WIDTH) * 1.05, (frame.sceneD ?? TERRAIN_DEPTH) * 1.05);
  const material = new THREE.MeshStandardMaterial({ color: 0x1f2d22, roughness: 0.95, metalness: 0, side: THREE.FrontSide });
  const plane = new THREE.Mesh(geometry, material);
  plane.rotation.x = -Math.PI / 2;
  plane.position.y = -0.018;
  return plane;
}

// Returns { group, labels }. Markers (red-triangle sprites) AND text labels live in one group so
// a rebuild can remove them all at once — previously only the labels were returned and the marker
// sprites leaked, leaving red triangles hanging in the air after a map switch (user-reported).
export function addPeakLabels(scene, peaks, frame) {
  const labeled = [...peaks].sort(
    (a, b) => b.prominence_m - a.prominence_m || b.elevation_m - a.elevation_m || a.name.localeCompare(b.name),
  );
  const group = new THREE.Group();
  const labels = [];
  for (const peak of labeled) {
    const point = localToScenePoint(peak.local_position, frame);
    const marker = createPeakMarker();
    marker.position.copy(point);
    group.add(marker);
    const label = createLabel(peak.name, "peak-label");
    label.position.copy(point);
    label.position.y += PEAK_LABEL_OFFSET;
    label.userData.occlusionAnchor = marker;
    group.add(label);
    labels.push(label);
  }
  scene.add(group);
  return { group, labels };
}

function createPeakMarker() {
  if (!peakGlyphTexture) {
    peakGlyphTexture = createPeakGlyphTexture();
  }
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: peakGlyphTexture, transparent: true, depthWrite: false }));
  sprite.scale.setScalar(PEAK_MARKER_SCALE);
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

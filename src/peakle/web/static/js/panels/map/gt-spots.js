"use strict";

// GT sample spots on the 3D map: a pin + clickable photo-thumbnail chip for
// every corpus sample inside the current terrain window. Clicking a chip selects
// the sample (inspector + list follow); the selected sample additionally shows a
// camera glyph at its refined pose so the labelled placement is visible in 3D.

import * as THREE from "three";
import { CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

import { api } from "../../api.js";
import { interpolate, localToScenePoint, normalize } from "../../geometry.js";
import { createCameraMarker } from "./cameras.js";

const SPOT_CAP = 250;
export const GT_COLOR = 0x8ee08e;

// Linear lat/lon -> local east/north across the terrain window (plenty accurate
// at map scale). Returns null when the point is outside the window.
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
  // NOTE: geometry.js `normalize` clamps its denominator to >= 1 (a guard for
  // meter ranges); degree spans are ~0.2, so it must not be used here.
  const lonT = (lonDeg - terrain.lon_min_deg) / (terrain.lon_max_deg - terrain.lon_min_deg);
  const latT = (latDeg - terrain.lat_min_deg) / (terrain.lat_max_deg - terrain.lat_min_deg);
  return {
    east_m: interpolate(terrain.x_min_m, terrain.x_max_m, lonT),
    north_m: interpolate(terrain.y_min_m, terrain.y_max_m, latT),
  };
}

export function terrainElevationAt(terrain, eastM, northM) {
  const col = normalize(eastM, terrain.x_min_m, terrain.x_max_m) * (terrain.grid_width - 1);
  const row = normalize(northM, terrain.y_min_m, terrain.y_max_m) * (terrain.grid_height - 1);
  const c0 = Math.max(0, Math.min(terrain.grid_width - 2, Math.floor(col)));
  const r0 = Math.max(0, Math.min(terrain.grid_height - 2, Math.floor(row)));
  const tc = Math.max(0, Math.min(1, col - c0));
  const tr = Math.max(0, Math.min(1, row - r0));
  const e = terrain.elevation_m;
  const top = interpolate(e[r0][c0], e[r0][c0 + 1], tc);
  const bottom = interpolate(e[r0 + 1][c0], e[r0 + 1][c0 + 1], tc);
  return interpolate(top, bottom, tr);
}

export function buildGtSpotsLayer(terrain, frame, samples, selectedName, onPick, onFocus) {
  const group = new THREE.Group();
  if (!samples || !samples.length) {
    return group;
  }
  let shown = 0;
  for (const sample of samples) {
    if (shown >= SPOT_CAP && sample.name !== selectedName) {
      continue;
    }
    const local = geoToLocal(terrain, sample.lat, sample.lon);
    if (!local) {
      continue;
    }
    shown += 1;
    const selected = sample.name === selectedName;
    const up = terrainElevationAt(terrain, local.east_m, local.north_m);
    const anchor = localToScenePoint({ ...local, up_m: up }, frame);

    const pin = new THREE.Mesh(
      new THREE.SphereGeometry(selected ? 0.009 : 0.006, 10, 10),
      new THREE.MeshBasicMaterial({ color: selected ? 0xffd24a : sample.quality === "CLEAN" ? GT_COLOR : 0xd07a6a }),
    );
    pin.position.copy(anchor);
    pin.position.y += 0.004;
    group.add(pin);

    const chip = document.createElement("div");
    chip.className = `gt-spot${selected ? " sel" : ""}${sample.quality === "CLEAN" ? "" : " suspect"}`;
    chip.title = sample.name;
    const img = document.createElement("img");
    img.loading = "lazy";
    img.src = api.gtThumbUrl(sample.name);
    img.alt = sample.name;
    chip.append(img);
    if (onFocus) {
      const focus = document.createElement("button");
      focus.type = "button";
      focus.className = "gt-spot-focus";
      focus.title = "Center the 3D map here";
      focus.textContent = "⌖";
      focus.addEventListener("click", (event) => {
        event.stopPropagation();
        onFocus(sample);
      });
      chip.append(focus);
    }
    // The CSS2D layer itself ignores pointer events; the chip opts back in.
    chip.addEventListener("pointerdown", (event) => event.stopPropagation());
    chip.addEventListener("click", (event) => {
      event.stopPropagation();
      onPick(sample.name);
    });
    // Occlusion tests against the floating chip itself, not the surface pin —
    // a surface anchor self-occludes against its own slope and hides every chip.
    const label = new CSS2DObject(chip);
    label.position.copy(anchor);
    label.position.y += 0.028;
    group.add(label);

    // Show the labelled camera placement for the selected sample: position is
    // GPS + the refinement's east/north correction, heading is the refined yaw.
    if (selected && sample.yaw_deg !== undefined) {
      const extrinsics = {
        position: {
          east_m: local.east_m + (sample.de_m ?? 0),
          north_m: local.north_m + (sample.dn_m ?? 0),
          up_m: up,
        },
        yaw_deg: sample.yaw_deg,
        pitch_deg: 0,
      };
      group.add(createCameraMarker(extrinsics, frame, 0xffd24a, `${sample.name} · GT`, "gt-camera"));
    }
  }
  return group;
}

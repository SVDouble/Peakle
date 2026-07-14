"use strict";

// GT sample spots on the 3D map: a pin + clickable photo-thumbnail chip for
// every corpus sample inside the current terrain window. Clicking a chip selects
// the sample (inspector + list follow); the selected sample additionally shows a
// pose glyph at the original dataset placement so it is visible in 3D.

import * as THREE from "three";
import { CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

import { api } from "../../api.js";
import { geoToLocal, localToScenePoint, terrainElevationAt } from "../../geometry.js";
import { createCameraMarker } from "./cameras.js";

const SPOT_CAP = 250;
export const GT_COLOR = 0x8ee08e;

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
    chip.title = `${sample.name}\nDouble-click to center the map`;
    const img = document.createElement("img");
    img.loading = "lazy";
    img.src = api.gtThumbUrl(sample.name);
    img.alt = sample.name;
    chip.append(img);
    // The CSS2D layer itself ignores pointer events; the chip opts back in.
    chip.addEventListener("pointerdown", (event) => event.stopPropagation());
    chip.addEventListener("click", (event) => {
      event.stopPropagation();
      onPick(sample.name);
    });
    chip.addEventListener("dblclick", (event) => {
      event.stopPropagation();
      onFocus?.(sample);
    });
    // Occlusion tests against the floating chip itself, not the surface pin —
    // a surface anchor self-occludes against its own slope and hides every chip.
    const label = new CSS2DObject(chip);
    label.position.copy(anchor);
    label.position.y += 0.028;
    group.add(label);

    // Show only the original dataset pose. GT-v2 offsets remain diagnostics in
    // GT Lab and must never masquerade as the reference camera.
    if (selected && sample.gt_yaw_deg !== undefined) {
      const extrinsics = {
        position: {
          east_m: local.east_m,
          north_m: local.north_m,
          up_m: sample.gt_elev_m ?? up,
        },
        yaw_deg: sample.gt_yaw_deg,
        pitch_deg: sample.gt_pitch_deg ?? 0,
      };
      group.add(createCameraMarker(extrinsics, frame, 0xffd24a, `${sample.name} · Original GT pose`, "gt-camera"));
    }
  }
  return group;
}

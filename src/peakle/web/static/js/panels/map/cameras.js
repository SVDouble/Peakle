"use strict";

// Camera marker glyphs (body + lens + frustum + label) placed at a pose, and the
// per-selection layer that shows the selected view's ground-truth and predicted
// cameras together with their FOV coverage overlays.

import * as THREE from "three";

import { cameraTargetScenePoint, localToScenePoint } from "../../geometry.js";
import { createFovOverlay } from "./fov.js";
import { createLabel } from "./terrain-mesh.js";

export const TRUE_COLOR = 0x53c6ff;
export const PREDICTED_COLOR = 0xff8c42;

// A compact, legible camera glyph: a vertical stem to the ground for depth, a
// bright core sphere, and a cone pointing where the camera looks. Reads cleanly
// at map scale where the old box+lens+wireframe glyph turned to mush.
export function createCameraMarker(extrinsics, frame, color, labelText, roleClass) {
  const position = localToScenePoint(extrinsics.position, frame);
  position.y += 0.012;
  const target = cameraTargetScenePoint(extrinsics, frame);
  const forward = target.clone().sub(position).normalize();

  const group = new THREE.Group();
  group.position.copy(position);

  const stemMaterial = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.7 });
  const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.0035, 0.0035, 0.085, 6), stemMaterial);
  stem.position.y = -0.0425;
  group.add(stem);

  const core = new THREE.Mesh(new THREE.SphereGeometry(0.015, 18, 18), new THREE.MeshBasicMaterial({ color }));
  group.add(core);

  // Cone points along +Y by default; aim it down the camera's view direction.
  const cone = new THREE.Mesh(
    new THREE.ConeGeometry(0.026, 0.075, 22),
    new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.6, roughness: 0.35 }),
  );
  cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), forward);
  cone.position.copy(forward.clone().multiplyScalar(0.05));
  group.add(cone);

  if (labelText) {
    const label = createLabel(labelText, `camera-label ${roleClass}`);
    label.position.set(0, 0.085, 0);
    label.userData.occlusionAnchor = group;
    group.add(label);
  }
  return group;
}

// A small dim glyph for an alternative candidate location from a prior-free
// search; the primary prediction keeps the full marker + label.
function createCandidateMarker(extrinsics, frame, color) {
  const position = localToScenePoint(extrinsics.position, frame);
  position.y += 0.012;
  const target = cameraTargetScenePoint(extrinsics, frame);
  const forward = target.clone().sub(position).normalize();
  const group = new THREE.Group();
  group.position.copy(position);
  group.add(
    new THREE.Mesh(new THREE.SphereGeometry(0.01, 12, 12), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.55 })),
  );
  const cone = new THREE.Mesh(
    new THREE.ConeGeometry(0.016, 0.04, 14),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.45 }),
  );
  cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), forward);
  cone.position.copy(forward.clone().multiplyScalar(0.03));
  group.add(cone);
  return group;
}

// Rebuilds the dynamic layer for the current selection: a small marker for every
// view, the selected view's ground-truth camera + FOV, and the selected solve's
// predicted camera + FOV.
export function buildSelectionLayer(terrain, frame, views, selectedViewId, selectedSolve) {
  const group = new THREE.Group();
  for (const view of views) {
    if (!view.true_extrinsics) {
      continue;
    }
    const isSelected = view.id === selectedViewId;
    if (!isSelected) {
      const marker = createCameraMarker(view.true_extrinsics, frame, TRUE_COLOR, null, "true-camera");
      marker.traverse((object) => {
        if (object.material) {
          object.material.transparent = true;
          object.material.opacity = 0.4;
        }
      });
      group.add(marker);
      continue;
    }
    group.add(createCameraMarker(view.true_extrinsics, frame, TRUE_COLOR, `${view.label} · True`, "true-camera"));
    group.add(createFovOverlay(terrain, frame, view.true_extrinsics, view.intrinsics, TRUE_COLOR));

    if (selectedSolve && view.solves.some((s) => s.id === selectedSolve.id)) {
      const predicted = selectedSolve.result.estimate.extrinsics;
      group.add(createCameraMarker(predicted, frame, PREDICTED_COLOR, "Predicted", "predicted-camera"));
      group.add(createFovOverlay(terrain, frame, predicted, view.intrinsics, PREDICTED_COLOR));
      // Other plausible locations from a prior-free search ("find all of them").
      for (const candidate of (selectedSolve.result.candidates ?? []).slice(1)) {
        group.add(createCandidateMarker(candidate.extrinsics, frame, PREDICTED_COLOR));
      }
    }
  }
  return group;
}

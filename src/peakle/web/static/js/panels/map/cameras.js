"use strict";

// Camera marker glyphs (body + lens + frustum + label) placed at a pose, and the
// per-selection layer that shows the selected view's ground-truth and predicted
// cameras together with their FOV coverage overlays.

import * as THREE from "three";

import { cameraTargetScenePoint, localToScenePoint } from "../../geometry.js";
import { createFovOverlay } from "./fov.js";
import { createLabel } from "./terrain-mesh.js";

export const TRUE_COLOR = 0x3b82f6;
export const PREDICTED_COLOR = 0xff8c42;

export function createCameraMarker(extrinsics, frame, color, labelText, roleClass) {
  const position = localToScenePoint(extrinsics.position, frame);
  position.y += 0.012;
  const target = cameraTargetScenePoint(extrinsics, frame);
  const forward = target.clone().sub(position).normalize();

  const group = new THREE.Group();
  group.position.copy(position);

  const bodyMaterial = new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.4, roughness: 0.45 });
  const lensMaterial = new THREE.MeshStandardMaterial({ color: 0x11130f, emissive: 0x0b1215, emissiveIntensity: 0.25, roughness: 0.25 });

  const marker = new THREE.Group();
  marker.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), forward);
  marker.add(new THREE.Mesh(new THREE.BoxGeometry(0.088, 0.052, 0.04), bodyMaterial));
  const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.021, 0.032, 24), lensMaterial);
  lens.rotation.x = Math.PI / 2;
  lens.position.z = 0.036;
  marker.add(lens);
  marker.add(createFrustumGlyph(color));
  group.add(marker);

  const standMaterial = new THREE.MeshStandardMaterial({ color: 0x0e100d, emissive: color, emissiveIntensity: 0.18, roughness: 0.55 });
  const stand = new THREE.Mesh(new THREE.CylinderGeometry(0.005, 0.006, 0.072, 8), standMaterial);
  stand.position.y = -0.038;
  group.add(stand);

  if (labelText) {
    const label = createLabel(labelText, `camera-label ${roleClass}`);
    label.position.set(0, 0.09, 0);
    label.userData.occlusionAnchor = group;
    group.add(label);
  }
  return group;
}

function createFrustumGlyph(color) {
  const originZ = 0.058;
  const farZ = 0.18;
  const halfWidth = 0.062;
  const halfHeight = 0.04;
  const vertices = new Float32Array([
    0, 0, originZ, -halfWidth, -halfHeight, farZ,
    0, 0, originZ, halfWidth, -halfHeight, farZ,
    0, 0, originZ, halfWidth, halfHeight, farZ,
    0, 0, originZ, -halfWidth, halfHeight, farZ,
    -halfWidth, -halfHeight, farZ, halfWidth, -halfHeight, farZ,
    halfWidth, -halfHeight, farZ, halfWidth, halfHeight, farZ,
    halfWidth, halfHeight, farZ, -halfWidth, halfHeight, farZ,
    -halfWidth, halfHeight, farZ, -halfWidth, -halfHeight, farZ,
  ]);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(vertices, 3));
  return new THREE.LineSegments(
    geometry,
    new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.95, depthWrite: false }),
  );
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
    group.add(createCameraMarker(view.true_extrinsics, frame, TRUE_COLOR, `${view.label} · true`, "true-camera"));
    group.add(createFovOverlay(terrain, frame, view.true_extrinsics, view.intrinsics, TRUE_COLOR));

    if (selectedSolve && view.solves.some((s) => s.id === selectedSolve.id)) {
      const predicted = selectedSolve.result.estimate.extrinsics;
      group.add(createCameraMarker(predicted, frame, PREDICTED_COLOR, "predicted", "predicted-camera"));
      group.add(createFovOverlay(terrain, frame, predicted, view.intrinsics, PREDICTED_COLOR));
    }
  }
  return group;
}

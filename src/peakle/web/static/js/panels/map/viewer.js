"use strict";

// The 3D map: orbitable terrain with peak labels, click-to-place camera creation,
// the selected view's ground-truth + predicted cameras with FOV coverage, and POV
// switching (free orbit / true camera / predicted camera).

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer } from "three/addons/renderers/CSS2DRenderer.js";

import { cameraTargetScenePoint, localToScenePoint, sceneToLocalEastNorth, terrainFrame, verticalFovDeg } from "../../geometry.js";
import { el } from "../../format.js";
import { buildSelectionLayer } from "./cameras.js";
import { addPeakLabels, createTerrainGroup } from "./terrain-mesh.js";

const CAMERA_INITIAL_POSITION = new THREE.Vector3(0.1, 1.55, 2.65);
const CAMERA_TARGET = new THREE.Vector3(0, 0.22, 0);
const PICK_MAX_DRAG_PX = 5;
const LABEL_OCCLUSION_MARGIN = 0.018;

export function setupMapPanel(store, root) {
  root.classList.add("map-panel");
  const canvas = el("canvas", { id: "mapCanvas" });
  const frameBox = el("div", { class: "frame-box", id: "mapFrameBox" }, [canvas]);
  const hint = el("p", { class: "control-hint", id: "mapHint" });
  const povControls = el(
    "div",
    { class: "segmented" },
    ["map", "true", "predicted"].map((povMode) =>
      el("button", {
        type: "button",
        class: povMode === "map" ? "active" : "",
        dataset: { pov: povMode },
        text: povMode === "map" ? "Map" : povMode === "true" ? "True POV" : "Predicted POV",
      }),
    ),
  );
  root.replaceChildren(frameBox, el("div", { class: "map-hud" }, [povControls, hint]));

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
  renderer.setClearColor(0x141512, 1);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x141512, 3.5, 6.4);
  scene.add(new THREE.HemisphereLight(0xdde8ee, 0x263a2d, 1.7));
  const sun = new THREE.DirectionalLight(0xffe4a8, 2.15);
  sun.position.set(-1.6, 2.4, 1.1);
  scene.add(sun);

  const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 20);
  camera.position.copy(CAMERA_INITIAL_POSITION);

  const labelRenderer = new CSS2DRenderer();
  labelRenderer.domElement.className = "label-layer";
  frameBox.append(labelRenderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.copy(CAMERA_TARGET);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 0.08;
  controls.maxDistance = 4.8;
  controls.minPolarAngle = 0.12;
  controls.maxPolarAngle = Math.PI / 2 - 0.05;

  let frame = null;
  let terrainGroup = null;
  let terrainMesh = null;
  let peakLabels = [];
  let selectionLayer = null;
  let mode = "map";
  const raycaster = new THREE.Raycaster();

  function rebuildTerrain() {
    if (!store.terrain) {
      return;
    }
    if (terrainGroup) {
      scene.remove(terrainGroup);
    }
    for (const label of peakLabels) {
      scene.remove(label);
    }
    frame = terrainFrame(store.terrain);
    const built = createTerrainGroup(store.terrain, frame);
    terrainGroup = built.group;
    terrainMesh = built.mesh;
    scene.add(terrainGroup);
    peakLabels = addPeakLabels(scene, store.peaks, frame);
    rebuildSelection();
  }

  function rebuildSelection() {
    if (!frame) {
      return;
    }
    if (selectionLayer) {
      scene.remove(selectionLayer);
    }
    selectionLayer = buildSelectionLayer(store.terrain, frame, store.views, store.selectedViewId, store.selectedSolve());
    scene.add(selectionLayer);
    if (mode !== "map") {
      applyPov(mode);
    }
  }

  function applyPov(nextMode) {
    mode = nextMode;
    syncPovButtons();
    const view = store.selectedView();
    let pose = null;
    if (nextMode === "true" && view?.true_extrinsics) {
      pose = view.true_extrinsics;
    } else if (nextMode === "predicted") {
      const solve = store.selectedSolve();
      pose = solve ? solve.result.estimate.extrinsics : null;
    }
    if (!pose) {
      mode = "map";
      syncPovButtons();
      controls.enabled = true;
      camera.fov = 45;
      camera.position.copy(CAMERA_INITIAL_POSITION);
      controls.target.copy(CAMERA_TARGET);
      controls.update();
      camera.updateProjectionMatrix();
      hint.textContent = store.placing ? "Click the map to place a camera." : "Free orbit. Select a view and choose a POV.";
      return;
    }
    controls.enabled = false;
    camera.fov = verticalFovDeg(view.intrinsics);
    camera.position.copy(localToScenePoint(pose.position, frame));
    camera.up.set(0, 1, 0);
    camera.lookAt(cameraTargetScenePoint(pose, frame));
    camera.updateProjectionMatrix();
    hint.textContent = `Looking through the ${nextMode} camera of ${view.label}.`;
  }

  function syncPovButtons() {
    for (const button of root.querySelectorAll("[data-pov]")) {
      button.classList.toggle("active", button.dataset.pov === mode);
    }
  }

  for (const button of root.querySelectorAll("[data-pov]")) {
    button.addEventListener("click", () => applyPov(button.dataset.pov));
  }

  // Click-to-place: raycast the terrain and create a view aimed at the tallest peak.
  let pointerDown = null;
  canvas.addEventListener("pointerdown", (event) => {
    pointerDown = { x: event.clientX, y: event.clientY };
  });
  canvas.addEventListener("pointerup", (event) => {
    const start = pointerDown;
    pointerDown = null;
    if (!start || !store.placing || mode !== "map" || !terrainMesh) {
      return;
    }
    if (Math.hypot(event.clientX - start.x, event.clientY - start.y) > PICK_MAX_DRAG_PX) {
      return;
    }
    placeAt(event);
  });

  function placeAt(event) {
    const rect = canvas.getBoundingClientRect();
    const pointer = new THREE.Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1,
    );
    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObject(terrainMesh, false);
    if (!hits.length) {
      return;
    }
    const local = sceneToLocalEastNorth(hits[0].point, frame);
    const yaw = headingToTallestPeak(local);
    // Turn off placing immediately so a second click during the in-flight render
    // does not drop a second camera.
    store.setPlacing(false);
    store.createView({ east_m: local.east_m, north_m: local.north_m, yaw_deg: yaw, pitch_deg: 3.0, eye_height_m: 150.0 }).catch((error) => {
      hint.textContent = error.message;
    });
  }

  function headingToTallestPeak(local) {
    if (!store.peaks.length) {
      return 0;
    }
    const peak = store.peaks.reduce((tallest, candidate) => (candidate.elevation_m > tallest.elevation_m ? candidate : tallest));
    return (Math.atan2(peak.local_position.east_m - local.east_m, peak.local_position.north_m - local.north_m) * 180) / Math.PI;
  }

  function updateLabelOcclusion() {
    const cameraPosition = new THREE.Vector3().setFromMatrixPosition(camera.matrixWorld);
    const target = new THREE.Vector3();
    const direction = new THREE.Vector3();
    const projected = new THREE.Vector3();
    for (const label of peakLabels) {
      const anchor = label.userData.occlusionAnchor ?? label;
      anchor.getWorldPosition(target);
      projected.copy(target).project(camera);
      const inView = projected.z >= -1 && projected.z <= 1 && Math.abs(projected.x) <= 1.08 && Math.abs(projected.y) <= 1.08;
      let occluded = false;
      if (inView && terrainMesh) {
        const distance = cameraPosition.distanceTo(target);
        const rayLength = distance - LABEL_OCCLUSION_MARGIN;
        if (rayLength > camera.near) {
          direction.copy(target).sub(cameraPosition).normalize();
          raycaster.set(cameraPosition, direction);
          raycaster.near = camera.near;
          raycaster.far = rayLength;
          occluded = raycaster.intersectObject(terrainMesh, false).length > 0;
        }
      }
      label.visible = inView && !occluded;
    }
  }

  function resize() {
    const width = Math.max(1, frameBox.clientWidth);
    const height = Math.max(1, frameBox.clientHeight);
    renderer.setSize(width, height, false);
    labelRenderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }

  new ResizeObserver(resize).observe(frameBox);
  resize();

  renderer.setAnimationLoop(() => {
    if (controls.enabled) {
      controls.update();
    }
    camera.updateMatrixWorld();
    scene.updateMatrixWorld();
    updateLabelOcclusion();
    renderer.render(scene, camera);
    labelRenderer.render(scene, camera);
  });

  store.on("scene", rebuildTerrain);
  store.on("views", rebuildSelection);
  store.on("selection", rebuildSelection);
  store.on("placing", () => {
    if (mode === "map") {
      hint.textContent = store.placing ? "Click the map to place a camera." : "Free orbit. Select a view and choose a POV.";
    }
  });
  rebuildTerrain();
  syncPovButtons();
}

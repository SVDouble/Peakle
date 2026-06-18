"use strict";

import * as THREE from "three";
import { createDockview, themeAbyss } from "dockview-core";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DObject, CSS2DRenderer } from "three/addons/renderers/CSS2DRenderer.js";

const TERRAIN_WIDTH = 2.35;
const TERRAIN_DEPTH = 1.72;
const TERRAIN_HEIGHT = 0.66;
const CAMERA_INITIAL_POSITION = new THREE.Vector3(0.1, 1.55, 2.65);
const CAMERA_TARGET = new THREE.Vector3(0, 0.22, 0);
const CAMERA_LOOK_DISTANCE_M = 3600;
const MAX_PEAK_LABELS = 8;
const LABEL_OCCLUSION_MARGIN = 0.018;
const LABEL_VIEW_PADDING = 1.08;
const CAMERA_PICK_MAX_DRAG_PX = 5;
const FOV_OVERLAY_VERTICAL_OFFSET = 0.011;
const PREDICTED_CAMERA_COMPACT_THRESHOLD_M = 180;
const ISOLINE_INTERVAL_M = 250;
const ISOLINE_VERTICAL_OFFSET = 0.0014;
const RASTER_EPSILON = 1e-8;
const SCALE_CANDIDATES_M = [100, 200, 500, 1000, 2000, 5000, 10000];
const SCALE_MIN_WIDTH_PX = 64;
const SCALE_MAX_WIDTH_PX = 150;
const VISIBILITY_RASTER_MAX_WIDTH = 820;
const FOV_TRUE_COLOR = 0x3b82f6;
const FOV_ESTIMATED_COLOR = 0xff3b30;

// The viewport is either the free-orbit "Map" or a locked "Camera POV". The POV
// embodies one camera role (true or estimated). These replace the former
// Compare/Orbit/True/Estimated set, which had two redundant free-orbit modes.
const CAMERA_MODES = {
  map: "map",
  pov: "pov",
};
const CAMERA_ROLES = {
  true: "true",
  estimated: "estimated",
};

const PANEL_TEMPLATES = {
  map: "tpl-map",
  camera: "tpl-camera",
  controls: "tpl-controls",
  peaks: "tpl-peaks",
  alignment: "tpl-alignment",
};

const layout = buildLayout();

const canvas = document.getElementById("terrainCanvas");
const statusBox = document.getElementById("webglStatus");
const cameraCanvas = document.getElementById("cameraCanvas");
const cameraOverlay = document.getElementById("cameraOverlay");

main().catch((error) => {
  statusBox.hidden = false;
  statusBox.textContent = `Viewer failed to load: ${error.message}`;
});

function buildLayout() {
  const root = document.getElementById("layoutRoot");
  const dockview = createDockview(root, {
    theme: themeAbyss,
    // Keep every panel mounted so background renderers (the camera image, the
    // alignment lab) keep running and keep their DOM/handlers when not focused.
    defaultRenderer: "always",
    createComponent: (options) => {
      const element = document.createElement("div");
      element.className = "panel-host";
      return {
        element,
        init: () => element.append(document.getElementById(PANEL_TEMPLATES[options.name]).content.cloneNode(true)),
      };
    },
  });

  const bounds = root.getBoundingClientRect();
  const width = Math.max(640, bounds.width);
  const height = Math.max(480, bounds.height);
  const panel = (id, component, title) => ({ id, contentComponent: component, title });
  const leaf = (size, ...panels) => ({
    type: "leaf",
    size,
    data: { views: panels.map((entry) => entry.id), activeView: panels[0].id, id: `group-${panels[0].id}` },
    panels,
  });
  const groups = [
    leaf(Math.round(width * 0.6), panel("map", "map", "Map"), panel("alignment", "alignment", "Alignment Lab")),
    {
      type: "branch",
      size: Math.round(width * 0.4),
      data: [
        leaf(Math.round(height * 0.48), panel("camera", "camera", "Camera")),
        leaf(Math.round(height * 0.34), panel("controls", "controls", "Camera View")),
        leaf(Math.round(height * 0.18), panel("peaks", "peaks", "Peaks")),
      ],
    },
  ];
  dockview.fromJSON({
    grid: {
      orientation: "HORIZONTAL",
      width,
      height,
      root: { type: "branch", size: height, data: groups },
    },
    panels: Object.fromEntries(groups.flatMap(collectLeafPanels).map((entry) => [entry.id, entry])),
    activeGroup: "group-map",
  });
  return dockview;
}

function collectLeafPanels(node) {
  return node.type === "leaf" ? node.panels : node.data.flatMap(collectLeafPanels);
}

async function main() {
  const response = await fetch("viewer-data.json");
  const data = await response.json();
  const appState = {
    activeView: null,
    data,
    views: data.views,
  };
  validateViews(appState.views);
  appState.activeView = appState.views[0];
  populatePanels(appState);
  setupComparisonPanel(appState.activeView);
  setupTerrainViewer(data, appState);
  // Dockview only mounts a tab's content when it is first shown, so defer the
  // lab until its panel exists in the DOM.
  initAlignmentLabWhenReady(data);
}

function initAlignmentLabWhenReady(data) {
  let started = false;
  const tryStart = () => {
    if (started || !document.getElementById("algoSelect")) {
      return;
    }
    started = true;
    setupAlignmentLab(data);
  };
  layout.onDidActivePanelChange(tryStart);
  tryStart();
}

function validateViews(views) {
  if (!Array.isArray(views) || views.length === 0) {
    throw new Error("viewer-data.json must contain a non-empty views array");
  }
}

function populatePanels(appState) {
  const data = appState.data;
  const view = appState.activeView;
  const visible = view.annotations.filter((annotation) => annotation.visible);
  const metrics = view.pose_estimate.metrics;
  document.getElementById("summary").textContent =
    `${data.peaks.length} synthetic peaks, ${visible.length} labels accepted, ${view.label}`;
  document.getElementById("positionError").textContent =
    formatNumber(metrics.position_error_m, "m");
  document.getElementById("yawError").textContent = formatNumber(metrics.yaw_error_deg, "deg");
  document.getElementById("contourMae").textContent = formatNumber(metrics.contour_mae_px, "px");
  document.getElementById("labelCount").textContent = String(visible.length);

  const peakList = document.getElementById("peakList");
  peakList.replaceChildren();
  for (const annotation of visible) {
    const peak = data.peaks.find((item) => item.id === annotation.peak_id);
    const li = document.createElement("li");
    const name = document.createElement("span");
    const elevation = document.createElement("span");
    name.textContent = annotation.peak_name;
    elevation.textContent = peak ? `${Math.round(peak.elevation_m)} m` : "";
    li.append(name, elevation);
    peakList.append(li);
  }
}

function refreshActiveViewPanels(appState) {
  populatePanels(appState);
  setupComparisonPanel(appState.activeView);
}

function setupComparisonPanel(view) {
  const trueCamera = view.true_camera;
  const estimatedCamera = view.pose_estimate.extrinsics;
  document.getElementById("comparePositionDelta").textContent = formatNumber(
    distanceMeters(trueCamera.position, estimatedCamera.position),
    "m",
  );
  document.getElementById("compareYawDelta").textContent = formatNumber(
    angleDeltaDeg(trueCamera.yaw_deg, estimatedCamera.yaw_deg),
    "deg",
  );
  document.getElementById("comparePitchDelta").textContent = formatNumber(
    Math.abs(trueCamera.pitch_deg - estimatedCamera.pitch_deg),
    "deg",
  );
}

// --- Live camera-image panel -------------------------------------------------
// Renders the right-hand image on the fly from the current viewport pose using
// the true pinhole intrinsics, then overlays the live skyline contour and peak
// labels. Replaces the former static render/annotated/contour PNG toggle.

const SVG_NS = "http://www.w3.org/2000/svg";
const CAMERA_OVERLAY_SAMPLE_WIDTH = 260;
const PEAK_OCCLUSION_MARGIN_M = 0.01;

function createCameraImagePanel(terrain, frame, data, mainTerrainMesh) {
  const intrinsics = data.scene.intrinsics;
  const aspect = intrinsics.width_px / intrinsics.height_px;

  const renderer = new THREE.WebGLRenderer({ canvas: cameraCanvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0xacc4d6, 1);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xacc4d6);
  scene.fog = new THREE.Fog(0xacc4d6, 2.8, 7.4);
  scene.add(new THREE.HemisphereLight(0xeaf1f6, 0x34402f, 1.7));
  const sun = new THREE.DirectionalLight(0xffe4a8, 2.2);
  sun.position.set(-1.6, 2.4, 1.1);
  scene.add(sun);
  scene.add(new THREE.Mesh(mainTerrainMesh.geometry, mainTerrainMesh.material));

  const camera = new THREE.PerspectiveCamera(verticalFovDeg(intrinsics), aspect, 0.02, 30);

  const gridPoints = new Array(terrain.grid_width * terrain.grid_height);
  for (let row = 0; row < terrain.grid_height; row += 1) {
    for (let col = 0; col < terrain.grid_width; col += 1) {
      gridPoints[terrainVertexIndex(terrain, row, col)] = localToScenePoint(
        terrainLocalGridPoint(terrain, frame, row, col),
        frame,
      );
    }
  }

  const panel = {
    aspect,
    fovDeg: verticalFovDeg(intrinsics),
    camera,
    frame,
    gridPoints,
    lastPose: new THREE.Matrix4().makeScale(0, 0, 0),
    occluder: new THREE.Mesh(mainTerrainMesh.geometry, new THREE.MeshBasicMaterial()),
    overlayHeight: 1,
    overlayWidth: 1,
    peaks: data.peaks.map((peak) => ({ name: peak.name, point: localToScenePoint(peak.local_position, frame) })),
    raycaster: new THREE.Raycaster(),
    renderer,
    sampleHeight: Math.round(CAMERA_OVERLAY_SAMPLE_WIDTH / aspect),
    sampleWidth: CAMERA_OVERLAY_SAMPLE_WIDTH,
    scene,
    terrain,
  };
  panel.resize = () => resizeCameraImage(panel);
  panel.update = (mainCamera) => updateCameraImage(panel, mainCamera);
  panel.resize();
  return panel;
}

function verticalFovDeg(intrinsics) {
  return THREE.MathUtils.radToDeg(2 * Math.atan(intrinsics.height_px / 2 / intrinsics.focal_length_px));
}

function resizeCameraImage(panel) {
  const frame = cameraCanvas.parentElement.parentElement;
  const box = fitContainBox(frame.clientWidth, frame.clientHeight, panel.aspect);
  setBox(cameraCanvas.parentElement, box);
  panel.renderer.setSize(box.width, box.height, false);
  panel.overlayWidth = box.width;
  panel.overlayHeight = box.height;
  panel.camera.aspect = panel.aspect;
  panel.camera.updateProjectionMatrix();
  panel.sampleWidth = CAMERA_OVERLAY_SAMPLE_WIDTH;
  panel.sampleHeight = Math.max(1, Math.round(CAMERA_OVERLAY_SAMPLE_WIDTH / panel.aspect));
  cameraOverlay.setAttribute("viewBox", `0 0 ${box.width} ${box.height}`);
  panel.lastPose.makeScale(0, 0, 0);
}

// Largest box of the given aspect ratio that fits inside the container, centered
// so the leftover container background reads as the camera frame mask.
function fitContainBox(containerWidth, containerHeight, aspect) {
  const width = Math.max(1, containerWidth);
  const height = Math.max(1, containerHeight);
  let boxWidth = width;
  let boxHeight = width / aspect;
  if (boxHeight > height) {
    boxHeight = height;
    boxWidth = height * aspect;
  }
  return {
    width: Math.max(1, Math.round(boxWidth)),
    height: Math.max(1, Math.round(boxHeight)),
    left: Math.round((width - boxWidth) / 2),
    top: Math.round((height - boxHeight) / 2),
  };
}

function setBox(element, box) {
  element.style.left = `${box.left}px`;
  element.style.top = `${box.top}px`;
  element.style.width = `${box.width}px`;
  element.style.height = `${box.height}px`;
}

// Letterboxes the map render to the pinhole frame while in Camera POV; in Map
// mode the render fills the panel.
function layoutMapFrame(viewer) {
  const panelElement = viewer.mapFrameBox.parentElement;
  const width = panelElement.clientWidth;
  const height = panelElement.clientHeight;
  if (viewer.mode === CAMERA_MODES.pov) {
    setBox(viewer.mapFrameBox, fitContainBox(width, height, viewer.cameraImage.aspect));
    panelElement.classList.add("pov-framed");
  } else {
    setBox(viewer.mapFrameBox, { left: 0, top: 0, width: Math.max(1, width), height: Math.max(1, height) });
    panelElement.classList.remove("pov-framed");
  }
}

function setScaleHudVisible(visible) {
  const hud = document.querySelector(".map-scale-hud");
  if (hud) {
    hud.hidden = !visible;
  }
}

// First-person look: rotate the view direction about a fixed camera position.
function setupPovLook(viewer) {
  let drag = null;
  canvas.addEventListener("pointerdown", (event) => {
    if (viewer.mode !== CAMERA_MODES.pov || event.button !== 0) {
      return;
    }
    drag = { x: event.clientX, y: event.clientY };
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!drag) {
      return;
    }
    viewer.povAzimuth += (event.clientX - drag.x) * 0.0032;
    viewer.povElevation = THREE.MathUtils.clamp(viewer.povElevation - (event.clientY - drag.y) * 0.0032, -1.2, 1.2);
    drag = { x: event.clientX, y: event.clientY };
    applyPovOrientation(viewer);
  });
  const release = (event) => {
    if (drag) {
      canvas.releasePointerCapture?.(event.pointerId);
      drag = null;
    }
  };
  canvas.addEventListener("pointerup", release);
  canvas.addEventListener("pointercancel", release);
}

function initPovOrientation(viewer, config) {
  const forward = config.target.clone().sub(config.position).normalize();
  viewer.povElevation = Math.asin(THREE.MathUtils.clamp(forward.y, -1, 1));
  viewer.povAzimuth = Math.atan2(forward.x, -forward.z);
  applyPovOrientation(viewer);
}

function applyPovOrientation(viewer) {
  const cosElevation = Math.cos(viewer.povElevation);
  const direction = new THREE.Vector3(
    cosElevation * Math.sin(viewer.povAzimuth),
    Math.sin(viewer.povElevation),
    -cosElevation * Math.cos(viewer.povAzimuth),
  );
  viewer.camera.up.set(0, 1, 0);
  viewer.camera.lookAt(viewer.camera.position.clone().add(direction));
}

function updateCameraImage(panel, mainCamera) {
  mainCamera.updateMatrixWorld();
  panel.camera.position.setFromMatrixPosition(mainCamera.matrixWorld);
  panel.camera.quaternion.setFromRotationMatrix(mainCamera.matrixWorld);
  panel.camera.updateMatrixWorld(true);
  panel.renderer.render(panel.scene, panel.camera);

  if (!matricesClose(panel.camera.matrixWorld, panel.lastPose)) {
    panel.lastPose.copy(panel.camera.matrixWorld);
    drawCameraOverlay(panel);
  }
}

function drawCameraOverlay(panel) {
  const elements = [skylinePathElement(panel), ...peakLabelElements(panel)].filter(Boolean);
  cameraOverlay.replaceChildren(...elements);
}

function skylinePathElement(panel) {
  const skyline = extractSkyline(panel);
  if (skyline.length < 2) {
    return null;
  }
  const scaleX = panel.overlayWidth / panel.sampleWidth;
  const scaleY = panel.overlayHeight / panel.sampleHeight;
  const points = skyline.map((point) => `${(point.x * scaleX).toFixed(1)},${(point.y * scaleY).toFixed(1)}`);
  return svgElement("polyline", { class: "skyline", points: points.join(" ") });
}

function extractSkyline(panel) {
  const { terrain, camera, sampleWidth: width, sampleHeight: height } = panel;
  camera.updateMatrixWorld();
  const projected = projectGridToSample(panel);
  const topRow = new Float32Array(width).fill(Number.POSITIVE_INFINITY);

  for (let row = 0; row < terrain.grid_height - 1; row += 1) {
    for (let col = 0; col < terrain.grid_width - 1; col += 1) {
      const a = projected[terrainVertexIndex(terrain, row, col)];
      const b = projected[terrainVertexIndex(terrain, row, col + 1)];
      const c = projected[terrainVertexIndex(terrain, row + 1, col)];
      const d = projected[terrainVertexIndex(terrain, row + 1, col + 1)];
      rasterizeTopEdge(topRow, width, height, a, b, c);
      rasterizeTopEdge(topRow, width, height, b, d, c);
    }
  }

  const points = [];
  for (let x = 0; x < width; x += 1) {
    if (Number.isFinite(topRow[x])) {
      points.push({ x: x + 0.5, y: topRow[x] });
    }
  }
  return points;
}

function projectGridToSample(panel) {
  const { terrain, camera, gridPoints, sampleWidth: width, sampleHeight: height } = panel;
  const projected = new Array(gridPoints.length);
  const ndc = new THREE.Vector3();
  const view = new THREE.Vector3();
  for (let index = 0; index < gridPoints.length; index += 1) {
    view.copy(gridPoints[index]).applyMatrix4(camera.matrixWorldInverse);
    if (view.z >= -camera.near) {
      projected[index] = { front: false, px: 0, py: 0 };
      continue;
    }
    ndc.copy(gridPoints[index]).project(camera);
    projected[index] = {
      front: true,
      px: (ndc.x * 0.5 + 0.5) * width,
      py: (1 - (ndc.y * 0.5 + 0.5)) * height,
    };
  }
  return projected;
}

function rasterizeTopEdge(topRow, width, height, a, b, c) {
  if (!a.front || !b.front || !c.front) {
    return;
  }
  const minX = Math.max(0, Math.floor(Math.min(a.px, b.px, c.px)));
  const maxX = Math.min(width - 1, Math.ceil(Math.max(a.px, b.px, c.px)));
  const minRow = Math.max(0, Math.floor(Math.min(a.py, b.py, c.py)));
  const maxRow = Math.min(height - 1, Math.ceil(Math.max(a.py, b.py, c.py)));
  if (minX > maxX || minRow > maxRow) {
    return;
  }
  const denominator = (b.py - c.py) * (a.px - c.px) + (c.px - b.px) * (a.py - c.py);
  if (Math.abs(denominator) <= RASTER_EPSILON) {
    return;
  }

  for (let x = minX; x <= maxX; x += 1) {
    const sampleX = x + 0.5;
    const ceiling = Math.min(maxRow, topRow[x]);
    for (let y = minRow; y <= ceiling; y += 1) {
      const sampleY = y + 0.5;
      const weightA = ((b.py - c.py) * (sampleX - c.px) + (c.px - b.px) * (sampleY - c.py)) / denominator;
      const weightB = ((c.py - a.py) * (sampleX - c.px) + (a.px - c.px) * (sampleY - c.py)) / denominator;
      const weightC = 1 - weightA - weightB;
      if (weightA >= -RASTER_EPSILON && weightB >= -RASTER_EPSILON && weightC >= -RASTER_EPSILON) {
        topRow[x] = y;
        break;
      }
    }
  }
}

function peakLabelElements(panel) {
  const { camera } = panel;
  camera.updateMatrixWorld();
  const ndc = new THREE.Vector3();
  const view = new THREE.Vector3();
  const elements = [];
  for (const peak of panel.peaks) {
    view.copy(peak.point).applyMatrix4(camera.matrixWorldInverse);
    if (view.z >= -camera.near || peakOccluded(panel, peak.point)) {
      continue;
    }
    ndc.copy(peak.point).project(camera);
    if (Math.abs(ndc.x) > 1 || Math.abs(ndc.y) > 1) {
      continue;
    }
    const x = (ndc.x * 0.5 + 0.5) * panel.overlayWidth;
    const y = (1 - (ndc.y * 0.5 + 0.5)) * panel.overlayHeight;
    elements.push(svgElement("circle", { class: "peak-dot", cx: x.toFixed(1), cy: y.toFixed(1), r: "3" }));
    const text = svgElement("text", { class: "peak-text", x: (x + 6).toFixed(1), y: (y - 5).toFixed(1) });
    text.textContent = peak.name;
    elements.push(text);
  }
  return elements;
}

function peakOccluded(panel, point) {
  const origin = panel.camera.position;
  const distance = origin.distanceTo(point);
  panel.raycaster.set(origin, point.clone().sub(origin).normalize());
  panel.raycaster.near = panel.camera.near;
  panel.raycaster.far = distance - PEAK_OCCLUSION_MARGIN_M;
  if (panel.raycaster.far <= panel.raycaster.near) {
    return false;
  }
  return panel.raycaster.intersectObject(panel.occluder, false).length > 0;
}

function svgElement(name, attributes) {
  const element = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attributes)) {
    element.setAttribute(key, value);
  }
  return element;
}

function matricesClose(a, b) {
  const ae = a.elements;
  const be = b.elements;
  for (let index = 0; index < 16; index += 1) {
    if (Math.abs(ae[index] - be[index]) > 1e-5) {
      return false;
    }
  }
  return true;
}

function setupTerrainViewer(data, appState) {
  const terrain = data.terrain;
  const frame = terrainFrame(terrain);
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    powerPreference: "high-performance",
  });
  renderer.setClearColor(0x141512, 1);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x141512, 3.5, 6.4);

  const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 20);
  camera.position.copy(CAMERA_INITIAL_POSITION);

  const labelRenderer = new CSS2DRenderer();
  labelRenderer.domElement.className = "label-layer";
  canvas.parentElement.append(labelRenderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  configureOrbitControls(controls);

  scene.add(new THREE.HemisphereLight(0xdde8ee, 0x263a2d, 1.7));
  const sun = new THREE.DirectionalLight(0xffe4a8, 2.15);
  sun.position.set(-1.6, 2.4, 1.1);
  scene.add(sun);

  const terrainMesh = createTerrainMesh(terrain);
  const terrainGroup = new THREE.Group();
  terrainGroup.add(terrainMesh);
  terrainGroup.add(createIsolines(terrain));
  terrainGroup.add(createBasePlane());
  scene.add(terrainGroup);

  const cameraImage = createCameraImagePanel(terrain, frame, data, terrainMesh);

  addPeakLabels(scene, data, frame);

  const viewConfigs = cameraConfigsFor(appState.views, frame);
  const overlays = addFovOverlays(scene, data, frame, viewConfigs);
  const cameraMarkers = addCameraMarkers(scene, viewConfigs);
  const cameraPickTargets = collectCameraPickTargets(cameraMarkers);
  const connectors = addCameraConnectors(scene, viewConfigs);
  camera.updateMatrixWorld(true);
  scene.updateMatrixWorld(true);

  const mapFrameBox = document.getElementById("mapFrameBox");
  const viewer = {
    activeViewId: appState.activeView.id,
    appState,
    camera,
    controls,
    viewConfigs,
    overlays,
    cameraMarkers,
    cameraPickTargets,
    connectors,
    labelOcclusion: createLabelOcclusion(scene, terrainMesh),
    scaleHud: createScaleHud(terrain, frame),
    cameraImage,
    mapFrameBox,
    baseFov: camera.fov,
    povAzimuth: 0,
    povElevation: 0,
    resizeMap: () => {
      layoutMapFrame(viewer);
      resizeRenderers(renderer, labelRenderer, camera);
    },
    mode: CAMERA_MODES.map,
    povRole: CAMERA_ROLES.true,
    selectedFootprint: null,
  };

  setupMarkerPicking(viewer);
  setupPovLook(viewer);
  setupCameraModeControls(viewer);
  applyCameraMode(viewer, CAMERA_MODES.map);

  const resizeObserver = new ResizeObserver(() => {
    viewer.resizeMap();
    cameraImage.resize();
  });
  resizeObserver.observe(mapFrameBox.parentElement);
  resizeObserver.observe(cameraCanvas.parentElement.parentElement);
  viewer.resizeMap();

  renderer.setAnimationLoop(() => {
    if (controls.enabled) {
      controls.update();
    }
    camera.updateMatrixWorld();
    scene.updateMatrixWorld();
    updateLabelOcclusion(viewer.labelOcclusion, camera);
    updateScaleHud(viewer.scaleHud, camera);
    renderer.render(scene, camera);
    labelRenderer.render(scene, camera);
    cameraImage.update(camera);
  });
}

function configureOrbitControls(controls) {
  controls.target.copy(CAMERA_TARGET);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 0.08;
  controls.maxDistance = 4.8;
  controls.minPolarAngle = 0.12;
  controls.maxPolarAngle = Math.PI / 2 - 0.05;
  controls.maxAzimuthAngle = Infinity;
  controls.minAzimuthAngle = -Infinity;
  controls.update();
}

function setupCameraModeControls(viewer) {
  const viewSelect = document.getElementById("viewSelect");
  viewSelect.replaceChildren(
    ...viewer.viewConfigs.map((viewConfig) => new Option(viewConfig.label, viewConfig.id)),
  );
  viewSelect.value = viewer.activeViewId;
  viewSelect.addEventListener("change", () => {
    activateViewerView(viewer, viewSelect.value);
    applyCameraMode(viewer, viewer.mode);
  });

  for (const button of document.querySelectorAll("[data-camera-mode]")) {
    button.addEventListener("click", () => applyCameraMode(viewer, button.dataset.cameraMode));
  }
  for (const button of document.querySelectorAll("[data-pov-role]")) {
    button.addEventListener("click", () => {
      viewer.povRole = button.dataset.povRole;
      applyCameraMode(viewer, CAMERA_MODES.pov);
    });
  }
}

function applyCameraMode(viewer, mode) {
  viewer.mode = mode;
  viewer.selectedFootprint = null;
  syncSegmented("[data-camera-mode]", "cameraMode", mode);
  syncSegmented("[data-pov-role]", "povRole", viewer.povRole);
  document.getElementById("povRoleControls").hidden = mode !== CAMERA_MODES.pov;
  document.getElementById("viewSelect").value = viewer.activeViewId;

  setScaleHudVisible(mode !== CAMERA_MODES.pov);
  for (const marker of viewer.cameraMarkers) {
    marker.visible = mode !== CAMERA_MODES.pov;
  }

  const viewConfig = activeViewConfig(viewer);
  if (mode === CAMERA_MODES.pov) {
    const config = viewConfig[viewer.povRole];
    viewer.controls.enabled = false;
    viewer.camera.fov = viewer.cameraImage.fovDeg;
    viewer.camera.position.copy(config.position);
    initPovOrientation(viewer, config);
    setOverlayVisibility(viewer, [viewer.povRole]);
    setConnectorVisibility(viewer, false);
    setCameraReadout(
      `${viewConfig.label}: ${config.label}`,
      "Locked to the camera position. Drag to look around; black bars mark the camera frame.",
    );
  } else {
    viewer.controls.enabled = true;
    viewer.camera.fov = viewer.baseFov;
    viewer.camera.position.copy(CAMERA_INITIAL_POSITION);
    viewer.controls.target.copy(CAMERA_TARGET);
    viewer.controls.update();
    setOverlayVisibility(viewer, [CAMERA_ROLES.true, CAMERA_ROLES.estimated]);
    setConnectorVisibility(viewer, true);
    setCameraReadout(`${viewConfig.label}: Map`, "Free orbit; blue is true camera coverage, amber is estimated.");
  }

  viewer.resizeMap();
  updateMarkerSelection(viewer);
}

function syncSegmented(selector, datasetKey, value) {
  for (const button of document.querySelectorAll(selector)) {
    button.classList.toggle("active", button.dataset[datasetKey] === value);
  }
}

function setCameraReadout(title, description) {
  document.getElementById("selectedCamera").textContent = title;
  document.getElementById("selectedCameraDescription").textContent = description;
}

function activeViewConfig(viewer) {
  return viewer.viewConfigs.find((viewConfig) => viewConfig.id === viewer.activeViewId) ?? viewer.viewConfigs[0];
}

function setOverlayVisibility(viewer, visibleRoles) {
  const visible = new Set(visibleRoles);
  for (const viewConfig of viewer.viewConfigs) {
    const viewOverlays = viewer.overlays[viewConfig.id];
    viewOverlays.true.visible = viewConfig.id === viewer.activeViewId && visible.has(CAMERA_ROLES.true);
    viewOverlays.estimated.visible = viewConfig.id === viewer.activeViewId && visible.has(CAMERA_ROLES.estimated);
  }
}

function setConnectorVisibility(viewer, visible) {
  for (const [viewId, connector] of Object.entries(viewer.connectors)) {
    connector.visible = visible && viewId === viewer.activeViewId;
  }
}

function createTerrainMesh(terrain) {
  const geometry = terrainGeometry(terrain);
  geometry.computeVertexNormals();
  return new THREE.Mesh(geometry, createTerrainMaterial());
}

// Hypsometric (elevation) gradient with soft contour banding, injected into a
// standard lit material. The elevation tint plus the banded "pattern" makes
// overlapping ridges separable on the flat camera image, where plain shading
// left similar-height mountains looking identical.
function createTerrainMaterial() {
  const material = new THREE.MeshStandardMaterial({ roughness: 0.9, metalness: 0, side: THREE.FrontSide });
  material.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", "#include <common>\nvarying float vTerrainElevation;")
      .replace("#include <begin_vertex>", "#include <begin_vertex>\nvTerrainElevation = position.y / float(" + TERRAIN_HEIGHT.toFixed(4) + ");");
    shader.fragmentShader = shader.fragmentShader
      .replace("#include <common>", `#include <common>
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
}`)
      .replace("#include <color_fragment>", `#include <color_fragment>
{
  float terrainT = clamp(vTerrainElevation, 0.0, 1.0);
  vec3 hypso = peakleHypsometric(terrainT);
  float bandEdge = fract(terrainT * 9.0);
  float band = smoothstep(0.0, 0.08, bandEdge) * smoothstep(1.0, 0.92, bandEdge);
  hypso *= mix(0.8, 1.0, band);
  diffuseColor.rgb = hypso;
}`);
  };
  return material;
}

// Marching-squares contour lines drawn directly on the terrain surface (a small
// lift only avoids z-fighting), so they hug the slopes instead of floating above
// them as the old constant-height ribbons did.
function createIsolines(terrain) {
  const positions = [];
  const firstLevel = Math.ceil(terrain.elevation_min_m / ISOLINE_INTERVAL_M) * ISOLINE_INTERVAL_M;
  for (let level = firstLevel; level <= terrain.elevation_max_m; level += ISOLINE_INTERVAL_M) {
    addIsolineLevel(positions, terrain, level);
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  const lines = new THREE.LineSegments(
    geometry,
    new THREE.LineBasicMaterial({
      color: 0xe7dab2,
      transparent: true,
      opacity: 0.26,
      depthWrite: false,
    }),
  );
  lines.renderOrder = 1;
  return lines;
}

function addIsolineLevel(positions, terrain, level) {
  for (let row = 0; row < terrain.grid_height - 1; row += 1) {
    for (let col = 0; col < terrain.grid_width - 1; col += 1) {
      const corners = [
        terrainSample(terrain, row, col),
        terrainSample(terrain, row, col + 1),
        terrainSample(terrain, row + 1, col + 1),
        terrainSample(terrain, row + 1, col),
      ];
      const intersections = [
        edgeLevelIntersection(corners[0], corners[1], level),
        edgeLevelIntersection(corners[1], corners[2], level),
        edgeLevelIntersection(corners[2], corners[3], level),
        edgeLevelIntersection(corners[3], corners[0], level),
      ].filter(Boolean);
      const unique = deduplicateIntersections(intersections);
      if (unique.length === 2) {
        pushIsolineSegment(positions, terrain, unique[0], unique[1]);
      } else if (unique.length === 4) {
        pushIsolineSegment(positions, terrain, unique[0], unique[1]);
        pushIsolineSegment(positions, terrain, unique[2], unique[3]);
      }
    }
  }
}

function terrainSample(terrain, row, col) {
  return {
    row,
    col,
    elevation: terrain.elevation_m[row][col],
  };
}

function edgeLevelIntersection(a, b, level) {
  const aDelta = a.elevation - level;
  const bDelta = b.elevation - level;
  if (aDelta === 0 && bDelta === 0) {
    return null;
  }
  if (aDelta * bDelta > 0) {
    return null;
  }
  const t = aDelta === bDelta ? 0 : aDelta / (aDelta - bDelta);
  if (t < 0 || t > 1) {
    return null;
  }
  return {
    row: interpolate(a.row, b.row, t),
    col: interpolate(a.col, b.col, t),
    elevation: level,
  };
}

function deduplicateIntersections(points) {
  const unique = [];
  for (const point of points) {
    if (
      !unique.some(
        (candidate) =>
          Math.abs(candidate.row - point.row) < 1e-5 &&
          Math.abs(candidate.col - point.col) < 1e-5,
      )
    ) {
      unique.push(point);
    }
  }
  return unique;
}

function pushIsolineSegment(positions, terrain, a, b) {
  const aPoint = terrainFloatPoint(terrain, a.row, a.col, a.elevation, ISOLINE_VERTICAL_OFFSET);
  const bPoint = terrainFloatPoint(terrain, b.row, b.col, b.elevation, ISOLINE_VERTICAL_OFFSET);
  positions.push(aPoint.x, aPoint.y, aPoint.z, bPoint.x, bPoint.y, bPoint.z);
}

function terrainGeometry(terrain) {
  const gridWidth = terrain.grid_width;
  const gridHeight = terrain.grid_height;
  const positions = new Float32Array(gridWidth * gridHeight * 3);
  const indices = [];

  for (let row = 0; row < gridHeight; row += 1) {
    for (let col = 0; col < gridWidth; col += 1) {
      const offset = (row * gridWidth + col) * 3;
      const point = terrainGridPoint(terrain, row, col, 0);
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

function createBasePlane() {
  const geometry = new THREE.PlaneGeometry(TERRAIN_WIDTH * 1.05, TERRAIN_DEPTH * 1.05);
  const material = new THREE.MeshStandardMaterial({
    color: 0x1f2d22,
    roughness: 0.95,
    metalness: 0,
    side: THREE.FrontSide,
  });
  const plane = new THREE.Mesh(geometry, material);
  plane.rotation.x = -Math.PI / 2;
  plane.position.y = -0.018;
  return plane;
}

function addPeakLabels(scene, data, frame) {
  const visiblePeakIds = new Set(
    data.views
      .flatMap((view) => view.annotations)
      .filter((annotation) => annotation.visible)
      .map((annotation) => annotation.peak_id),
  );
  const labeledPeaks = data.peaks
    .filter((peak) => visiblePeakIds.has(peak.id))
    .slice(0, MAX_PEAK_LABELS);

  for (const peak of labeledPeaks) {
    const point = localToScenePoint(peak.local_position, frame);
    const marker = createPeakMarker();
    marker.position.copy(point);
    scene.add(marker);

    const label = createLabel(peak.name, "peak-label");
    label.position.copy(marker.position);
    label.position.y += 0.078;
    label.userData.occlusionAnchor = marker;
    scene.add(label);
  }
}

// A flat camera-facing triangle glyph: the cartographic summit symbol. Reads as
// a clean map marker from any orbit angle without the bulk of a 3D model.
let peakGlyphTexture = null;

function createPeakMarker() {
  if (!peakGlyphTexture) {
    peakGlyphTexture = createPeakGlyphTexture();
  }
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({ map: peakGlyphTexture, transparent: true, depthWrite: false }),
  );
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

function cameraConfigsFor(views, frame) {
  return views.map((view) => {
    const positionDeltaM = distanceMeters(view.true_camera.position, view.pose_estimate.extrinsics.position);
    const compactEstimate = positionDeltaM <= PREDICTED_CAMERA_COMPACT_THRESHOLD_M;
    return {
      id: view.id,
      label: view.label,
      positionDeltaM,
      true: cameraConfig(
        view,
        CAMERA_ROLES.true,
        "True camera",
        view.label,
        view.true_camera,
        FOV_TRUE_COLOR,
        compactEstimate ? 0 : -0.055,
        compactEstimate ? 0 : -0.035,
        false,
        frame,
      ),
      estimated: cameraConfig(
        view,
        CAMERA_ROLES.estimated,
        compactEstimate ? "Predicted position" : "Estimated camera",
        compactEstimate ? "Fit" : `${view.label} fit`,
        view.pose_estimate.extrinsics,
        FOV_ESTIMATED_COLOR,
        compactEstimate ? 0 : 0.055,
        compactEstimate ? 0 : 0.035,
        compactEstimate,
        frame,
      ),
    };
  });
}

function cameraConfig(view, role, label, markerLabel, extrinsics, color, markerOffset, labelOffset, compact, frame) {
  const position = localToScenePoint(extrinsics.position, frame);
  position.y += 0.038;
  return {
    compact,
    label,
    markerLabel,
    extrinsics,
    color,
    markerOffset,
    labelOffset,
    position,
    role,
    target: cameraTargetPoint(extrinsics, frame),
    viewId: view.id,
    viewLabel: view.label,
  };
}

function addCameraMarkers(scene, viewConfigs) {
  const markers = [];
  for (const viewConfig of viewConfigs) {
    const trueMarker = createCameraMarker(viewConfig.true);
    const estimatedMarker = viewConfig.estimated.compact
      ? createPredictedPositionMarker(viewConfig.estimated)
      : createCameraMarker(viewConfig.estimated);
    markers.push(trueMarker, estimatedMarker);
    scene.add(trueMarker);
    scene.add(estimatedMarker);
  }
  return markers;
}

function createCameraMarker(config) {
  const forward = config.target.clone().sub(config.position).normalize();
  const group = new THREE.Group();
  group.position.copy(config.position);
  group.userData.cameraMode = config.role;
  group.userData.viewId = config.viewId;

  const bodyMaterial = new THREE.MeshStandardMaterial({
    color: config.color,
    emissive: config.color,
    emissiveIntensity: 0.35,
    roughness: 0.45,
  });
  const lensMaterial = new THREE.MeshStandardMaterial({
    color: 0x11130f,
    emissive: 0x0b1215,
    emissiveIntensity: 0.25,
    roughness: 0.25,
  });

  const marker = new THREE.Group();
  marker.position.x = config.markerOffset;
  marker.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), forward);

  const body = new THREE.Mesh(new THREE.BoxGeometry(0.088, 0.052, 0.04), bodyMaterial);
  marker.add(body);

  const finder = new THREE.Mesh(new THREE.BoxGeometry(0.036, 0.016, 0.02), bodyMaterial);
  finder.position.set(-0.022, 0.034, -0.004);
  marker.add(finder);

  const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.021, 0.032, 24), lensMaterial);
  lens.rotation.x = Math.PI / 2;
  lens.position.z = 0.036;
  marker.add(lens);

  marker.add(createCameraFrustumGlyph(config.color));

  const clickTarget = new THREE.Mesh(
    new THREE.BoxGeometry(0.18, 0.16, 0.18),
    new THREE.MeshBasicMaterial({
      transparent: true,
      opacity: 0,
      depthWrite: false,
    }),
  );
  clickTarget.userData.cameraPickTarget = true;
  clickTarget.userData.cameraMode = config.role;
  clickTarget.userData.viewId = config.viewId;
  marker.add(clickTarget);
  group.add(marker);

  const standMaterial = new THREE.MeshStandardMaterial({
    color: 0x0e100d,
    emissive: config.color,
    emissiveIntensity: 0.18,
    roughness: 0.55,
  });
  const stand = new THREE.Mesh(new THREE.CylinderGeometry(0.005, 0.006, 0.072, 8), standMaterial);
  stand.position.set(config.markerOffset, -0.038, 0);
  group.add(stand);

  const foot = new THREE.Mesh(new THREE.TorusGeometry(0.038, 0.003, 6, 24), standMaterial);
  foot.rotation.x = Math.PI / 2;
  foot.position.set(config.markerOffset, -0.077, 0);
  group.add(foot);

  const label = createLabel(config.markerLabel, `camera-label ${config.role}-camera`);
  label.position.set(config.markerOffset + config.labelOffset, 0.09, 0);
  label.userData.occlusionAnchor = group;
  group.add(label);
  return group;
}

function createPredictedPositionMarker(config) {
  const group = new THREE.Group();
  group.position.copy(config.position);
  group.userData.cameraMode = config.role;
  group.userData.viewId = config.viewId;

  const material = new THREE.MeshBasicMaterial({
    color: config.color,
    depthWrite: false,
    polygonOffset: true,
    polygonOffsetFactor: -2,
    polygonOffsetUnits: -2,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.94,
  });
  const ring = new THREE.Mesh(new THREE.RingGeometry(0.021, 0.034, 32), material);
  ring.rotation.x = -Math.PI / 2;
  group.add(ring);

  const crossGeometry = new THREE.BufferGeometry();
  crossGeometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute([
      -0.044, 0.002, 0,
      0.044, 0.002, 0,
      0, 0.002, -0.044,
      0, 0.002, 0.044,
    ], 3),
  );
  const cross = new THREE.LineSegments(
    crossGeometry,
    new THREE.LineBasicMaterial({
      color: 0x2d2105,
      transparent: true,
      opacity: 0.95,
      depthWrite: false,
    }),
  );
  group.add(cross);

  const clickTarget = new THREE.Mesh(
    new THREE.CylinderGeometry(0.07, 0.07, 0.08, 18),
    new THREE.MeshBasicMaterial({
      transparent: true,
      opacity: 0,
      depthWrite: false,
    }),
  );
  clickTarget.userData.cameraPickTarget = true;
  clickTarget.userData.cameraMode = config.role;
  clickTarget.userData.viewId = config.viewId;
  group.add(clickTarget);

  const label = createLabel(config.markerLabel, "camera-label estimated-camera");
  label.position.set(0.045, 0.052, 0);
  label.userData.occlusionAnchor = group;
  group.add(label);
  return group;
}

function collectCameraPickTargets(markers) {
  const targets = [];
  for (const marker of markers) {
    marker.traverse((object) => {
      if (object.userData?.cameraPickTarget) {
        targets.push(object);
      }
    });
  }
  return targets;
}

function createLabelOcclusion(scene, terrainMesh) {
  const labels = [];
  scene.traverse((object) => {
    if (object.element?.classList.contains("terrain-label")) {
      labels.push(object);
    }
  });
  return {
    cameraPosition: new THREE.Vector3(),
    direction: new THREE.Vector3(),
    labels,
    projectedPosition: new THREE.Vector3(),
    raycaster: new THREE.Raycaster(),
    targetPosition: new THREE.Vector3(),
    terrainMesh,
    terrainOccluder: createTerrainOccluder(terrainMesh),
  };
}

function createTerrainOccluder(terrainMesh) {
  const occluder = new THREE.Mesh(
    terrainMesh.geometry,
    new THREE.MeshBasicMaterial({ side: THREE.DoubleSide }),
  );
  occluder.matrixAutoUpdate = false;
  occluder.matrixWorld.copy(terrainMesh.matrixWorld);
  return occluder;
}

function syncTerrainOccluder(occlusion) {
  occlusion.terrainMesh.updateWorldMatrix(true, false);
  occlusion.terrainOccluder.matrixWorld.copy(occlusion.terrainMesh.matrixWorld);
}

function updateLabelOcclusion(occlusion, camera) {
  occlusion.cameraPosition.setFromMatrixPosition(camera.matrixWorld);
  syncTerrainOccluder(occlusion);

  for (const label of occlusion.labels) {
    const anchor = label.userData.occlusionAnchor ?? label;
    anchor.getWorldPosition(occlusion.targetPosition);
    label.visible =
      isPointInCameraView(occlusion.targetPosition, camera, occlusion.projectedPosition) &&
      !isOccludedByTerrain(occlusion, camera);
  }
}

function isOccludedByTerrain(occlusion, camera) {
  const labelDistance = occlusion.cameraPosition.distanceTo(occlusion.targetPosition);
  const rayLength = labelDistance - LABEL_OCCLUSION_MARGIN;
  if (rayLength <= camera.near) {
    return false;
  }

  occlusion.direction
    .copy(occlusion.targetPosition)
    .sub(occlusion.cameraPosition)
    .normalize();
  occlusion.raycaster.set(occlusion.cameraPosition, occlusion.direction);
  occlusion.raycaster.near = camera.near;
  occlusion.raycaster.far = rayLength;
  return occlusion.raycaster.intersectObject(occlusion.terrainOccluder, false).length > 0;
}

function isPointInCameraView(point, camera, projectedPosition) {
  projectedPosition.copy(point).project(camera);
  return (
    projectedPosition.z >= -1 &&
    projectedPosition.z <= 1 &&
    Math.abs(projectedPosition.x) <= LABEL_VIEW_PADDING &&
    Math.abs(projectedPosition.y) <= LABEL_VIEW_PADDING
  );
}

function updateMarkerSelection(viewer) {
  for (const marker of viewer.cameraMarkers) {
    const active = marker.userData.viewId === viewer.activeViewId;
    const role = marker.userData.cameraMode;
    const selected = markerSelected(viewer, active, role);
    marker.traverse((object) => {
      if (object.material?.emissiveIntensity !== undefined) {
        object.material.emissiveIntensity = selected ? 0.65 : active ? 0.34 : 0.14;
      }
      if (object.material?.opacity !== undefined && object.userData?.cameraPickTarget !== true) {
        object.material.opacity = active ? 1.0 : 0.48;
        object.material.transparent = object.material.transparent || !active;
      }
      if (object.element?.classList.contains("camera-label")) {
        object.element.style.opacity = active ? "1" : "0.42";
      }
    });
  }
}

function markerSelected(viewer, active, role) {
  if (!active) {
    return false;
  }
  if (viewer.selectedFootprint) {
    return role === viewer.selectedFootprint;
  }
  return viewer.mode === CAMERA_MODES.map || role === viewer.povRole;
}

function addFovOverlays(scene, data, frame, viewConfigs) {
  const overlays = {};
  for (const viewConfig of viewConfigs) {
    overlays[viewConfig.id] = {
      true: createFovOverlay(data, frame, viewConfig.true, FOV_TRUE_COLOR),
      estimated: createFovOverlay(data, frame, viewConfig.estimated, FOV_ESTIMATED_COLOR),
    };
    scene.add(overlays[viewConfig.id].true);
    scene.add(overlays[viewConfig.id].estimated);
  }
  return overlays;
}

// Builds the camera-coverage highlight: an additive "glow" fill over the terrain
// the camera actually sees, plus a crisp outline of the coverage boundary. The
// fill uses additive blending so it reads clearly against terrain and so an
// overlapping true/estimated pair shows agreement (bright) versus mismatch
// (single-colour fringes).
function createFovOverlay(data, frame, config, color) {
  const terrain = data.terrain;
  const projection = cameraProjection(config.extrinsics, data.scene.intrinsics);
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
  group.visible = false;
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
      opacity: 0.34,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      polygonOffset: true,
      polygonOffsetFactor: -2,
      polygonOffsetUnits: -2,
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
  // Shared interior edges are emitted by both adjacent triangles (count 2) and
  // dropped; only the true coverage boundary survives with count 1.
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

function cameraProjection(extrinsics, intrinsics) {
  return {
    axes: cameraAxes(extrinsics),
    extrinsics,
    frustumPlanes: cameraFrustumPlanes(intrinsics),
    intrinsics,
  };
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
  return {
    camera: localToCameraPoint(local, projection.extrinsics, projection.axes),
    local,
    scene,
  };
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
  const toCamera = localVector(centroid, cameraPosition);
  return dot(normal, toCamera) > 0;
}

function visibleTerrainTriangleIds(terrain, frame, projection) {
  const vertices = projectedTerrainGrid(terrain, frame, projection);
  const raster = visibilityRaster(projection.intrinsics);
  const depthBuffer = new Float32Array(raster.width * raster.height);
  depthBuffer.fill(Number.NEGATIVE_INFINITY);
  const ownerBuffer = new Int32Array(raster.width * raster.height);
  ownerBuffer.fill(-1);

  for (let row = 0; row < terrain.grid_height - 1; row += 1) {
    for (let col = 0; col < terrain.grid_width - 1; col += 1) {
      const topLeft = vertices[terrainVertexIndex(terrain, row, col)];
      const topRight = vertices[terrainVertexIndex(terrain, row, col + 1)];
      const bottomLeft = vertices[terrainVertexIndex(terrain, row + 1, col)];
      const bottomRight = vertices[terrainVertexIndex(terrain, row + 1, col + 1)];
      rasterizeVisibilityTriangle(
        raster,
        depthBuffer,
        ownerBuffer,
        terrainTriangleId(terrain, row, col, 0),
        topLeft,
        topRight,
        bottomLeft,
      );
      rasterizeVisibilityTriangle(
        raster,
        depthBuffer,
        ownerBuffer,
        terrainTriangleId(terrain, row, col, 1),
        topRight,
        bottomRight,
        bottomLeft,
      );
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
      const projected = projectCameraPoint(cameraPoint, projection.intrinsics);
      vertices[terrainVertexIndex(terrain, row, col)] = projected;
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

      const inverseDepth =
        weightA * a.inverseDepth + weightB * b.inverseDepth + weightC * c.inverseDepth;
      const bufferIndex = y * raster.width + x;
      if (inverseDepth > depthBuffer[bufferIndex]) {
        depthBuffer[bufferIndex] = inverseDepth;
        ownerBuffer[bufferIndex] = triangleId;
      }
    }
  }
}

function terrainVertexIndex(terrain, row, col) {
  return row * terrain.grid_width + col;
}

function terrainTriangleId(terrain, row, col, triangleOffset) {
  return ((row * (terrain.grid_width - 1) + col) * 2) + triangleOffset;
}

function addCameraConnectors(scene, viewConfigs) {
  const connectors = {};
  for (const viewConfig of viewConfigs) {
    const connector = createCameraConnector(viewConfig.true.position, viewConfig.estimated.position);
    connectors[viewConfig.id] = connector;
    scene.add(connector);
  }
  return connectors;
}

function createCameraConnector(truePosition, estimatedPosition) {
  const geometry = new THREE.BufferGeometry().setFromPoints([truePosition, estimatedPosition]);
  const material = new THREE.LineDashedMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.55,
    dashSize: 0.035,
    gapSize: 0.025,
  });
  const line = new THREE.Line(geometry, material);
  line.computeLineDistances();
  return line;
}

function setupMarkerPicking(viewer) {
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  let pointerDown = null;

  canvas.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || !viewer.controls.enabled) {
      pointerDown = null;
      return;
    }
    pointerDown = {
      x: event.clientX,
      y: event.clientY,
    };
  });

  canvas.addEventListener("pointerup", (event) => {
    if (!pointerDown || event.button !== 0 || !viewer.controls.enabled) {
      pointerDown = null;
      return;
    }

    const movement = Math.hypot(event.clientX - pointerDown.x, event.clientY - pointerDown.y);
    pointerDown = null;
    if (movement > CAMERA_PICK_MAX_DRAG_PX) {
      return;
    }

    const pick = pickCamera(event, viewer, raycaster, pointer);
    if (pick) {
      selectCameraFootprint(viewer, pick);
    } else {
      clearCameraFootprint(viewer);
    }
  });
}

function pickCamera(event, viewer, raycaster, pointer) {
  const rect = canvas.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, viewer.camera);

  const intersections = raycaster.intersectObjects(viewer.cameraPickTargets, false);
  return intersections.length ? findCameraPick(intersections[0].object) : null;
}

function findCameraPick(object) {
  let current = object;
  while (current) {
    if (current.userData?.cameraMode) {
      return {
        cameraMode: current.userData.cameraMode,
        viewId: current.userData.viewId,
      };
    }
    current = current.parent;
  }
  return null;
}

function selectCameraFootprint(viewer, pick) {
  activateViewerView(viewer, pick.viewId);
  const viewConfig = activeViewConfig(viewer);
  const role = pick.cameraMode;
  const config = viewConfig[role];
  if (!config) {
    return;
  }

  viewer.selectedFootprint = role;
  viewer.controls.enabled = true;
  setOverlayVisibility(viewer, [role]);
  setConnectorVisibility(viewer, false);
  setCameraReadout(`${viewConfig.label}: ${config.label} footprint`, "Free orbit; highlighted terrain is visible from this camera.");
  updateMarkerSelection(viewer);
}

function clearCameraFootprint(viewer) {
  if (!viewer.selectedFootprint) {
    return;
  }
  viewer.selectedFootprint = null;
  const viewConfig = activeViewConfig(viewer);
  setOverlayVisibility(viewer, [CAMERA_ROLES.true, CAMERA_ROLES.estimated]);
  setConnectorVisibility(viewer, true);
  setCameraReadout(`${viewConfig.label}: Map`, "Free orbit; blue is true camera coverage, amber is estimated.");
  updateMarkerSelection(viewer);
}

function activateViewerView(viewer, viewId) {
  if (viewer.activeViewId === viewId) {
    return;
  }
  const nextView = viewer.appState.views.find((view) => view.id === viewId);
  if (!nextView) {
    return;
  }
  viewer.activeViewId = viewId;
  viewer.appState.activeView = nextView;
  document.getElementById("viewSelect").value = viewId;
  refreshActiveViewPanels(viewer.appState);
}

function createScaleHud(terrain, frame) {
  return {
    bar: document.getElementById("scaleBar"),
    frame,
    label: document.getElementById("scaleLabel"),
    lastLabel: "",
    lastWidth: 0,
    terrain,
  };
}

function updateScaleHud(hud, camera) {
  const center = localToScenePoint(
    {
      east_m: 0,
      north_m: 0,
      up_m: (hud.frame.zMin + hud.frame.zMax) / 2,
    },
    hud.frame,
  );
  let selected = null;
  for (const distanceM of SCALE_CANDIDATES_M) {
    const widthPx = scaleCandidateWidthPx(distanceM, center, hud.frame, camera);
    if (!Number.isFinite(widthPx) || widthPx <= 0) {
      continue;
    }
    if (widthPx <= SCALE_MAX_WIDTH_PX) {
      selected = { distanceM, widthPx };
    }
  }

  if (!selected) {
    const distanceM = SCALE_CANDIDATES_M[0];
    selected = {
      distanceM,
      widthPx: scaleCandidateWidthPx(distanceM, center, hud.frame, camera),
    };
  }
  if (!Number.isFinite(selected.widthPx)) {
    hud.bar.style.width = "96px";
    hud.label.textContent = "-";
    return;
  }

  const widthPx = Math.max(SCALE_MIN_WIDTH_PX, Math.min(SCALE_MAX_WIDTH_PX, selected.widthPx));
  const label = formatDistance(selected.distanceM);
  if (Math.abs(widthPx - hud.lastWidth) > 1 || label !== hud.lastLabel) {
    hud.bar.style.width = `${Math.round(widthPx)}px`;
    hud.label.textContent = label;
    hud.lastWidth = widthPx;
    hud.lastLabel = label;
  }
}

function scaleCandidateWidthPx(distanceM, center, frame, camera) {
  const sceneWidth = (distanceM / (frame.xMax - frame.xMin)) * TERRAIN_WIDTH;
  const left = center.clone();
  const right = center.clone();
  left.x -= sceneWidth / 2;
  right.x += sceneWidth / 2;
  const leftScreen = projectSceneToCanvas(left, camera);
  const rightScreen = projectSceneToCanvas(right, camera);
  return Math.abs(rightScreen.x - leftScreen.x);
}

function projectSceneToCanvas(point, camera) {
  const projected = point.clone().project(camera);
  return {
    x: ((projected.x + 1) / 2) * canvas.clientWidth,
    y: ((1 - projected.y) / 2) * canvas.clientHeight,
  };
}

function terrainFrame(terrain) {
  return {
    xMin: terrain.x_min_m,
    xMax: terrain.x_max_m,
    yMin: terrain.y_min_m,
    yMax: terrain.y_max_m,
    zMin: terrain.elevation_min_m,
    zMax: terrain.elevation_max_m,
  };
}

function terrainLocalGridPoint(terrain, frame, row, col) {
  return {
    east_m: interpolate(frame.xMin, frame.xMax, col / (terrain.grid_width - 1)),
    north_m: interpolate(frame.yMin, frame.yMax, row / (terrain.grid_height - 1)),
    up_m: terrain.elevation_m[row][col],
  };
}

function terrainGridPoint(terrain, row, col, verticalOffset) {
  const frame = terrainFrame(terrain);
  const localPoint = terrainLocalGridPoint(terrain, frame, row, col);
  const point = localToScenePoint(localPoint, frame);
  point.y += verticalOffset;
  return point;
}

function terrainFloatPoint(terrain, row, col, elevation, verticalOffset) {
  const frame = terrainFrame(terrain);
  const localPoint = {
    east_m: interpolate(frame.xMin, frame.xMax, col / (terrain.grid_width - 1)),
    north_m: interpolate(frame.yMin, frame.yMax, row / (terrain.grid_height - 1)),
    up_m: elevation,
  };
  const point = localToScenePoint(localPoint, frame);
  point.y += verticalOffset;
  return point;
}

function localToScenePoint(localPoint, frame) {
  const east = normalize(localPoint.east_m, frame.xMin, frame.xMax) - 0.5;
  const north = normalize(localPoint.north_m, frame.yMin, frame.yMax) - 0.5;
  const elevation = normalize(localPoint.up_m, frame.zMin, frame.zMax);
  return new THREE.Vector3(
    east * TERRAIN_WIDTH,
    elevation * TERRAIN_HEIGHT,
    -north * TERRAIN_DEPTH,
  );
}

function cameraTargetPoint(extrinsics, frame) {
  const yaw = THREE.MathUtils.degToRad(extrinsics.yaw_deg);
  const pitch = THREE.MathUtils.degToRad(extrinsics.pitch_deg);
  const origin = extrinsics.position;
  const target = {
    east_m: origin.east_m + Math.sin(yaw) * Math.cos(pitch) * CAMERA_LOOK_DISTANCE_M,
    north_m: origin.north_m + Math.cos(yaw) * Math.cos(pitch) * CAMERA_LOOK_DISTANCE_M,
    up_m: origin.up_m + Math.sin(pitch) * CAMERA_LOOK_DISTANCE_M,
  };
  return localToScenePoint(target, frame);
}

function localToCameraPoint(point, extrinsics, axes) {
  const vector = [
    point.east_m - extrinsics.position.east_m,
    point.north_m - extrinsics.position.north_m,
    point.up_m - extrinsics.position.up_m,
  ];
  return {
    depth: dot(vector, axes.forward),
    x: dot(vector, axes.right),
    y: dot(vector, axes.down),
  };
}

function projectCameraPoint(point, intrinsics) {
  const depth = point.depth;
  if (depth <= 1) {
    return {
      depth,
      inverseDepth: Number.NEGATIVE_INFINITY,
      u: Number.NaN,
      v: Number.NaN,
      valid: false,
    };
  }
  return {
    depth,
    inverseDepth: 1 / depth,
    valid: true,
    u: intrinsics.focal_length_px * (point.x / depth) + intrinsics.principal_x_px,
    v: intrinsics.focal_length_px * (point.y / depth) + intrinsics.principal_y_px,
  };
}

function cameraAxes(extrinsics) {
  const yaw = THREE.MathUtils.degToRad(extrinsics.yaw_deg);
  const pitch = THREE.MathUtils.degToRad(extrinsics.pitch_deg);
  const forward = [
    Math.sin(yaw) * Math.cos(pitch),
    Math.cos(yaw) * Math.cos(pitch),
    Math.sin(pitch),
  ];
  const right = [Math.cos(yaw), -Math.sin(yaw), 0];
  const down = cross(forward, right);
  return {
    right: unit(right),
    down: unit(down),
    forward: unit(forward),
  };
}

function createLabel(text, className) {
  const element = document.createElement("div");
  element.className = `terrain-label ${className}`;
  element.textContent = text;
  return new CSS2DObject(element);
}

function createCameraFrustumGlyph(color) {
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
    new THREE.LineBasicMaterial({
      color,
      transparent: true,
      opacity: 0.95,
      depthWrite: false,
    }),
  );
}

function resizeRenderers(renderer, labelRenderer, camera) {
  const width = Math.max(1, canvas.clientWidth);
  const height = Math.max(1, canvas.clientHeight);
  renderer.setSize(width, height, false);
  labelRenderer.setSize(width, height);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

function formatNumber(value, unit) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return `${Number(value).toFixed(1)} ${unit}`;
}

function formatDistance(valueM) {
  if (valueM >= 1000) {
    return `${Number(valueM / 1000).toFixed(valueM % 1000 === 0 ? 0 : 1)} km`;
  }
  return `${Math.round(valueM)} m`;
}

function distanceMeters(a, b) {
  return Math.hypot(a.east_m - b.east_m, a.north_m - b.north_m, a.up_m - b.up_m);
}

function angleDeltaDeg(a, b) {
  return Math.abs(((a - b + 180) % 360) - 180);
}

function normalize(value, min, max) {
  return (value - min) / Math.max(max - min, 1);
}

function interpolate(min, max, t) {
  return min + (max - min) * t;
}

function pushTriangle(positions, a, b, c) {
  positions.push(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z);
}

function localVector(from, to) {
  return [
    to.east_m - from.east_m,
    to.north_m - from.north_m,
    to.up_m - from.up_m,
  ];
}

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function unit(vector) {
  const norm = Math.hypot(vector[0], vector[1], vector[2]);
  return [vector[0] / norm, vector[1] / norm, vector[2] / norm];
}

// --- Alignment Lab -----------------------------------------------------------
// Runtime camera localization: the position is known (click to place anywhere),
// the orientation is not. We render the ground-truth skyline outline, corrupt it
// with noise, then let a chosen solver recover the heading from that outline.
// The solvers map to families from docs/development/camera-fitting-research.md:
// bounded coarse-to-fine grid, Nelder-Mead local refinement, and differential
// evolution global search.

const ALIGN_SAMPLE_WIDTH = 176;
const ALIGN_TERRAIN_STRIDE = 3;
const ALIGN_EYE_HEIGHT_M = 3;
const ALIGN_YAW_BOUNDS = [-180, 180];
const ALIGN_PITCH_BOUNDS = [-12, 26];

const ALIGN_SOLVERS = {
  grid: {
    label: "Coarse-to-fine grid",
    blurb: "Bounded global scan over heading and pitch, then a local refine around the best cell.",
    create: createGridSolver,
    usesPrior: false,
  },
  nelder: {
    label: "Nelder-Mead (local)",
    blurb: "Derivative-free simplex refining from a compass prior toward the highest terrain.",
    create: createNelderMeadSolver,
    usesPrior: true,
  },
  evolution: {
    label: "Differential evolution",
    blurb: "Population-based global search; robust to the many local minima of skyline matching.",
    create: createDifferentialEvolutionSolver,
    usesPrior: false,
  },
};

function setupAlignmentLab(data) {
  const terrain = data.terrain;
  const frame = terrainFrame(terrain);
  const lab = {
    terrain,
    frame,
    intrinsics: data.scene.intrinsics,
    sampleWidth: ALIGN_SAMPLE_WIDTH,
    sampleHeight: Math.max(1, Math.round((ALIGN_SAMPLE_WIDTH * data.scene.intrinsics.height_px) / data.scene.intrinsics.width_px)),
    grid: buildAlignGrid(terrain, frame, ALIGN_TERRAIN_STRIDE),
    lookTarget: highestLocalPoint(terrain, frame),
    algo: "grid",
    noisePx: 6,
    position: null,
    trueYaw: 0,
    truePitch: 4,
    priorYaw: 0,
    target: null,
    best: null,
    solver: null,
    running: false,
    raf: 0,
    evals: 0,
    dom: {
      select: document.getElementById("algoSelect"),
      blurb: document.getElementById("algoBlurb"),
      noise: document.getElementById("noiseSlider"),
      noiseValue: document.getElementById("noiseValue"),
      run: document.getElementById("algoRun"),
      step: document.getElementById("algoStep"),
      reset: document.getElementById("algoReset"),
      status: document.getElementById("algoStatus"),
      yawErr: document.getElementById("algoYawErr"),
      pitchErr: document.getElementById("algoPitchErr"),
      score: document.getElementById("algoScore"),
      evalCount: document.getElementById("algoEvals"),
      mapCanvas: document.getElementById("alignMap"),
      plot: document.getElementById("alignPlot"),
    },
  };
  lab.mapContext = lab.dom.mapCanvas.getContext("2d");
  lab.topScratch = new Float32Array(lab.sampleWidth);
  lab.projection = lab.grid.points.map((row) => row.map(() => ({ front: false, px: 0, py: 0 })));

  lab.dom.select.replaceChildren(
    ...Object.entries(ALIGN_SOLVERS).map(([key, meta]) => new Option(meta.label, key)),
  );
  lab.dom.select.value = lab.algo;
  lab.dom.select.addEventListener("change", () => {
    lab.algo = lab.dom.select.value;
    resetAlignSolver(lab);
  });
  lab.dom.noise.addEventListener("input", () => {
    lab.noisePx = Number(lab.dom.noise.value);
    lab.dom.noiseValue.textContent = `${lab.noisePx} px`;
    if (lab.position) {
      regenerateAlignTarget(lab);
      resetAlignSolver(lab);
    }
  });
  lab.dom.run.addEventListener("click", () => (lab.running ? stopAlignSolver(lab) : runAlignSolver(lab)));
  lab.dom.step.addEventListener("click", () => stepAlignSolver(lab));
  lab.dom.reset.addEventListener("click", () => resetAlignSolver(lab));
  lab.dom.mapCanvas.addEventListener("click", (event) => {
    const rect = lab.dom.mapCanvas.getBoundingClientRect();
    placeAlignCamera(lab, alignCanvasToLocal(lab, event.clientX - rect.left, event.clientY - rect.top));
  });

  const resizeObserver = new ResizeObserver(() => {
    renderAlignMinimap(lab);
    drawAlignMinimap(lab);
    drawAlignPlot(lab);
  });
  resizeObserver.observe(lab.dom.mapCanvas);
  resizeObserver.observe(lab.dom.plot);

  syncSolverBlurb(lab);
  renderAlignMinimap(lab);
  placeAlignCamera(lab, lowestLocalPoint(terrain, frame));
}

function buildAlignGrid(terrain, frame, stride) {
  const rowIndices = strideIndices(terrain.grid_height, stride);
  const colIndices = strideIndices(terrain.grid_width, stride);
  const points = rowIndices.map((row) => colIndices.map((col) => terrainLocalGridPoint(terrain, frame, row, col)));
  return { points };
}

function strideIndices(count, stride) {
  const indices = [];
  for (let value = 0; value < count - 1; value += stride) {
    indices.push(value);
  }
  indices.push(count - 1);
  return indices;
}

function alignSkyline(lab, yawDeg, pitchDeg) {
  const { intrinsics, sampleWidth: width, sampleHeight: height, grid, projection } = lab;
  const axes = cameraAxes({ yaw_deg: yawDeg, pitch_deg: pitchDeg });
  const extrinsics = { position: lab.position };
  const scaleX = width / intrinsics.width_px;
  const scaleY = height / intrinsics.height_px;
  const rows = grid.points.length;
  const cols = grid.points[0].length;
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const projected = projectCameraPoint(localToCameraPoint(grid.points[row][col], extrinsics, axes), intrinsics);
      const slot = projection[row][col];
      slot.front = projected.valid;
      slot.px = projected.u * scaleX;
      slot.py = projected.v * scaleY;
    }
  }

  lab.topScratch.fill(Number.POSITIVE_INFINITY);
  for (let row = 0; row < rows - 1; row += 1) {
    for (let col = 0; col < cols - 1; col += 1) {
      rasterizeTopEdge(lab.topScratch, width, height, projection[row][col], projection[row][col + 1], projection[row + 1][col]);
      rasterizeTopEdge(lab.topScratch, width, height, projection[row][col + 1], projection[row + 1][col + 1], projection[row + 1][col]);
    }
  }
  return lab.topScratch;
}

// Mean absolute skyline residual in sample rows, with missing-coverage columns
// charged the full image height so the objective pulls overlap into agreement.
function scoreAlignProfiles(profile, target, height) {
  let sum = 0;
  let counted = 0;
  let mismatch = 0;
  for (let column = 0; column < target.length; column += 1) {
    const a = profile[column];
    const b = target[column];
    const aFinite = Number.isFinite(a);
    const bFinite = Number.isFinite(b);
    if (aFinite && bFinite) {
      sum += Math.abs(a - b);
      counted += 1;
    } else if (aFinite !== bFinite) {
      mismatch += 1;
    }
  }
  const total = counted + mismatch;
  return total === 0 ? height : (sum + mismatch * height) / total;
}

function regenerateAlignTarget(lab) {
  const clean = alignSkyline(lab, lab.trueYaw, lab.truePitch);
  const noiseRows = (lab.noisePx * lab.sampleHeight) / lab.intrinsics.height_px;
  const target = new Float32Array(lab.sampleWidth);
  for (let column = 0; column < target.length; column += 1) {
    target[column] = Number.isFinite(clean[column]) ? clean[column] + gaussianNoise() * noiseRows : Number.POSITIVE_INFINITY;
  }
  lab.target = target;
}

function placeAlignCamera(lab, local) {
  stopAlignSolver(lab);
  const up = sampleAlignElevation(lab, local.east, local.north) + ALIGN_EYE_HEIGHT_M;
  lab.position = { east_m: local.east, north_m: local.north, up_m: up };
  lab.priorYaw = headingDeg(local.east, local.north, lab.lookTarget.east, lab.lookTarget.north);
  lab.trueYaw = wrapDeg(lab.priorYaw + (Math.random() * 80 - 40));
  lab.truePitch = 1 + Math.random() * 7;
  regenerateAlignTarget(lab);
  resetAlignSolver(lab);
}

function resetAlignSolver(lab) {
  stopAlignSolver(lab);
  if (!lab.position) {
    return;
  }
  lab.evals = 0;
  const evaluate = (yaw, pitch) => {
    lab.evals += 1;
    return scoreAlignProfiles(alignSkyline(lab, yaw, pitch), lab.target, lab.sampleHeight);
  };
  const bounds = { yaw: ALIGN_YAW_BOUNDS, pitch: ALIGN_PITCH_BOUNDS };
  const seed = { yaw: lab.priorYaw, pitch: 4 };
  lab.solver = ALIGN_SOLVERS[lab.algo].create(evaluate, bounds, seed);
  lab.best = ALIGN_SOLVERS[lab.algo].usesPrior ? { yaw: seed.yaw, pitch: seed.pitch, score: NaN } : null;
  syncSolverBlurb(lab);
  lab.dom.status.textContent = "Ready. Run or step the solver.";
  updateAlignReadout(lab, lab.best ? { ...lab.best, done: false } : null);
  drawAlignMinimap(lab);
  drawAlignPlot(lab);
}

function runAlignSolver(lab) {
  if (!lab.solver || lab.running) {
    return;
  }
  lab.running = true;
  lab.dom.run.textContent = "Pause";
  const tick = () => {
    if (!lab.running) {
      return;
    }
    const result = lab.solver.step();
    lab.best = result;
    updateAlignReadout(lab, result);
    if (result.done) {
      stopAlignSolver(lab);
      lab.dom.status.textContent = `Converged after ${lab.evals} evaluations.`;
    } else {
      lab.raf = requestAnimationFrame(tick);
    }
  };
  lab.raf = requestAnimationFrame(tick);
}

function stepAlignSolver(lab) {
  if (!lab.solver || lab.running) {
    return;
  }
  const result = lab.solver.step();
  lab.best = result;
  updateAlignReadout(lab, result);
  if (result.done) {
    lab.dom.status.textContent = `Converged after ${lab.evals} evaluations.`;
  }
}

function stopAlignSolver(lab) {
  lab.running = false;
  cancelAnimationFrame(lab.raf);
  lab.dom.run.textContent = "Run";
}

function syncSolverBlurb(lab) {
  lab.dom.blurb.textContent = ALIGN_SOLVERS[lab.algo].blurb;
}

function updateAlignReadout(lab, result) {
  if (!result) {
    lab.dom.yawErr.textContent = "-";
    lab.dom.pitchErr.textContent = "-";
    lab.dom.score.textContent = "-";
    lab.dom.evalCount.textContent = "0";
    return;
  }
  lab.dom.yawErr.textContent = formatNumber(angleDeltaDeg(result.yaw, lab.trueYaw), "deg");
  lab.dom.pitchErr.textContent = formatNumber(Math.abs(result.pitch - lab.truePitch), "deg");
  const scorePx = Number.isFinite(result.score)
    ? (result.score * lab.intrinsics.height_px) / lab.sampleHeight
    : NaN;
  lab.dom.score.textContent = formatNumber(scorePx, "px");
  lab.dom.evalCount.textContent = String(lab.evals);
  drawAlignMinimap(lab);
  drawAlignPlot(lab);
}

// --- Solvers ---

function createGridSolver(evaluate, bounds) {
  let queue = buildGridQueue(bounds.yaw, 6, bounds.pitch, 4);
  let index = 0;
  let best = null;
  let refined = false;
  const evaluateChunk = (count) => {
    for (let step = 0; step < count && index < queue.length; step += 1, index += 1) {
      const [yaw, pitch] = queue[index];
      const score = evaluate(yaw, pitch);
      if (!best || score < best.score) {
        best = { yaw, pitch, score };
      }
    }
  };
  return {
    step() {
      evaluateChunk(28);
      if (index >= queue.length) {
        if (!refined && best) {
          refined = true;
          index = 0;
          queue = buildGridQueue(
            [best.yaw - 6, best.yaw + 6],
            1,
            [Math.max(bounds.pitch[0], best.pitch - 4), Math.min(bounds.pitch[1], best.pitch + 4)],
            0.5,
          );
        } else {
          return { ...best, done: true };
        }
      }
      return { ...best, done: false };
    },
  };
}

function buildGridQueue([yawStart, yawEnd], yawStep, [pitchStart, pitchEnd], pitchStep) {
  const queue = [];
  for (let yaw = yawStart; yaw <= yawEnd; yaw += yawStep) {
    for (let pitch = pitchStart; pitch <= pitchEnd; pitch += pitchStep) {
      queue.push([wrapDeg(yaw), pitch]);
    }
  }
  return queue;
}

function createNelderMeadSolver(evaluate, bounds, seed) {
  const vertex = (yaw, pitch) => ({ yaw, pitch, score: evaluate(wrapDeg(yaw), clamp(pitch, bounds.pitch)) });
  let simplex = [vertex(seed.yaw, seed.pitch), vertex(seed.yaw + 18, seed.pitch), vertex(seed.yaw, seed.pitch + 6)];
  let iterations = 0;
  return {
    step() {
      simplex.sort((a, b) => a.score - b.score);
      const [best, , worst] = simplex;
      const centroidYaw = (simplex[0].yaw + simplex[1].yaw) / 2;
      const centroidPitch = (simplex[0].pitch + simplex[1].pitch) / 2;
      const reflected = vertex(centroidYaw + (centroidYaw - worst.yaw), centroidPitch + (centroidPitch - worst.pitch));
      if (reflected.score < best.score) {
        const expanded = vertex(centroidYaw + 2 * (centroidYaw - worst.yaw), centroidPitch + 2 * (centroidPitch - worst.pitch));
        simplex[2] = expanded.score < reflected.score ? expanded : reflected;
      } else if (reflected.score < simplex[1].score) {
        simplex[2] = reflected;
      } else {
        const contracted = vertex(centroidYaw + 0.5 * (worst.yaw - centroidYaw), centroidPitch + 0.5 * (worst.pitch - centroidPitch));
        if (contracted.score < worst.score) {
          simplex[2] = contracted;
        } else {
          simplex = [best, vertex((best.yaw + simplex[1].yaw) / 2, (best.pitch + simplex[1].pitch) / 2), vertex((best.yaw + worst.yaw) / 2, (best.pitch + worst.pitch) / 2)];
        }
      }
      iterations += 1;
      simplex.sort((a, b) => a.score - b.score);
      const spread = Math.abs(simplex[0].yaw - simplex[2].yaw) + Math.abs(simplex[0].pitch - simplex[2].pitch);
      const top = simplex[0];
      return { yaw: wrapDeg(top.yaw), pitch: clamp(top.pitch, bounds.pitch), score: top.score, done: spread < 0.05 || iterations > 80 };
    },
  };
}

function createDifferentialEvolutionSolver(evaluate, bounds) {
  const size = 24;
  const population = [];
  for (let member = 0; member < size; member += 1) {
    const yaw = randomIn(bounds.yaw);
    const pitch = randomIn(bounds.pitch);
    population.push({ yaw, pitch, score: evaluate(yaw, pitch) });
  }
  let generation = 0;
  let stagnation = 0;
  let bestScore = Math.min(...population.map((member) => member.score));
  return {
    step() {
      for (let target = 0; target < size; target += 1) {
        const [a, b, c] = pickThree(population, target);
        let yaw = wrapDeg(a.yaw + 0.7 * angleDiff(b.yaw, c.yaw));
        let pitch = clamp(a.pitch + 0.7 * (b.pitch - c.pitch), bounds.pitch);
        if (Math.random() > 0.9) {
          yaw = population[target].yaw;
        }
        if (Math.random() > 0.9) {
          pitch = population[target].pitch;
        }
        const score = evaluate(yaw, pitch);
        if (score < population[target].score) {
          population[target] = { yaw, pitch, score };
        }
      }
      generation += 1;
      const best = population.reduce((lowest, member) => (member.score < lowest.score ? member : lowest));
      stagnation = best.score < bestScore - 1e-4 ? 0 : stagnation + 1;
      bestScore = Math.min(bestScore, best.score);
      return { yaw: best.yaw, pitch: best.pitch, score: best.score, done: generation > 60 || stagnation > 12 };
    },
  };
}

// --- Lab drawing ---

function renderAlignMinimap(lab) {
  const canvas = lab.dom.mapCanvas;
  const width = Math.max(1, canvas.clientWidth);
  const height = Math.max(1, canvas.clientHeight);
  canvas.width = width;
  canvas.height = height;
  const image = lab.mapContext.createImageData(width, height);
  const { frame } = lab;
  const elevationRange = Math.max(1, frame.zMax - frame.zMin);
  const metersPerPixel = (frame.xMax - frame.xMin) / width;
  for (let y = 0; y < height; y += 1) {
    const north = interpolate(frame.yMax, frame.yMin, y / height);
    for (let x = 0; x < width; x += 1) {
      const east = interpolate(frame.xMin, frame.xMax, x / width);
      const elevation = sampleAlignElevation(lab, east, north);
      const rgb = alignElevationRgb((elevation - frame.zMin) / elevationRange);
      const shade = alignHillshade(lab, east, north, metersPerPixel);
      const offset = (y * width + x) * 4;
      image.data[offset] = Math.min(255, rgb[0] * shade);
      image.data[offset + 1] = Math.min(255, rgb[1] * shade);
      image.data[offset + 2] = Math.min(255, rgb[2] * shade);
      image.data[offset + 3] = 255;
    }
  }
  lab.mapBaseImage = image;
}

// Cheap relief shading from the terrain gradient lit from the north-west.
function alignHillshade(lab, east, north, step) {
  const dEast = sampleAlignElevation(lab, east + step, north) - sampleAlignElevation(lab, east - step, north);
  const dNorth = sampleAlignElevation(lab, east, north + step) - sampleAlignElevation(lab, east, north - step);
  const scale = 1.4 / (2 * step);
  const light = (-dEast * scale + dNorth * scale) * 0.5 + 0.78;
  return Math.max(0.45, Math.min(1.25, light));
}

function drawAlignMinimap(lab) {
  if (!lab.mapBaseImage) {
    return;
  }
  const context = lab.mapContext;
  context.putImageData(lab.mapBaseImage, 0, 0);
  if (!lab.position) {
    return;
  }
  const origin = alignLocalToCanvas(lab, lab.position.east_m, lab.position.north_m);
  drawAlignHeading(context, origin, lab.trueYaw, "rgba(108, 235, 150, 0.95)");
  if (lab.best) {
    drawAlignHeading(context, origin, lab.best.yaw, "rgba(255, 183, 77, 0.95)");
  }
  context.beginPath();
  context.arc(origin.x, origin.y, 5, 0, Math.PI * 2);
  context.fillStyle = "#f2eddf";
  context.strokeStyle = "#1a1c15";
  context.lineWidth = 2;
  context.fill();
  context.stroke();
}

function drawAlignHeading(context, origin, yawDeg, color) {
  const yaw = THREE.MathUtils.degToRad(yawDeg);
  const length = 46;
  context.beginPath();
  context.moveTo(origin.x, origin.y);
  context.lineTo(origin.x + Math.sin(yaw) * length, origin.y - Math.cos(yaw) * length);
  context.strokeStyle = color;
  context.lineWidth = 2.5;
  context.stroke();
}

function drawAlignPlot(lab) {
  const plot = lab.dom.plot;
  const width = Math.max(1, plot.clientWidth);
  const height = Math.max(1, plot.clientHeight);
  plot.setAttribute("viewBox", `0 0 ${width} ${height}`);
  const elements = [];
  if (lab.target) {
    elements.push(alignProfilePath(lab, lab.target, width, height, "target-line"));
  }
  if (lab.best && lab.position && Number.isFinite(lab.best.yaw)) {
    elements.push(alignProfilePath(lab, alignSkyline(lab, lab.best.yaw, lab.best.pitch), width, height, "estimate-line"));
  }
  plot.replaceChildren(...elements.filter(Boolean));
}

function alignProfilePath(lab, profile, width, height, className) {
  const points = [];
  for (let column = 0; column < profile.length; column += 1) {
    if (Number.isFinite(profile[column])) {
      const x = (column / (profile.length - 1)) * width;
      const y = (profile[column] / lab.sampleHeight) * height;
      points.push(`${x.toFixed(1)},${y.toFixed(1)}`);
    }
  }
  if (points.length < 2) {
    return null;
  }
  return svgElement("polyline", { class: className, points: points.join(" ") });
}

// --- Lab helpers ---

function sampleAlignElevation(lab, east, north) {
  const { terrain, frame } = lab;
  const colFloat = ((east - frame.xMin) / Math.max(1, frame.xMax - frame.xMin)) * (terrain.grid_width - 1);
  const rowFloat = ((north - frame.yMin) / Math.max(1, frame.yMax - frame.yMin)) * (terrain.grid_height - 1);
  const col = Math.max(0, Math.min(terrain.grid_width - 2, Math.floor(colFloat)));
  const row = Math.max(0, Math.min(terrain.grid_height - 2, Math.floor(rowFloat)));
  const tx = colFloat - col;
  const ty = rowFloat - row;
  const top = interpolate(terrain.elevation_m[row][col], terrain.elevation_m[row][col + 1], tx);
  const bottom = interpolate(terrain.elevation_m[row + 1][col], terrain.elevation_m[row + 1][col + 1], tx);
  return interpolate(top, bottom, ty);
}

function highestLocalPoint(terrain, frame) {
  return extremeLocalPoint(terrain, frame, (value, best) => value > best);
}

function lowestLocalPoint(terrain, frame) {
  return extremeLocalPoint(terrain, frame, (value, best) => value < best);
}

function extremeLocalPoint(terrain, frame, isBetter) {
  let bestValue = terrain.elevation_m[0][0];
  let bestRow = 0;
  let bestCol = 0;
  for (let row = 0; row < terrain.grid_height; row += 1) {
    for (let col = 0; col < terrain.grid_width; col += 1) {
      if (isBetter(terrain.elevation_m[row][col], bestValue)) {
        bestValue = terrain.elevation_m[row][col];
        bestRow = row;
        bestCol = col;
      }
    }
  }
  const point = terrainLocalGridPoint(terrain, frame, bestRow, bestCol);
  return { east: point.east_m, north: point.north_m };
}

function alignCanvasToLocal(lab, canvasX, canvasY) {
  const width = Math.max(1, lab.dom.mapCanvas.clientWidth);
  const height = Math.max(1, lab.dom.mapCanvas.clientHeight);
  return {
    east: interpolate(lab.frame.xMin, lab.frame.xMax, canvasX / width),
    north: interpolate(lab.frame.yMax, lab.frame.yMin, canvasY / height),
  };
}

function alignLocalToCanvas(lab, east, north) {
  const width = Math.max(1, lab.dom.mapCanvas.clientWidth);
  const height = Math.max(1, lab.dom.mapCanvas.clientHeight);
  return {
    x: ((east - lab.frame.xMin) / Math.max(1, lab.frame.xMax - lab.frame.xMin)) * width,
    y: ((lab.frame.yMax - north) / Math.max(1, lab.frame.yMax - lab.frame.yMin)) * height,
  };
}

function alignElevationRgb(t) {
  const clamped = Math.max(0, Math.min(1, t));
  const stops = [
    [38, 58, 44],
    [62, 104, 67],
    [146, 149, 93],
    [218, 209, 179],
  ];
  const scaled = clamped * (stops.length - 1);
  const index = Math.min(stops.length - 2, Math.floor(scaled));
  const fraction = scaled - index;
  return stops[index].map((value, channel) => Math.round(interpolate(value, stops[index + 1][channel], fraction)));
}

function headingDeg(fromEast, fromNorth, toEast, toNorth) {
  return (Math.atan2(toEast - fromEast, toNorth - fromNorth) * 180) / Math.PI;
}

function wrapDeg(value) {
  return ((((value + 180) % 360) + 360) % 360) - 180;
}

function angleDiff(a, b) {
  return wrapDeg(a - b);
}

function clamp(value, [low, high]) {
  return Math.max(low, Math.min(high, value));
}

function randomIn([low, high]) {
  return low + Math.random() * (high - low);
}

function pickThree(population, exclude) {
  const indices = [];
  while (indices.length < 3) {
    const candidate = Math.floor(Math.random() * population.length);
    if (candidate !== exclude && !indices.includes(candidate)) {
      indices.push(candidate);
    }
  }
  return indices.map((index) => population[index]);
}

function gaussianNoise() {
  let u = 0;
  let v = 0;
  while (u === 0) {
    u = Math.random();
  }
  while (v === 0) {
    v = Math.random();
  }
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

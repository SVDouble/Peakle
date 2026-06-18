"use strict";

import * as THREE from "three";
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
const ISOLINE_CORE_WIDTH = 0.0028;
const ISOLINE_HALO_WIDTH = 0.008;
const ISOLINE_INTERVAL_M = 250;
const ISOLINE_VERTICAL_OFFSET = 0.006;
const RASTER_EPSILON = 1e-8;
const SCALE_CANDIDATES_M = [100, 200, 500, 1000, 2000, 5000, 10000];
const SCALE_MIN_WIDTH_PX = 64;
const SCALE_MAX_WIDTH_PX = 150;
const VISIBILITY_RASTER_MAX_WIDTH = 520;

const CAMERA_MODES = {
  orbit: "orbit",
  true: "true",
  estimated: "estimated",
  compare: "compare",
};

const canvas = document.getElementById("terrainCanvas");
const statusBox = document.getElementById("webglStatus");
const sceneImage = document.getElementById("sceneImage");
const showRender = document.getElementById("showRender");
const showAnnotated = document.getElementById("showAnnotated");
const showContours = document.getElementById("showContours");
const imageContourLegend = document.getElementById("imageContourLegend");

main().catch((error) => {
  statusBox.hidden = false;
  statusBox.textContent = `Viewer failed to load: ${error.message}`;
});

async function main() {
  const response = await fetch("viewer-data.json");
  const data = await response.json();
  const appState = {
    activeView: null,
    data,
    imageMode: "render",
    views: data.views,
  };
  validateViews(appState.views);
  appState.activeView = appState.views[0];
  populatePanels(appState);
  setupImageToggle(appState);
  setupComparisonPanel(appState.activeView);
  setupTerrainViewer(data, appState);
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

function setupImageToggle(appState) {
  const modes = {
    annotated: {
      alt: "Synthetic mountain view with peak labels",
      button: showAnnotated,
      showLegend: false,
    },
    contours: {
      alt: "True and estimated 2D skyline contour comparison",
      button: showContours,
      showLegend: true,
    },
    render: {
      alt: "Synthetic mountain view",
      button: showRender,
      showLegend: false,
    },
  };

  for (const [mode, config] of Object.entries(modes)) {
    config.button.addEventListener("click", () => selectImageMode(appState, modes, mode));
  }
  selectImageMode(appState, modes, appState.imageMode);
}

function selectImageMode(appState, modes, mode) {
  const selected = modes[mode];
  appState.imageMode = mode;
  sceneImage.src = appState.activeView.images[imageModeKey(mode)];
  sceneImage.alt = selected.alt;
  imageContourLegend.hidden = !selected.showLegend;
  for (const [key, config] of Object.entries(modes)) {
    config.button.classList.toggle("active", key === mode);
  }
}

function imageModeKey(mode) {
  return mode === "contours" ? "contour_debug" : mode;
}

function refreshActiveViewPanels(appState) {
  populatePanels(appState);
  setupComparisonPanel(appState.activeView);
  selectImageMode(appState, imageToggleModes(), appState.imageMode);
}

function imageToggleModes() {
  return {
    annotated: {
      alt: "Synthetic mountain view with peak labels",
      button: showAnnotated,
      showLegend: false,
    },
    contours: {
      alt: "True and estimated 2D skyline contour comparison",
      button: showContours,
      showLegend: true,
    },
    render: {
      alt: "Synthetic mountain view",
      button: showRender,
      showLegend: false,
    },
  };
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

  addPeakLabels(scene, data, frame);

  const viewConfigs = cameraConfigsFor(appState.views, frame);
  const overlays = addFovOverlays(scene, data, frame, viewConfigs);
  const cameraMarkers = addCameraMarkers(scene, viewConfigs);
  const cameraPickTargets = collectCameraPickTargets(cameraMarkers);
  const connectors = addCameraConnectors(scene, viewConfigs);
  camera.updateMatrixWorld(true);
  scene.updateMatrixWorld(true);

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
    mode: CAMERA_MODES.compare,
    selectedFootprint: null,
  };

  setupMarkerPicking(viewer);
  setupCameraModeControls(viewer);
  applyCameraMode(viewer, CAMERA_MODES.compare);

  const resizeObserver = new ResizeObserver(() => resizeRenderers(renderer, labelRenderer, camera));
  resizeObserver.observe(canvas);
  resizeRenderers(renderer, labelRenderer, camera);

  renderer.setAnimationLoop(() => {
    controls.update();
    camera.updateMatrixWorld();
    scene.updateMatrixWorld();
    updateLabelOcclusion(viewer.labelOcclusion, camera);
    updateScaleHud(viewer.scaleHud, camera);
    renderer.render(scene, camera);
    labelRenderer.render(scene, camera);
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
  for (const button of document.querySelectorAll("[data-camera-mode]")) {
    button.addEventListener("click", () => {
      applyCameraMode(viewer, button.dataset.cameraMode);
    });
  }
}

function applyCameraMode(viewer, mode) {
  viewer.mode = mode;
  viewer.selectedFootprint = null;
  for (const button of document.querySelectorAll("[data-camera-mode]")) {
    button.classList.toggle("active", button.dataset.cameraMode === mode);
  }

  const viewConfig = activeViewConfig(viewer);
  if (mode === CAMERA_MODES.true || mode === CAMERA_MODES.estimated) {
    const config = viewConfig[mode];
    viewer.controls.enabled = false;
    viewer.camera.position.copy(config.position);
    viewer.camera.lookAt(config.target);
    viewer.controls.target.copy(config.target);
    setOverlayVisibility(viewer, [mode]);
    setConnectorVisibility(viewer, false);
    document.getElementById("selectedCamera").textContent = `${viewConfig.label}: ${config.label}`;
    document.getElementById("selectedCameraDescription").textContent =
      "Locked to camera POV; highlighted terrain is inside this camera frame.";
  } else if (mode === CAMERA_MODES.compare) {
    viewer.controls.enabled = true;
    viewer.camera.position.copy(CAMERA_INITIAL_POSITION);
    viewer.controls.target.copy(CAMERA_TARGET);
    viewer.controls.update();
    setOverlayVisibility(viewer, [CAMERA_MODES.true, CAMERA_MODES.estimated]);
    setConnectorVisibility(viewer, true);
    document.getElementById("selectedCamera").textContent = `${viewConfig.label}: Compare`;
    document.getElementById("selectedCameraDescription").textContent =
      "Blue is true camera coverage; amber is estimated camera coverage.";
  } else {
    viewer.controls.enabled = true;
    viewer.camera.position.copy(CAMERA_INITIAL_POSITION);
    viewer.controls.target.copy(CAMERA_TARGET);
    viewer.controls.update();
    setOverlayVisibility(viewer, []);
    setConnectorVisibility(viewer, false);
    document.getElementById("selectedCamera").textContent = "Orbit";
    document.getElementById("selectedCameraDescription").textContent =
      "Free orbit view; camera footprints are hidden.";
  }

  updateMarkerSelection(viewer);
}

function activeViewConfig(viewer) {
  return viewer.viewConfigs.find((viewConfig) => viewConfig.id === viewer.activeViewId) ?? viewer.viewConfigs[0];
}

function setOverlayVisibility(viewer, visibleRoles) {
  const visible = new Set(visibleRoles);
  for (const viewConfig of viewer.viewConfigs) {
    const viewOverlays = viewer.overlays[viewConfig.id];
    viewOverlays.true.visible = viewConfig.id === viewer.activeViewId && visible.has(CAMERA_MODES.true);
    viewOverlays.estimated.visible = viewConfig.id === viewer.activeViewId && visible.has(CAMERA_MODES.estimated);
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

  const material = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.88,
    metalness: 0,
    side: THREE.FrontSide,
  });

  return new THREE.Mesh(geometry, material);
}

function createIsolines(terrain) {
  const buffers = {
    core: [],
    halo: [],
  };
  const firstLevel = Math.ceil(terrain.elevation_min_m / ISOLINE_INTERVAL_M) * ISOLINE_INTERVAL_M;
  for (
    let level = firstLevel;
    level <= terrain.elevation_max_m;
    level += ISOLINE_INTERVAL_M
  ) {
    addIsolineLevel(buffers, terrain, level);
  }

  const group = new THREE.Group();
  group.add(createIsolineMesh(buffers.halo, 0xf2e2b4, 0.62, 2));
  group.add(createIsolineMesh(buffers.core, 0x1b1710, 0.86, 3));
  return group;
}

function createIsolineMesh(positions, color, opacity, renderOrder) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  const material = new THREE.MeshBasicMaterial({
    color,
    depthWrite: false,
    polygonOffset: true,
    polygonOffsetFactor: -2,
    polygonOffsetUnits: -2,
    side: THREE.DoubleSide,
    transparent: true,
    opacity,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.renderOrder = renderOrder;
  return mesh;
}

function addIsolineLevel(buffers, terrain, level) {
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
        pushIsolineSegment(buffers, terrain, unique[0], unique[1]);
      } else if (unique.length === 4) {
        pushIsolineSegment(buffers, terrain, unique[0], unique[1]);
        pushIsolineSegment(buffers, terrain, unique[2], unique[3]);
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

function pushIsolineSegment(buffers, terrain, a, b) {
  const aPoint = terrainFloatPoint(terrain, a.row, a.col, a.elevation, ISOLINE_VERTICAL_OFFSET);
  const bPoint = terrainFloatPoint(terrain, b.row, b.col, b.elevation, ISOLINE_VERTICAL_OFFSET);
  pushRibbonSegment(buffers.halo, aPoint, bPoint, ISOLINE_HALO_WIDTH);
  aPoint.y += 0.001;
  bPoint.y += 0.001;
  pushRibbonSegment(buffers.core, aPoint, bPoint, ISOLINE_CORE_WIDTH);
}

function pushRibbonSegment(positions, a, b, width) {
  const dx = b.x - a.x;
  const dz = b.z - a.z;
  const length = Math.hypot(dx, dz);
  if (length <= 1e-8) {
    return;
  }

  const offsetX = (-dz / length) * width * 0.5;
  const offsetZ = (dx / length) * width * 0.5;
  const aLeft = new THREE.Vector3(a.x + offsetX, a.y, a.z + offsetZ);
  const aRight = new THREE.Vector3(a.x - offsetX, a.y, a.z - offsetZ);
  const bLeft = new THREE.Vector3(b.x + offsetX, b.y, b.z + offsetZ);
  const bRight = new THREE.Vector3(b.x - offsetX, b.y, b.z - offsetZ);
  pushTriangle(positions, aLeft, aRight, bLeft);
  pushTriangle(positions, aRight, bRight, bLeft);
}

function terrainGeometry(terrain) {
  const gridWidth = terrain.grid_width;
  const gridHeight = terrain.grid_height;
  const positions = new Float32Array(gridWidth * gridHeight * 3);
  const colors = new Float32Array(gridWidth * gridHeight * 3);
  const indices = [];

  for (let row = 0; row < gridHeight; row += 1) {
    for (let col = 0; col < gridWidth; col += 1) {
      const pointIndex = row * gridWidth + col;
      const offset = pointIndex * 3;
      const point = terrainGridPoint(terrain, row, col, 0);
      positions[offset] = point.x;
      positions[offset + 1] = point.y;
      positions[offset + 2] = point.z;

      const color = terrainColor(elevationRatio(terrain, terrain.elevation_m[row][col]));
      colors[offset] = color.r;
      colors[offset + 1] = color.g;
      colors[offset + 2] = color.b;
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
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
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
    marker.position.y += 0.012;
    scene.add(marker);

    const label = createLabel(peak.name, "peak-label");
    label.position.copy(marker.position);
    label.position.y += 0.058;
    label.userData.occlusionAnchor = marker;
    scene.add(label);
  }
}

function createPeakMarker() {
  const group = new THREE.Group();
  const markerMaterial = new THREE.MeshBasicMaterial({
    color: 0xd84232,
    polygonOffset: true,
    polygonOffsetFactor: -2,
    polygonOffsetUnits: -2,
    side: THREE.DoubleSide,
  });
  const triangle = new THREE.Mesh(createPeakTriangleGeometry(0.044), markerMaterial);
  triangle.position.y = 0.008;
  group.add(triangle);

  const outline = new THREE.LineLoop(
    createPeakTriangleGeometry(0.048),
    new THREE.LineBasicMaterial({
      color: 0x29100d,
      depthWrite: false,
      transparent: true,
      opacity: 0.9,
    }),
  );
  outline.position.y = 0.01;
  group.add(outline);
  return group;
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
        CAMERA_MODES.true,
        "True camera",
        view.label,
        view.true_camera,
        0x67c7ff,
        compactEstimate ? 0 : -0.055,
        compactEstimate ? 0 : -0.035,
        false,
        frame,
      ),
      estimated: cameraConfig(
        view,
        CAMERA_MODES.estimated,
        compactEstimate ? "Predicted position" : "Estimated camera",
        compactEstimate ? "Fit" : `${view.label} fit`,
        view.pose_estimate.extrinsics,
        0xffd15a,
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
  return viewer.mode === CAMERA_MODES.compare || role === viewer.mode;
}

function addFovOverlays(scene, data, frame, viewConfigs) {
  const overlays = {};
  for (const viewConfig of viewConfigs) {
    overlays[viewConfig.id] = {
      true: createFovOverlay(data, frame, viewConfig.true, 0x67c7ff),
      estimated: createFovOverlay(data, frame, viewConfig.estimated, 0xffd15a),
    };
    scene.add(overlays[viewConfig.id].true);
    scene.add(overlays[viewConfig.id].estimated);
  }
  return overlays;
}

function createFovOverlay(data, frame, config, color) {
  const terrain = data.terrain;
  const projection = cameraProjection(config.extrinsics, data.scene.intrinsics);
  const visibleTriangleIds = visibleTerrainTriangleIds(terrain, frame, projection);
  const positions = [];
  for (let row = 0; row < terrain.grid_height - 1; row += 1) {
    for (let col = 0; col < terrain.grid_width - 1; col += 1) {
      const a = terrainLocalGridPoint(terrain, frame, row, col);
      const b = terrainLocalGridPoint(terrain, frame, row, col + 1);
      const c = terrainLocalGridPoint(terrain, frame, row + 1, col);
      const d = terrainLocalGridPoint(terrain, frame, row + 1, col + 1);
      addOverlayTriangle(positions, frame, projection, visibleTriangleIds, terrainTriangleId(terrain, row, col, 0), [
        a,
        b,
        c,
      ]);
      addOverlayTriangle(positions, frame, projection, visibleTriangleIds, terrainTriangleId(terrain, row, col, 1), [
        b,
        d,
        c,
      ]);
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  const material = new THREE.MeshBasicMaterial({
    color,
    transparent: true,
    opacity: 0.33,
    depthWrite: false,
    polygonOffset: true,
    polygonOffsetFactor: -1,
    polygonOffsetUnits: -1,
    side: THREE.FrontSide,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.renderOrder = 1;
  mesh.visible = false;
  return mesh;
}

function addOverlayTriangle(positions, frame, projection, visibleTriangleIds, triangleId, localVertices) {
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
  const mode = pick.cameraMode;
  const config = viewConfig[mode];
  if (!config) {
    return;
  }

  viewer.selectedFootprint = mode;
  viewer.controls.enabled = true;
  setOverlayVisibility(viewer, [mode]);
  setConnectorVisibility(viewer, false);
  document.getElementById("selectedCamera").textContent = `${viewConfig.label}: ${config.label} footprint`;
  document.getElementById("selectedCameraDescription").textContent =
    "Free orbit view; highlighted terrain is visible from this camera.";
  updateMarkerSelection(viewer);
}

function clearCameraFootprint(viewer) {
  if (!viewer.selectedFootprint) {
    return;
  }
  viewer.selectedFootprint = null;
  const viewConfig = activeViewConfig(viewer);
  if (viewer.mode === CAMERA_MODES.compare) {
    setOverlayVisibility(viewer, [CAMERA_MODES.true, CAMERA_MODES.estimated]);
    setConnectorVisibility(viewer, true);
    document.getElementById("selectedCamera").textContent = `${viewConfig.label}: Compare`;
    document.getElementById("selectedCameraDescription").textContent =
      "Blue is true camera coverage; amber is estimated camera coverage.";
  } else if (viewer.mode === CAMERA_MODES.orbit) {
    setOverlayVisibility(viewer, []);
    setConnectorVisibility(viewer, false);
    document.getElementById("selectedCamera").textContent = "Orbit";
    document.getElementById("selectedCameraDescription").textContent =
      "Free orbit view; camera footprints are hidden.";
  }
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

function projectLocalPoint(point, extrinsics, intrinsics) {
  const axes = cameraAxes(extrinsics);
  const cameraPoint = localToCameraPoint(point, extrinsics, axes);
  return projectCameraPoint(cameraPoint, intrinsics);
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

function createPeakTriangleGeometry(size) {
  const geometry = new THREE.BufferGeometry();
  const halfWidth = size * 0.5;
  const halfDepth = size * 0.45;
  const vertices = new Float32Array([
    0, 0, -halfDepth,
    -halfWidth, 0, halfDepth,
    halfWidth, 0, halfDepth,
  ]);
  geometry.setAttribute("position", new THREE.BufferAttribute(vertices, 3));
  geometry.setIndex([0, 1, 2]);
  geometry.computeVertexNormals();
  return geometry;
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

function terrainColor(t) {
  const low = new THREE.Color(0x2f6843);
  const mid = new THREE.Color(0x92955d);
  const high = new THREE.Color(0xdad1b3);
  if (t < 0.58) {
    return low.lerp(mid, t / 0.58);
  }
  return mid.lerp(high, (t - 0.58) / 0.42);
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

function elevationRatio(terrain, elevation) {
  return normalize(elevation, terrain.elevation_min_m, terrain.elevation_max_m);
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

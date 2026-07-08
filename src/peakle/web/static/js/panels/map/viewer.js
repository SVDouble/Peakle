"use strict";

// The 3D map: orbitable terrain with peak labels, click-to-place camera creation,
// the selected view's ground-truth + predicted cameras with FOV coverage, and a
// Map/POV switch backed by a small pose table (ground truth + solves).

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer } from "three/addons/renderers/CSS2DRenderer.js";

import { api } from "../../api.js";
import {
  PHYSICAL_TERRAIN_VERTICAL_SCALE,
  cameraTargetScenePoint,
  localToScenePoint,
  scaledTerrainFrame,
  sceneToLocalEastNorth,
  terrainFrame,
} from "../../geometry.js";
import { el, fitContainBox, formatNumber, svgElement } from "../../format.js";
import { GT_LAYER_NAMES } from "../camera-image.js";
import { buildSelectionLayer } from "./cameras.js";
import { buildGtSpotsLayer, terrainElevationAt } from "./gt-spots.js";
import { addPeakLabels, createTerrainGroup } from "./terrain-mesh.js";

const CAMERA_INITIAL_POSITION = new THREE.Vector3(0.1, 1.55, 2.65);
const CAMERA_TARGET = new THREE.Vector3(0, 0.22, 0);
const PICK_MAX_DRAG_PX = 5;
const LABEL_OCCLUSION_MARGIN = 0.018;
const SCENE_DEM_SKY_LAYER = "dem_sky";

export function setupMapPanel(store, root) {
  root.classList.add("map-panel");
  const canvas = el("canvas", { id: "mapCanvas" });
  // The canvas + labels live in a viewport box that fills the panel in map mode
  // and shrinks to the camera image's aspect ratio in POV modes (letterboxing),
  // so the through-the-lens framing matches the rendered image instead of being
  // cropped to the panel's shape.
  // In POV for a GT sample this holds the same outline layer PNGs the Inspect panel shows,
  // so the map and the inspector stay in sync (the POV camera matches the GT pose's fov/aspect,
  // so the photo-coordinate overlays line up with the 3D render).
  const povOverlay = el("div", { class: "pov-overlay" });
  // The GT photograph, overlaid on the 3D render in a fixed POV so you can align the DEM to the
  // photo by eye (opacity slider below); it sits under the outline overlay, over the canvas.
  const povPhoto = el("img", { class: "pov-photo", alt: "" });
  const viewport = el("div", { class: "map-viewport" }, [canvas, povPhoto, povOverlay]);
  const frameBox = el("div", { class: "frame-box", id: "mapFrameBox" }, [viewport]);
  const hint = el("p", { class: "control-hint", id: "mapHint" });
  const povControls = el(
    "div",
    { class: "segmented" },
    ["map", "pov"].map((povMode) =>
      el("button", {
        type: "button",
        class: povMode === "map" ? "active" : "",
        dataset: { pov: povMode },
        text: povMode === "map" ? "Map" : "POV",
      }),
    ),
  );
  const povTable = el("table", { class: "pov-solutions", hidden: true });
  // Photo-overlay opacity control (only meaningful in a POV that has a photo).
  const opacityRange = el("input", {
    type: "range", min: "0", max: "1", step: "0.05", value: String(store.photoOpacity),
    oninput: () => store.setPhotoOpacity(Number(opacityRange.value)),
  });
  const opacityControl = el("label", { class: "photo-opacity", hidden: true }, [
    el("span", { text: "Photo" }),
    opacityRange,
  ]);
  root.replaceChildren(frameBox, el("div", { class: "map-hud" }, [povControls, povTable, opacityControl, hint]));

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
  viewport.append(labelRenderer.domElement);

  // Blender-style navigation: orbit on left/middle drag, screen-space pan on
  // right drag, zoom toward the cursor, and (nearly) full vertical orbit.
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.copy(CAMERA_TARGET);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.minDistance = 0.08;
  controls.maxDistance = 6.0;
  controls.minPolarAngle = 0.02;
  controls.maxPolarAngle = Math.PI - 0.05;
  controls.screenSpacePanning = true;
  controls.zoomToCursor = true;
  controls.rotateSpeed = 0.85;
  controls.panSpeed = 0.8;
  controls.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.ROTATE, RIGHT: THREE.MOUSE.PAN };
  controls.touches = { ONE: THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_PAN };

  let frame = null;
  let terrainGroup = null;
  let terrainMesh = null;
  let terrainControls = null;
  let peakLabels = [];
  let peakGroup = null;
  let selectionLabels = [];
  let selectionLayer = null;
  let gtSpotsLayer = null;
  let gtSpotLabels = [];
  let mode = "map";
  let povPoseKey = null;
  let lastSelectionKey = null;
  let lastSelectedSolveId = null;
  let povAspect = null; // aspect of the active POV camera image, for letterboxing
  // One raycaster, used only for click-to-place picking (a single cast per click).
  // Label occlusion samples the elevation heightfield instead: raycasting the full
  // terrain mesh per label per frame collapsed to ~4 fps on a focused 320x320 grid
  // (user-reported lag).
  const raycaster = new THREE.Raycaster();

  function rebuildTerrain() {
    if (!store.terrain) {
      return;
    }
    if (terrainGroup) {
      scene.remove(terrainGroup);
    }
    if (peakGroup) {
      scene.remove(peakGroup);
    }
    // Sweep peak-label DOM nodes wholesale: a CSS2DObject leaving the scene never removes its
    // element, so old text would otherwise pile up (and the sprites hang) after a map switch.
    for (const node of labelRenderer.domElement.querySelectorAll(".peak-label")) {
      node.remove();
    }
    frame = terrainFrame(store.terrain);
    const built = createTerrainGroup(store.terrain, frame, store.peaks);
    terrainGroup = built.group;
    terrainMesh = built.mesh;
    terrainControls = built;
    scene.add(terrainGroup);
    applyDisplay();
    const peakBuilt = addPeakLabels(scene, store.peaks, frame);
    peakGroup = peakBuilt.group;
    peakLabels = peakBuilt.labels;
    rebuildSelection();
    rebuildGtSpots();
  }

  function applyDisplay() {
    if (!terrainControls) {
      return;
    }
    terrainControls.setShadingMode(store.display.shadingMode);
    terrainControls.setContours(store.display.contours);
  }

  function rebuildSelection() {
    syncPovSelection();
    renderPovTable();
    if (!frame) {
      return;
    }
    if (selectionLayer) {
      scene.remove(selectionLayer);
    }
    // Sweep every camera label out of the DOM regardless of which flow removed
    // its object: CSS2DObjects nested in marker groups never fire `removed` when
    // an ancestor leaves the scene, and labels still in the new layer re-append
    // themselves on the next render pass. This makes stale labels impossible.
    for (const node of labelRenderer.domElement.querySelectorAll(".camera-label")) {
      node.remove();
    }
    selectionLayer = buildSelectionLayer(store.terrain, frame, store.views, store.selectedViewId, store.selectedSolve());
    scene.add(selectionLayer);
    selectionLabels = [];
    selectionLayer.traverse((object) => {
      if (object.isCSS2DObject) {
        selectionLabels.push(object);
      }
    });
    if (mode !== "map") {
      applyPov(mode);
    }
  }

  function rebuildGtSpots() {
    if (!frame || !store.terrain) {
      return;
    }
    if (gtSpotsLayer) {
      scene.remove(gtSpotsLayer);
    }
    // Same wholesale DOM sweep as camera labels: chips of removed spots must not
    // survive a rebuild; live ones re-append on the next render pass.
    for (const node of labelRenderer.domElement.querySelectorAll(".gt-spot")) {
      node.remove();
    }
    gtSpotsLayer = buildGtSpotsLayer(
      store.terrain,
      frame,
      store.gtSamples,
      store.selectedGtName,
      (name) => store.selectGtSample(name, { focus: true }),
      (sample) => {
        hint.textContent = `Recentering map on ${sample.name}…`;
        store.focusGtSample(sample).catch((error) => {
          hint.textContent = error.message;
        });
      },
    );
    scene.add(gtSpotsLayer);
    gtSpotLabels = [];
    gtSpotsLayer.traverse((object) => {
      if (object.isCSS2DObject) {
        gtSpotLabels.push(object);
      }
    });
    syncWorldVerticalScale();
  }

  function activeTerrainFrame() {
    return frame && mode === "pov" ? scaledTerrainFrame(frame, PHYSICAL_TERRAIN_VERTICAL_SCALE) : frame;
  }

  function syncWorldVerticalScale() {
    // Orbit mode uses mild relief emphasis for readability. Through-the-lens POV must use
    // physical vertical scale, or the 3D silhouette drifts away from photo-coordinate overlays.
    const verticalScale = mode === "pov" ? PHYSICAL_TERRAIN_VERTICAL_SCALE : 1;
    for (const group of [terrainGroup, peakGroup, selectionLayer, gtSpotsLayer]) {
      if (group) {
        group.scale.y = verticalScale;
      }
    }
    labelRenderer.domElement.style.display = mode === "pov" ? "none" : "";
  }

  function currentSelectionKey() {
    return store.selectedViewId ? `view:${store.selectedViewId}` : store.selectedGtName ? `gt:${store.selectedGtName}` : "";
  }

  function syncPovSelection() {
    const selectionKey = currentSelectionKey();
    if (selectionKey !== lastSelectionKey) {
      povPoseKey = null;
      lastSelectionKey = selectionKey;
    }
    if (store.selectedSolveId && store.selectedSolveId !== lastSelectedSolveId) {
      povPoseKey = `solve:${store.selectedSolveId}`;
    }
    lastSelectedSolveId = store.selectedSolveId;
  }

  function povRows() {
    const view = store.selectedView();
    if (view?.true_extrinsics) {
      const cam = store.selectedCamera();
      if (!cam) {
        return [];
      }
      const rows = [
        {
          key: "truth",
          role: view.source === "gt" ? "GT" : "Truth",
          name: view.source === "gt" ? "Dataset pose" : "Ground truth",
          detail: `${formatNumber(view.true_extrinsics.yaw_deg, "deg")} · ${formatNumber(view.true_extrinsics.pitch_deg, "deg")}`,
          pose: cam.scenePoseFromLocal(view.true_extrinsics, store.terrain),
          label: `${view.label} · Ground truth`,
          photoSrc: view.photo_url ?? null,
        },
      ];
      for (const summary of view.solves) {
        rows.push({
          key: `solve:${summary.id}`,
          role: "Solve",
          name: strategyLabel(summary.strategy),
          detail: `yaw ${formatNumber(summary.metrics.yaw_error_deg, "deg")} · fit ${formatNumber(summary.metrics.contour_mae_px, "px")}`,
          pose: cam.scenePoseFromLocal(summary.extrinsics, store.terrain),
          label: `${view.label} · ${strategyLabel(summary.strategy)}`,
          photoSrc: view.photo_url ?? null,
          solveId: summary.id,
        });
      }
      return rows;
    }

    const sample = store.selectedGtSample();
    if (!sample) {
      return [];
    }
    const cam = store.selectedCamera();
    const pose = cam?.scenePose(store.terrain, "true");
    return [
      {
        key: "truth",
        role: "GT",
        name: "Dataset pose",
        detail: `${sample.quality} · fov ${formatNumber(sample.fov_deg, "deg")}`,
        pose,
        label: sample.name,
        photoSrc: cam?.photoSrc ?? null,
        layersSample: cam?.hasLayers ? sample : null,
      },
    ];
  }

  function strategyLabel(name) {
    return store.scene?.strategies.find((s) => s.name === name)?.label ?? name;
  }

  function activePov() {
    const rows = povRows();
    if (!rows.length) {
      return null;
    }
    if (!povPoseKey || !rows.some((row) => row.key === povPoseKey)) {
      povPoseKey = store.selectedSolveId && rows.some((row) => row.key === `solve:${store.selectedSolveId}`) ? `solve:${store.selectedSolveId}` : rows[0].key;
    }
    const row = rows.find((candidate) => candidate.key === povPoseKey) ?? rows[0];
    return row.pose ? row : null;
  }

  function renderPovTable() {
    syncPovSelection();
    const rows = povRows();
    povTable.hidden = rows.length === 0;
    if (!rows.length) {
      povTable.replaceChildren();
      return;
    }
    activePov();
    const body = el(
      "tbody",
      {},
      rows.map((row) =>
        el(
          "tr",
          {
            class: row.key === povPoseKey ? "active" : "",
            onclick: () => selectPovRow(row),
          },
          [
            el("th", { text: row.role }),
            el("td", {}, [el("span", { class: "pov-solution-name", text: row.name }), el("span", { class: "pov-solution-detail", text: row.detail })]),
          ],
        ),
      ),
    );
    povTable.replaceChildren(body);
  }

  function selectPovRow(row) {
    povPoseKey = row.key;
    if (row.solveId && store.selectedViewId && row.solveId !== store.selectedSolveId) {
      store.selectSolve(store.selectedViewId, row.solveId).catch((error) => {
        hint.textContent = error.message;
      });
    }
    applyPov("pov");
  }

  // Sync the POV outline overlay with the Inspect panel: same enabled layers, same sample. Only a
  // GT camera has precomputed outline layers; a placed view shows its skyline via the inspector SVG.
  function updatePovOverlay() {
    const sample = mode === "pov" ? activePov()?.layersSample : null;
    if (!sample) {
      if (povOverlay.childElementCount) {
        povOverlay.replaceChildren();
      }
      return;
    }
    const query = gtAdjustQuery();
    const layers = GT_LAYER_NAMES.filter((layer) => store.gtDisplay[layer]);
    const want = layers.map((layer) => (layer === SCENE_DEM_SKY_LAYER ? sceneSkylineKey(sample, query) : api.gtLayerUrl(sample.name, layer)));
    const have = [...povOverlay.children].map((node) => node.dataset.src);
    if (want.join("|") !== have.join("|")) {
      povOverlay.replaceChildren(
        ...layers.map((layer) => {
          if (layer === SCENE_DEM_SKY_LAYER) {
            return createSceneSkylineLayer(sample, query);
          }
          const src = api.gtLayerUrl(sample.name, layer);
          const img = el("img", { class: "pov-layer", alt: "" });
          img.dataset.src = src;
          img.src = src;
          return img;
        }),
      );
    }
  }

  function gtAdjustQuery() {
    const adjust = store.gtAdjust ?? {};
    return new URLSearchParams({
      dyaw: String(adjust.dyaw ?? 0),
      de: String(adjust.de ?? 0),
      dn: String(adjust.dn ?? 0),
      dv: String(adjust.dv ?? 0),
    }).toString();
  }

  function sceneSkylineKey(sample, query) {
    return `scene-dem-sky:${sample.name}?${query}`;
  }

  function createSceneSkylineLayer(sample, query) {
    const key = sceneSkylineKey(sample, query);
    const svg = svgElement("svg", {
      class: "pov-layer pov-vector",
      viewBox: `0 0 ${sample.width} ${sample.height}`,
      preserveAspectRatio: "none",
    });
    svg.dataset.src = key;
    fetch(`/api/gt/samples/${encodeURIComponent(sample.name)}/scene-skyline?${query}`)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`${response.status} ${response.statusText}`);
        }
        return response.json();
      })
      .then(({ rows }) => {
        if (svg.dataset.src !== key) {
          return;
        }
        const points = rows
          .map((row, col) => (row === null || row === undefined ? null : `${col.toFixed(1)},${Number(row).toFixed(1)}`))
          .filter(Boolean)
          .join(" ");
        svg.replaceChildren(svgElement("polyline", { class: "pov-dem-skyline", points }));
      })
      .catch(() => {
        if (svg.dataset.src === key) {
          svg.replaceChildren();
        }
      });
    return svg;
  }

  function applyPov(nextMode) {
    mode = nextMode;
    syncWorldVerticalScale();
    syncPovButtons();
    renderPovTable();
    const active = nextMode === "map" ? null : activePov();
    if (!active) {
      mode = "map";
      syncWorldVerticalScale();
      syncPovButtons();
      controls.enabled = true;
      camera.fov = 45;
      camera.position.copy(CAMERA_INITIAL_POSITION);
      controls.target.copy(CAMERA_TARGET);
      controls.update();
      resize();
      updatePovOverlay();
      updatePovPhoto();
      hint.textContent = store.placing ? "Click the map to place a camera." : "Free orbit. Select a view or GT sample, then choose a POV.";
      return;
    }
    povAspect = active.pose.aspect;
    controls.enabled = false;
    camera.fov = active.pose.vfovDeg;
    const povFrame = activeTerrainFrame();
    if (!povFrame) {
      return;
    }
    camera.position.copy(localToScenePoint(active.pose.position, povFrame));
    camera.up.set(0, 1, 0);
    camera.lookAt(cameraTargetScenePoint(active.pose, povFrame));
    // resize() recomputes the letterbox box for the new mode and sets the matching aspect.
    resize();
    updatePovOverlay();
    updatePovPhoto();
    hint.textContent = `Looking through ${active.label}.`;
  }

  // The GT photograph overlaid on the 3D render for eyeball alignment; visible only in a POV that
  // has a photo (a GT sample or a GT-derived view), driven by the opacity slider.
  function updatePovPhoto() {
    const src = mode === "pov" ? (activePov()?.photoSrc ?? null) : null;
    opacityControl.hidden = !src;
    if (!src) {
      povPhoto.style.display = "none";
      return;
    }
    if (povPhoto.dataset.src !== src) {
      povPhoto.dataset.src = src;
      povPhoto.src = src;
    }
    povPhoto.style.opacity = String(store.photoOpacity);
    povPhoto.style.display = store.photoOpacity > 0 ? "block" : "none";
  }

  function syncPovButtons() {
    for (const button of root.querySelectorAll("[data-pov]")) {
      button.classList.toggle("active", button.dataset.pov === mode);
    }
  }

  for (const button of root.querySelectorAll("[data-pov]")) {
    button.addEventListener("click", () => applyPov(button.dataset.pov));
  }

  // Double-click any terrain point to recenter the geographic window there (the
  // in-3D "move the map"): raycast the terrain, convert the hit to lat/lon via the
  // frame's geographic corners, and focus the heightmap on it.
  canvas.addEventListener("dblclick", (event) => {
    if (mode !== "map" || !terrainMesh || !store.terrain?.lat_min_deg) {
      return;
    }
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
    const t = store.terrain;
    const lon = t.lon_min_deg + ((local.east_m - t.x_min_m) / (t.x_max_m - t.x_min_m)) * (t.lon_max_deg - t.lon_min_deg);
    const lat = t.lat_min_deg + ((local.north_m - t.y_min_m) / (t.y_max_m - t.y_min_m)) * (t.lat_max_deg - t.lat_min_deg);
    hint.textContent = "Recentering map…";
    store.focusScene(lat, lon).then(
      () => {
        if (mode === "map") {
          hint.textContent = store.placing ? "Click the map to place a camera." : "Free orbit. Select a view or GT sample, then choose a POV.";
        }
      },
      (error) => {
        hint.textContent = error.message;
      },
    );
  });

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

  // Heightfield line-of-sight: walk the camera->anchor segment and compare each
  // sample's scene height against the terrain surface. O(steps) per label vs a
  // full-mesh raycast; the last stretch near the anchor is skipped so a label
  // hugging its own slope does not self-occlude.
  const OCCLUSION_STEPS = 36;

  function terrainOccludes(camPos, target) {
    const sceneFrame = activeTerrainFrame();
    if (!sceneFrame) {
      return false;
    }
    const t1 = 1 - Math.max(LABEL_OCCLUSION_MARGIN / Math.max(camPos.distanceTo(target), 1e-6), 0.04);
    const zSpan = Math.max(sceneFrame.zMax - sceneFrame.zMin, 1);
    for (let i = 1; i <= OCCLUSION_STEPS; i += 1) {
      const t = (i / OCCLUSION_STEPS) * t1;
      const x = camPos.x + (target.x - camPos.x) * t;
      const y = camPos.y + (target.y - camPos.y) * t;
      const z = camPos.z + (target.z - camPos.z) * t;
      const local = sceneToLocalEastNorth({ x, z }, sceneFrame);
      if (
        local.east_m < sceneFrame.xMin ||
        local.east_m > sceneFrame.xMax ||
        local.north_m < sceneFrame.yMin ||
        local.north_m > sceneFrame.yMax
      ) {
        continue;
      }
      const elev = terrainElevationAt(store.terrain, local.east_m, local.north_m);
      const surfaceY = ((elev - sceneFrame.zMin) / zSpan) * (sceneFrame.sceneH ?? 0.66);
      if (y < surfaceY - 0.002) {
        return true;
      }
    }
    return false;
  }

  function updateLabelOcclusion() {
    const cameraPosition = new THREE.Vector3().setFromMatrixPosition(camera.matrixWorld);
    const target = new THREE.Vector3();
    const projected = new THREE.Vector3();
    for (const label of [...peakLabels, ...selectionLabels, ...gtSpotLabels]) {
      const anchor = label.userData.occlusionAnchor ?? label;
      anchor.getWorldPosition(target);
      projected.copy(target).project(camera);
      const inView = projected.z >= -1 && projected.z <= 1 && Math.abs(projected.x) <= 1.08 && Math.abs(projected.y) <= 1.08;
      label.visible = inView && !(store.terrain && frame && terrainOccludes(cameraPosition, target));
    }
  }

  function viewportBox() {
    const width = Math.max(1, frameBox.clientWidth);
    const height = Math.max(1, frameBox.clientHeight);
    if (mode !== "map" && povAspect) {
      return fitContainBox(width, height, povAspect);
    }
    return { width, height, left: 0, top: 0 };
  }

  function resize() {
    const box = viewportBox();
    viewport.style.left = `${box.left}px`;
    viewport.style.top = `${box.top}px`;
    viewport.style.width = `${box.width}px`;
    viewport.style.height = `${box.height}px`;
    renderer.setSize(box.width, box.height, false);
    labelRenderer.setSize(box.width, box.height);
    camera.aspect = box.width / box.height;
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
  store.on("display", applyDisplay);
  store.on("views", rebuildSelection);
  store.on("selection", rebuildSelection);
  store.on("gt", () => {
    syncPovSelection();
    renderPovTable();
    rebuildGtSpots();
    // If a GT sample is selected while in a POV mode, re-aim the camera through it;
    // if the selection cleared the pose, fall back to map.
    if (mode !== "map") {
      applyPov(mode);
    }
  });
  // Outline toggles in the Inspect panel drive the POV overlay too (map <-> inspect in sync).
  store.on("gt-display", updatePovOverlay);
  // Pose adjustment re-aims the active GT POV live; the opacity slider updates the photo overlay.
  store.on("gt-adjust", () => {
    renderPovTable();
    if (mode !== "map") {
      applyPov(mode);
    }
  });
  store.on("photo-opacity", updatePovPhoto);
  store.on("placing", () => {
    viewport.classList.toggle("placing", store.placing);
    if (mode === "map") {
      hint.textContent = store.placing ? "Click the map to place a camera." : "Free orbit. Select a view and choose a POV.";
    }
  });
  rebuildTerrain();
  renderPovTable();
  syncPovButtons();
}

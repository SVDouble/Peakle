"use strict";

// Client-side mirror of the server scene plus a tiny pub/sub. Panels subscribe
// to change events and read state through getters; all mutations go through the
// async actions, which call the API and then refresh local state.

import { api } from "./api.js";
import { gtCamera, viewCamera } from "./camera.js";
import { geoDistanceMeters, geoToLocal } from "./geometry.js";

class Store {
  constructor() {
    this.scene = null;
    this.terrain = null;
    this.peaks = [];
    this.views = [];
    this.selectedViewId = null;
    this.selectedSolveId = null;
    this.selectedPoseKey = "truth";
    this.placing = false;
    this.solveCache = new Map();
    // Client-side map appearance, applied live by the viewer (never round-trips
    // to the server). `shadingMode` is one of terrain-mesh.js' SHADING_MODES ids.
    this.display = { shadingMode: "relief", contours: true };
    // GT dataset slice: samples load lazily on first use; selecting a GT sample
    // and selecting a view are mutually exclusive (one inspector).
    this.gtSamples = null;
    this.gtError = null;
    this.selectedGtName = null;
    // Per-layer visibility (keys = GT Lab layer names). Skylines on by default.
    this.gtDisplay = { gt_sky: true, dem_sky: true };
    this.photoOpacity = 0.5; // GT photo overlaid on the 3D terrain in POV
    this._gtLoading = false;
    this._listeners = new Map();
  }

  on(event, callback) {
    if (!this._listeners.has(event)) {
      this._listeners.set(event, new Set());
    }
    this._listeners.get(event).add(callback);
    return () => this._listeners.get(event)?.delete(callback);
  }

  emit(event) {
    for (const callback of this._listeners.get(event) ?? []) {
      callback(this);
    }
    for (const callback of this._listeners.get("change") ?? []) {
      callback(this);
    }
  }

  // --- getters ---

  viewById(id) {
    return this.views.find((view) => view.id === id) ?? null;
  }

  selectedView() {
    return this.viewById(this.selectedViewId);
  }

  selectedSolve() {
    return this.selectedSolveId ? (this.solveCache.get(this.selectedSolveId) ?? null) : null;
  }

  activePoseKey() {
    return this.selectedPoseKey ?? (this.selectedSolveId ? `solve:${this.selectedSolveId}` : "truth");
  }

  gtByName(name) {
    return this.gtSamples?.find((sample) => sample.name === name) ?? null;
  }

  selectedGtSample() {
    return this.selectedGtName ? this.gtByName(this.selectedGtName) : null;
  }

  gtViewForSample(name) {
    return this.views.find((view) => view.source === "gt" && view.gt_name === name) ?? null;
  }

  // The unified selection: whichever of a placed view or a GT sample is active, normalized to the
  // one view descriptor the map POV, inspector, and markers all consume. This is the dedup: there
  // is a single selected view with poses, not two parallel selection paths.
  selectedViewDescriptor() {
    const view = this.selectedView();
    if (view && view.true_extrinsics) {
      return viewCamera(view, this.selectedSolve());
    }
    const sample = this.selectedGtSample();
    if (sample) {
      return gtCamera(sample);
    }
    return null;
  }

  // --- actions ---

  async init() {
    await this.refreshSceneState();
    this.emitSceneAndViews();
    this.emit("selection");
  }

  async rebuildScene(config) {
    await this.refreshSceneState(await api.setConfig(config));
    this.selectedViewId = null;
    this.selectedSolveId = null;
    this.selectedPoseKey = "truth";
    this.selectedGtName = null;
    this.solveCache.clear();
    this.emitSceneAndViews();
    this.emit("selection");
  }

  async createView(placement) {
    const view = await api.createView(placement);
    this.views = [...this.views, view];
    this.placing = false;
    this.emit("views");
    this.selectView(view.id);
    return view;
  }

  async createPhotoView(file, params) {
    const view = await api.createViewFromPhoto(file, params);
    await this.refreshSceneState();
    this.selectedGtName = null;
    this.selectedPoseKey = "truth";
    this.solveCache.clear();
    this.emitSceneAndViews();
    this.emit("gt");
    this.selectView(view.id);
    return view;
  }

  // Fork any view (placed or GT-derived) under a new name; the copy is selected so you can move it
  // freely while the original stays put.
  async duplicateView(id, label) {
    const view = await api.duplicateView(id, label);
    this.views = [...this.views, view];
    this.emit("views");
    this.selectView(view.id);
    return view;
  }

  async patchView(id, changes) {
    const view = await api.patchView(id, changes);
    this.views = this.views.map((existing) => (existing.id === id ? view : existing));
    if (this.selectedViewId === id && !view.solves.some((solve) => solve.id === this.selectedSolveId)) {
      this.selectedSolveId = null;
      this.selectedPoseKey = "truth";
    }
    this.emit("views");
    this.emit("selection");
    return view;
  }

  async deleteView(id) {
    await api.deleteView(id);
    this.views = this.views.filter((view) => view.id !== id);
    this.emit("views");
    if (this.selectedViewId === id) {
      const nextView = this.views.length ? this.views[this.views.length - 1] : null;
      if (nextView) {
        this.selectView(nextView.id);
      } else {
        this.selectedViewId = null;
        this.selectedSolveId = null;
        this.selectedPoseKey = "truth";
        this.emit("selection");
      }
    } else {
      this.emit("selection");
    }
  }

  selectView(id) {
    this.selectedViewId = id;
    const view = this.viewById(id);
    this.selectedSolveId = view && view.solves.length ? view.solves[view.solves.length - 1].id : null;
    this.selectedPoseKey = this.selectedSolveId ? `solve:${this.selectedSolveId}` : "truth";
    this.placing = false;
    if (this.selectedGtName) {
      this.selectedGtName = null;
      this.emit("gt");
    }
    this.emit("selection");
    // Load the latest solve's full trace into the cache so the map and inspector
    // can show its prediction (summaries in the view list omit the trace).
    if (this.selectedSolveId && !this.solveCache.has(this.selectedSolveId)) {
      const solveId = this.selectedSolveId;
      api
        .getSolve(id, solveId)
        .then((solve) => {
          this.solveCache.set(solve.id, solve);
          if (this.selectedSolveId === solveId) {
            this.emit("selection");
          }
        })
        .catch(() => {});
    }
  }

  selectViewTruth(id = this.selectedViewId) {
    this.selectedViewId = id;
    this.selectedSolveId = null;
    this.selectedPoseKey = "truth";
    this.placing = false;
    if (this.selectedGtName) {
      this.selectedGtName = null;
      this.emit("gt");
    }
    this.emit("selection");
  }

  selectGtTruth(name = this.selectedGtName) {
    this.selectedGtName = name;
    this.selectedViewId = null;
    this.selectedSolveId = null;
    this.selectedPoseKey = "truth";
    this.placing = false;
    this.emit("gt");
    this.emit("selection");
  }

  setPlacing(active) {
    this.placing = active;
    this.emit("placing");
  }

  async loadGtSamples() {
    if (this.gtSamples || this._gtLoading) {
      return;
    }
    this._gtLoading = true;
    try {
      this.gtSamples = await api.listGtSamples();
      this.gtError = null;
    } catch (error) {
      this.gtSamples = [];
      this.gtError = error.message;
    } finally {
      this._gtLoading = false;
    }
    this.emit("gt");
  }

  selectGtSample(name, options = {}) {
    const sample = this.gtByName(name);
    if (options.focus && sample && this.shouldFocusGtSample(sample)) {
      this.selectedGtName = null;
      this.selectedViewId = null;
      this.selectedSolveId = null;
      this.selectedPoseKey = "truth";
      this.placing = false;
      this.gtError = null;
      this.emit("gt");
      this.emit("selection");
      this.focusGtSample(sample, options.extentM).catch((error) => {
        this.gtError = error.message;
        this.emit("gt");
      });
      return;
    }
    this.selectedGtName = name;
    if (name) {
      this.selectedViewId = null;
      this.selectedSolveId = null;
      this.selectedPoseKey = "truth";
      this.placing = false;
    }
    this.gtError = null;
    this.emit("gt");
    this.emit("selection");
  }

  shouldFocusGtSample(sample, toleranceM = 2500) {
    const terrain = this.terrain;
    if (!terrain || terrain.lat_min_deg === undefined || !Number.isFinite(sample.lat) || !Number.isFinite(sample.lon)) {
      return true;
    }
    const centerLat = (terrain.lat_min_deg + terrain.lat_max_deg) / 2;
    const centerLon = (terrain.lon_min_deg + terrain.lon_max_deg) / 2;
    return geoDistanceMeters(sample.lat, sample.lon, centerLat, centerLon) > toleranceM;
  }

  setGtDisplay(changes) {
    this.gtDisplay = { ...this.gtDisplay, ...changes };
    this.emit("gt-display");
  }

  // Opacity of the GT photograph overlaid on the 3D terrain in POV (0 = off), for aligning
  // the DEM render against the photo by eye.
  setPhotoOpacity(value) {
    this.photoOpacity = value;
    this.emit("photo-opacity");
  }

  async focusScene(latDeg, lonDeg, extentM, options = {}) {
    await this.refreshSceneState(await api.focusScene(latDeg, lonDeg, extentM));
    this.selectedViewId = null;
    this.selectedSolveId = null;
    this.selectedPoseKey = "truth";
    this.selectedGtName = options.selectedGtName ?? null;
    this.gtError = null;
    this.solveCache.clear();
    this.emitSceneAndViews();
    this.emit("selection");
    this.emit("gt");
  }

  async focusGtSample(sample, extentM) {
    await this.focusScene(sample.lat, sample.lon, extentM, { selectedGtName: sample.name });
  }

  // Materialize a GT sample as the mutable DEM/refined-pose view backing that image. By default the
  // backing view is not selected in the image list: GT image rows stay immutable and solver outputs
  // appear as view candidates under that image.
  async openGtView(name, options = {}) {
    const select = options.select ?? false;
    const existing = this.gtViewForSample(name);
    if (existing) {
      if (select) {
        this.selectView(existing.id);
      }
      return existing;
    }
    const sample = this.gtByName(name);
    const opensOnCurrentTerrain = this.gtSampleOnCurrentTerrain(sample);
    const view = await api.openGtView(name);
    if (opensOnCurrentTerrain) {
      this.views = [...this.views, view];
      this.emit("views");
    } else {
      await this.refreshSceneState();
      this.solveCache.clear();
      this.emitSceneAndViews();
    }
    if (select) {
      this.selectedGtName = null;
      this.emit("gt");
      this.selectView(view.id);
      return view;
    }
    this.selectedGtName = name;
    this.selectedViewId = null;
    this.emit("gt");
    this.emit("selection");
    return view;
  }

  gtSampleOnCurrentTerrain(sample, marginM = 250) {
    if (!sample || !this.terrain) {
      return false;
    }
    const local = geoToLocal(this.terrain, sample.lat, sample.lon);
    if (!local) {
      return false;
    }
    const eastM = local.east_m + (sample.de_m ?? 0);
    const northM = local.north_m + (sample.dn_m ?? 0);
    return (
      eastM >= this.terrain.x_min_m + marginM &&
      eastM <= this.terrain.x_max_m - marginM &&
      northM >= this.terrain.y_min_m + marginM &&
      northM <= this.terrain.y_max_m - marginM
    );
  }

  async refreshSceneState(scene = undefined) {
    if (scene === undefined) {
      [this.scene, this.terrain, this.peaks, this.views] = await Promise.all([
        api.getScene(),
        api.getTerrain(),
        api.getPeaks(),
        api.listViews(),
      ]);
      return;
    }
    this.scene = scene;
    [this.terrain, this.peaks, this.views] = await Promise.all([api.getTerrain(), api.getPeaks(), api.listViews()]);
  }

  emitSceneAndViews() {
    this.emit("scene");
    this.emit("views");
  }

  setDisplay(changes) {
    this.display = { ...this.display, ...changes };
    this.emit("display");
  }

  async runSolve(viewId, strategy, params) {
    const solve = await api.createSolve(viewId, { strategy, params: params ?? {} });
    this.solveCache.set(solve.id, solve);
    const view = await api.getView(viewId);
    this.views = this.views.map((existing) => (existing.id === viewId ? view : existing));
    this.selectedSolveId = solve.id;
    this.selectedPoseKey = `solve:${solve.id}`;
    this.emit("views");
    this.emit("selection");
    this.emit("pose-selection");
    return solve;
  }

  async selectSolve(viewId, solveId) {
    this.selectedSolveId = solveId;
    this.selectedPoseKey = solveId ? `solve:${solveId}` : "truth";
    this.emit("selection");
    if (solveId && !this.solveCache.has(solveId)) {
      this.solveCache.set(solveId, await api.getSolve(viewId, solveId));
      this.emit("selection");
    }
  }

  async deleteSolve(viewId, solveId) {
    await api.deleteSolve(viewId, solveId);
    this.solveCache.delete(solveId);
    const view = await api.getView(viewId);
    this.views = this.views.map((existing) => (existing.id === viewId ? view : existing));
    if (this.selectedSolveId === solveId) {
      this.selectedSolveId = view.solves.length ? view.solves[view.solves.length - 1].id : null;
      this.selectedPoseKey = this.selectedSolveId ? `solve:${this.selectedSolveId}` : "truth";
    }
    this.emit("views");
    this.emit("selection");
  }

  async selectPose(targetInfo, poseKey) {
    this.selectedPoseKey = poseKey;
    if (poseKey === "gt-depth") {
      this.selectedSolveId = null;
      if (targetInfo?.kind === "gt") {
        this.selectedGtName = targetInfo.sample.name;
        this.selectedViewId = null;
      }
      this.emit("selection");
      this.emit("pose-selection");
      return;
    }
    if (poseKey === "truth") {
      if (targetInfo?.kind === "gt") {
        this.selectGtTruth(targetInfo.sample.name);
      } else if (targetInfo?.view) {
        this.selectViewTruth(targetInfo.view.id);
      } else {
        this.selectedSolveId = null;
        this.emit("selection");
      }
      this.emit("pose-selection");
      return;
    }
    if (poseKey?.startsWith("solve:")) {
      const solveId = poseKey.slice("solve:".length);
      const view = targetInfo?.kind === "gt" ? this.gtViewForSample(targetInfo.sample.name) : targetInfo?.view;
      if (view) {
        const selected = this.selectSolve(view.id, solveId);
        this.emit("pose-selection");
        await selected;
        this.emit("pose-selection");
      }
    }
  }
}

export const store = new Store();

"use strict";

// Client-side mirror of the server scene plus a tiny pub/sub. Panels subscribe
// to change events and read state through getters; all mutations go through the
// async actions, which call the API and then refresh local state.

import { api } from "./api.js";

class Store {
  constructor() {
    this.scene = null;
    this.terrain = null;
    this.peaks = [];
    this.views = [];
    this.selectedViewId = null;
    this.selectedSolveId = null;
    this.placing = false;
    this.solveCache = new Map();
    // Client-side map appearance, applied live by the viewer (never round-trips
    // to the server). `shadingMode` is one of terrain-mesh.js' SHADING_MODES ids.
    this.display = { shadingMode: "relief", contours: true };
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

  // --- actions ---

  async init() {
    const [scene, terrain, peaks, views] = await Promise.all([
      api.getScene(),
      api.getTerrain(),
      api.getPeaks(),
      api.listViews(),
    ]);
    this.scene = scene;
    this.terrain = terrain;
    this.peaks = peaks;
    this.views = views;
    this.emit("scene");
    this.emit("views");
    this.emit("selection");
  }

  async rebuildScene(config) {
    this.scene = await api.setConfig(config);
    [this.terrain, this.peaks, this.views] = await Promise.all([api.getTerrain(), api.getPeaks(), api.listViews()]);
    this.selectedViewId = null;
    this.selectedSolveId = null;
    this.solveCache.clear();
    this.emit("scene");
    this.emit("views");
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

  async patchView(id, changes) {
    const view = await api.patchView(id, changes);
    this.views = this.views.map((existing) => (existing.id === id ? view : existing));
    if (this.selectedViewId === id && !view.solves.some((solve) => solve.id === this.selectedSolveId)) {
      this.selectedSolveId = null;
    }
    this.emit("views");
    this.emit("selection");
    return view;
  }

  async deleteView(id) {
    await api.deleteView(id);
    this.views = this.views.filter((view) => view.id !== id);
    if (this.selectedViewId === id) {
      this.selectedViewId = this.views.length ? this.views[this.views.length - 1].id : null;
      this.selectedSolveId = null;
    }
    this.emit("views");
    this.emit("selection");
  }

  selectView(id) {
    this.selectedViewId = id;
    const view = this.viewById(id);
    this.selectedSolveId = view && view.solves.length ? view.solves[view.solves.length - 1].id : null;
    this.placing = false;
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

  setPlacing(active) {
    this.placing = active;
    this.emit("placing");
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
    this.emit("views");
    this.emit("selection");
    return solve;
  }

  async selectSolve(viewId, solveId) {
    this.selectedSolveId = solveId;
    if (solveId && !this.solveCache.has(solveId)) {
      this.solveCache.set(solveId, await api.getSolve(viewId, solveId));
    }
    this.emit("selection");
  }

  async deleteSolve(viewId, solveId) {
    await api.deleteSolve(viewId, solveId);
    this.solveCache.delete(solveId);
    const view = await api.getView(viewId);
    this.views = this.views.map((existing) => (existing.id === viewId ? view : existing));
    if (this.selectedSolveId === solveId) {
      this.selectedSolveId = view.solves.length ? view.solves[view.solves.length - 1].id : null;
    }
    this.emit("views");
    this.emit("selection");
  }
}

export const store = new Store();

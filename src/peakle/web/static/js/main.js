"use strict";

// Entry point: build the dockview layout, wire each panel to its host element,
// then load the scene. Panels subscribe to the store in their setup, so they are
// wired before `store.init()` emits.

import { buildLayout } from "./layout.js";
import { store } from "./store.js";
import { setupCameraPanel } from "./panels/camera-image.js";
import { setupConfigPanel } from "./panels/config.js";
import { setupMapPanel } from "./panels/map/viewer.js";
import { setupMinimapPanel } from "./panels/minimap.js";
import { setupSolvePanel } from "./panels/solve.js";
import { setupViewsPanel } from "./panels/views.js";

const PANEL_SETUP = {
  map: setupMapPanel,
  config: setupConfigPanel,
  views: setupViewsPanel,
  overview: setupMinimapPanel,
  camera: setupCameraPanel,
  solve: setupSolvePanel,
};

main().catch((error) => {
  const root = document.getElementById("layoutRoot");
  root.innerHTML = `<div class="boot-error">Failed to load workbench: ${error.message}</div>`;
  // eslint-disable-next-line no-console
  console.error(error);
});

async function main() {
  setupLoadingIndicator();
  const layout = buildLayout(document.getElementById("layoutRoot"));
  for (const [name, setup] of Object.entries(PANEL_SETUP)) {
    wirePanel(layout, name, setup);
  }
  // Dev console handle: inspect state and dispatch actions from the browser.
  window.peakle = { store };
  await store.init();
}

function setupLoadingIndicator() {
  const label = document.createElement("span");
  label.textContent = "Loading...";
  const spinner = document.createElement("span");
  spinner.className = "app-loading-spinner";
  const indicator = document.createElement("div");
  indicator.className = "app-loading";
  indicator.setAttribute("role", "status");
  indicator.setAttribute("aria-live", "polite");
  indicator.hidden = true;
  indicator.append(spinner, label);
  document.body.append(indicator);
  store.on("loading", () => {
    indicator.hidden = !store.loading.active;
    label.textContent = store.loading.message || "Loading...";
  });
}

function wirePanel(layout, name, setup) {
  const rootId = `${name}Panel`;
  let mounted = false;
  const tryMount = () => {
    if (mounted) {
      return;
    }
    const element = document.getElementById(rootId);
    if (!element) {
      return;
    }
    mounted = true;
    setup(store, element);
  };
  layout.onDidLayoutChange(tryMount);
  layout.onDidActivePanelChange(tryMount);
  tryMount();
}

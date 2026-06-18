"use strict";

// Dockview layout: a large 3D map on the left and a stacked control column on the
// right (config, views, camera image, solve inspector). Every panel host gets a
// stable id so the panel modules can find and populate it.

import { createDockview, themeAbyss } from "dockview-core";

const PANELS = [
  { id: "map", title: "Map" },
  { id: "config", title: "Setup" },
  { id: "views", title: "Views" },
  { id: "camera", title: "Camera" },
  { id: "solve", title: "Solve" },
];

export function buildLayout(rootElement) {
  const dockview = createDockview(rootElement, {
    theme: themeAbyss,
    // Keep every panel mounted so the renderers/handlers stay alive when not focused.
    defaultRenderer: "always",
    createComponent: (options) => {
      const element = document.createElement("div");
      element.className = "panel-host panel";
      element.id = `${options.name}Panel`;
      return { element, init: () => {} };
    },
  });

  const bounds = rootElement.getBoundingClientRect();
  const width = Math.max(960, bounds.width);
  const height = Math.max(560, bounds.height);
  const panel = (id) => ({ id, contentComponent: id, title: PANELS.find((p) => p.id === id).title });
  const leaf = (size, ...panels) => ({
    type: "leaf",
    size,
    data: { views: panels.map((p) => p.id), activeView: panels[0].id, id: `group-${panels[0].id}` },
    panels,
  });

  const groups = [
    leaf(Math.round(width * 0.6), panel("map")),
    {
      type: "branch",
      size: Math.round(width * 0.4),
      data: [
        leaf(Math.round(height * 0.2), panel("config")),
        leaf(Math.round(height * 0.28), panel("views")),
        leaf(Math.round(height * 0.26), panel("camera")),
        leaf(Math.round(height * 0.26), panel("solve")),
      ],
    },
  ];

  dockview.fromJSON({
    grid: { orientation: "HORIZONTAL", width, height, root: { type: "branch", size: height, data: groups } },
    panels: Object.fromEntries(groups.flatMap(collectLeafPanels).map((entry) => [entry.id, entry])),
    activeGroup: "group-map",
  });
  return dockview;
}

function collectLeafPanels(node) {
  return node.type === "leaf" ? node.panels : node.data.flatMap(collectLeafPanels);
}

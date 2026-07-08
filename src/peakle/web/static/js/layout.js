"use strict";

// Dockview layout: side columns carry workflow/inspection panels and the 3D map
// owns the center. Every panel host gets a stable id so the panel modules can
// find and populate it. There is one Views list: placed cameras and GT samples
// are just views from different providers.

import { createDockview, themeAbyss } from "dockview-core";

const PANELS = [
  { id: "map", title: "Map" },
  { id: "config", title: "Setup" },
  { id: "views", title: "Views" },
  { id: "overview", title: "Overview" },
  { id: "camera", title: "Inspect" },
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

  // The map is the app: give it the middle and keep workflow panels in side
  // columns. The side rails stay readable but bounded so wide screens keep the
  // terrain as the dominant surface.
  const minMapWidth = 420;
  let leftWidth = Math.min(360, Math.max(300, Math.round(width * 0.2)));
  let rightWidth = Math.min(420, Math.max(340, Math.round(width * 0.22)));
  const overflow = leftWidth + rightWidth + minMapWidth - width;
  if (overflow > 0) {
    const totalSideWidth = leftWidth + rightWidth;
    const leftShrink = Math.round(overflow * (leftWidth / totalSideWidth));
    leftWidth = Math.max(260, leftWidth - leftShrink);
    rightWidth = Math.max(280, rightWidth - (overflow - leftShrink));
  }
  const mapWidth = width - leftWidth - rightWidth;
  const groups = [
    {
      type: "branch",
      size: leftWidth,
      data: [
        leaf(Math.round(height * 0.6), panel("views"), panel("config")),
        leaf(Math.round(height * 0.4), panel("overview")),
      ],
    },
    leaf(mapWidth, panel("map")),
    {
      type: "branch",
      size: rightWidth,
      data: [leaf(Math.round(height * 0.58), panel("camera")), leaf(Math.round(height * 0.42), panel("solve"))],
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

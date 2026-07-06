"use strict";

// Dockview layout: the 3D map dominates; a narrow right rail carries two tabbed
// groups — Views | GT data | Setup on top, Inspect | Solve below. Every panel
// host gets a stable id so the panel modules can find and populate it.

import { createDockview, themeAbyss } from "dockview-core";

const PANELS = [
  { id: "map", title: "Map" },
  { id: "config", title: "Setup" },
  { id: "views", title: "Views" },
  { id: "gt", title: "GT data" },
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

  // The map is the app: give it ~72% of the width. The right rail stacks the
  // work tabs (Views | GT data | Setup — Setup is rarely touched, so it rides
  // last) over the inspector tabs (Inspect | Solve).
  const groups = [
    leaf(Math.round(width * 0.72), panel("map")),
    {
      type: "branch",
      size: Math.round(width * 0.28),
      data: [
        leaf(Math.round(height * 0.52), panel("views"), panel("gt"), panel("config")),
        leaf(Math.round(height * 0.48), panel("camera"), panel("solve")),
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

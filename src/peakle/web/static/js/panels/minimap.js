"use strict";

import { el } from "../format.js";
import { setupMinimap } from "./map/minimap.js";

export function setupMinimapPanel(store, root) {
  root.classList.add("minimap-panel");
  const mapHost = el("div", { class: "minimap minimap-panel-map" });
  root.replaceChildren(mapHost);

  let minimap = null;
  const init = () => {
    if (!minimap && mapHost.clientWidth > 0 && mapHost.clientHeight > 0) {
      minimap = setupMinimap(store, mapHost);
    }
    minimap?.invalidate();
  };

  const resizeObserver = new ResizeObserver(() => requestAnimationFrame(init));
  resizeObserver.observe(root);
  requestAnimationFrame(init);
}

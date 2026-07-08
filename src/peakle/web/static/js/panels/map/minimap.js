"use strict";

// 2D overview minimap (MapLibre GL JS + OpenTopoMap raster).
// Purpose: move the heightmap anywhere. It shows the current terrain window as a
// rectangle and every GT sample as a circle (one GeoJSON layer, not N DOM nodes,
// so 364 points stay smooth), and recenters the 3D heightmap on click — map
// background focuses that point; a GT circle selects + focuses that sample. The
// 3D heightmap stays the primary work surface; this is the navigator.

// Raster style (no API key): OpenTopoMap gives topographic context that suits terrain work.
const STYLE = {
  version: 8,
  sources: {
    otm: {
      type: "raster",
      tiles: ["https://a.tile.opentopomap.org/{z}/{x}/{y}.png", "https://b.tile.opentopomap.org/{z}/{x}/{y}.png"],
      tileSize: 256,
      maxzoom: 17,
      attribution: "© OpenTopoMap (CC-BY-SA)",
    },
  },
  layers: [{ id: "otm", type: "raster", source: "otm" }],
};

const EMPTY_FC = { type: "FeatureCollection", features: [] };

export function setupMinimap(store, host) {
  if (typeof window.maplibregl === "undefined") {
    host.remove(); // CDN blocked — degrade silently, the 3D map still works
    return null;
  }
  const maplibregl = window.maplibregl;
  const map = new maplibregl.Map({
    container: host,
    style: STYLE,
    center: [8.2, 46.8],
    zoom: 7,
    attributionControl: false,
  });
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");

  function gtFeatures() {
    return (store.gtSamples ?? [])
      .filter((s) => Number.isFinite(s.lat) && Number.isFinite(s.lon))
      .map((s) => ({
        type: "Feature",
        geometry: { type: "Point", coordinates: [s.lon, s.lat] },
        properties: { name: s.name, clean: s.quality === "CLEAN" ? 1 : 0, sel: s.name === store.selectedGtName ? 1 : 0 },
      }));
  }

  function windowFeature() {
    const t = store.terrain;
    if (!t || t.lat_min_deg === undefined) {
      return EMPTY_FC;
    }
    const [w, e, s, n] = [t.lon_min_deg, t.lon_max_deg, t.lat_min_deg, t.lat_max_deg];
    return {
      type: "FeatureCollection",
      features: [
        {
          type: "Feature",
          geometry: { type: "Polygon", coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] },
          properties: {},
        },
      ],
    };
  }

  function refreshGt() {
    map.getSource("gt")?.setData({ type: "FeatureCollection", features: gtFeatures() });
  }

  function refreshWindow(fit) {
    const fc = windowFeature();
    map.getSource("window")?.setData(fc);
    const t = store.terrain;
    if (fit && t && t.lat_min_deg !== undefined) {
      map.fitBounds([[t.lon_min_deg, t.lat_min_deg], [t.lon_max_deg, t.lat_max_deg]], {
        padding: 40,
        maxZoom: 11,
        animate: false,
      });
    }
  }

  map.on("load", () => {
    map.addSource("window", { type: "geojson", data: windowFeature() });
    map.addLayer({
      id: "window-fill",
      type: "fill",
      source: "window",
      paint: { "fill-color": "#f0c75e", "fill-opacity": 0.06 },
    });
    map.addLayer({
      id: "window-line",
      type: "line",
      source: "window",
      paint: { "line-color": "#f0c75e", "line-width": 2 },
    });

    map.addSource("gt", { type: "geojson", data: { type: "FeatureCollection", features: gtFeatures() } });
    map.addLayer({
      id: "gt-circles",
      type: "circle",
      source: "gt",
      paint: {
        "circle-radius": ["case", ["==", ["get", "sel"], 1], 7, 4],
        "circle-color": ["case", ["==", ["get", "sel"], 1], "#ffd24a", ["==", ["get", "clean"], 1], "#8ee08e", "#d07a6a"],
        "circle-stroke-width": 1,
        "circle-stroke-color": "#141512",
      },
    });
    refreshWindow(true);

    // Circle click: select + focus that sample. Background click: focus the point.
    map.on("click", "gt-circles", (event) => {
      const name = event.features?.[0]?.properties?.name;
      if (name) {
        const s = store.gtByName(name);
        if (s) {
          store.focusGtSample(s).catch(() => {});
        }
      }
    });
    map.on("click", (event) => {
      const hits = map.queryRenderedFeatures(event.point, { layers: ["gt-circles"] });
      if (!hits.length) {
        store.focusScene(event.lngLat.lat, event.lngLat.lng).catch(() => {});
      }
    });
    map.on("mouseenter", "gt-circles", () => {
      map.getCanvas().style.cursor = "pointer";
    });
    map.on("mouseleave", "gt-circles", () => {
      map.getCanvas().style.cursor = "";
    });
  });

  store.on("scene", () => refreshWindow(true));
  store.on("gt", refreshGt);

  return { map, invalidate: () => map.resize() };
}

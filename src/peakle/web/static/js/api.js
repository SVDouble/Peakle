"use strict";

// The only module that talks to the backend. Every panel goes through here.

async function request(method, path, body) {
  const options = { method, headers: {} };
  if (body !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  const response = await fetch(path, options);
  return responseJson(response, method, path);
}

async function responseJson(response, method, path) {
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      if (payload && payload.detail) {
        detail = typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail);
      }
    } catch {
      // non-JSON error body; keep the status text
    }
    throw new Error(`${method} ${path} failed: ${detail}`);
  }
  if (response.status === 204) {
    return null;
  }
  return response.json();
}

async function uploadPhoto(path, file) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": file.type || "application/octet-stream" },
    body: file,
  });
  return responseJson(response, "POST", path);
}

export const api = {
  getScene: () => request("GET", "/api/scene"),
  setConfig: (config) => request("PUT", "/api/scene/config", config),
  focusScene: (latDeg, lonDeg, extentM) =>
    request("PUT", "/api/scene/focus", { lat_deg: latDeg, lon_deg: lonDeg, ...(extentM ? { extent_m: extentM } : {}) }),
  getTerrain: () => request("GET", "/api/terrain"),
  getPeaks: () => request("GET", "/api/peaks"),
  getAlgorithms: () => request("GET", "/api/algorithms"),
  listJobs: (kind = undefined) => request("GET", `/api/jobs${kind ? `?kind=${encodeURIComponent(kind)}` : ""}`),
  getJob: (id) => request("GET", `/api/jobs/${encodeURIComponent(id)}`),
  createJob: (body) => request("POST", "/api/jobs", body),

  listViews: () => request("GET", "/api/views"),
  getView: (id) => request("GET", `/api/views/${id}`),
  createView: (placement) => request("POST", "/api/views", placement),
  createViewFromPhoto: (file, params) => {
    const query = new URLSearchParams({
      lat_deg: String(params.lat_deg),
      lon_deg: String(params.lon_deg),
      horizontal_fov_deg: String(params.horizontal_fov_deg),
      eye_height_m: String(params.eye_height_m ?? 2.0),
    });
    if (params.label) {
      query.set("label", params.label);
    }
    return uploadPhoto(`/api/views/from-photo?${query}`, file);
  },
  patchView: (id, changes) => request("PATCH", `/api/views/${id}`, changes),
  deleteView: (id) => request("DELETE", `/api/views/${id}`),
  duplicateView: (id, label) => request("POST", `/api/views/${id}/duplicate`, label ? { label } : {}),
  viewImageUrl: (id) => `/api/views/${id}/image`,
  viewPhotoUrl: (id) => `/api/views/${id}/photo`,

  listGtSamples: () => request("GET", "/api/gt/samples"),
  getGtAlignmentAudit: (limit = 100, includeClean = false, metric = false) =>
    request(
      "GET",
      `/api/gt/alignment-audit?limit=${encodeURIComponent(limit)}&include_clean=${includeClean ? "true" : "false"}&metric=${metric ? "true" : "false"}`,
    ),
  gtThumbUrl: (name) => `/api/gt/samples/${encodeURIComponent(name)}/thumb.jpg`,
  gtLayerUrl: (name, layer) => `/api/gt/samples/${encodeURIComponent(name)}/layers/${layer}.png`,
  openGtView: (name) => request("POST", `/api/gt/samples/${encodeURIComponent(name)}/open-view`),

  listSolves: (viewId) => request("GET", `/api/views/${viewId}/solves`),
  getSolve: (viewId, solveId) => request("GET", `/api/views/${viewId}/solves/${solveId}`),
  createSolve: (viewId, body) => request("POST", `/api/views/${viewId}/solves`, body),
  deleteSolve: (viewId, solveId) => request("DELETE", `/api/views/${viewId}/solves/${solveId}`),
};

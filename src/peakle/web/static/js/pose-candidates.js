"use strict";

import { formatNumber } from "./format.js";

export function selectedPoseTarget(store) {
  const view = store.selectedView();
  if (view?.true_extrinsics) {
    return { kind: "view", label: view.label, view };
  }
  const sample = store.selectedGtSample?.() ?? (store.selectedGtName ? store.gtByName(store.selectedGtName) : null);
  if (sample) {
    return { kind: "gt", label: sample.name, sample, view: store.gtViewForSample?.(sample.name) ?? null };
  }
  return null;
}

export function poseTargetKey(targetInfo) {
  if (!targetInfo) {
    return "";
  }
  return targetInfo.kind === "view" ? `view:${targetInfo.view.id}` : `gt:${targetInfo.sample.name}`;
}

export function poseCandidates(store, targetInfo = selectedPoseTarget(store)) {
  if (targetInfo?.kind === "gt") {
    const backingView = targetInfo.view ?? store.gtViewForSample?.(targetInfo.sample.name) ?? null;
    const candidates = [groundTruthCandidate(targetInfo.sample)];
    candidates.push(...(backingView?.solves ?? []).map((summary) => solveCandidate(store, backingView, summary)));
    return candidates;
  }
  const view = targetInfo?.view;
  if (!view) {
    return [];
  }
  const sample = view.gt_name ? store.gtByName(view.gt_name) : null;
  const candidates = [];
  if (sample) {
    candidates.push(groundTruthCandidate(sample));
  }
  if (view.true_extrinsics && view.source !== "gt") {
    candidates.push(viewCandidate(store, view, sample));
  }
  candidates.push(...view.solves.map((summary) => solveCandidate(store, view, summary)));
  return candidates;
}

export function candidatePrimaryKey(store) {
  return store.activePoseKey?.() ?? (store.selectedSolveId ? `solve:${store.selectedSolveId}` : "truth");
}

export function defaultCompareKeys(candidates, primaryKey = "truth") {
  if (candidates.length < 2) {
    return [];
  }
  const primary = candidates.find((candidate) => candidate.key === primaryKey) ?? candidates.find(hasMapFit) ?? candidates[0];
  const secondary =
    candidates.find((candidate) => candidate.key !== primary.key && hasMapFit(candidate) && hasMapFit(primary)) ??
    candidates.find((candidate) => candidate.key !== primary.key && hasMapFit(candidate)) ??
    candidates.find((candidate) => candidate.key !== primary.key);
  return [primary, secondary].filter(Boolean).map((candidate) => candidate.key);
}

export function comparisonText(primary, other) {
  const parts = [metricDelta("map fit", primary.metrics.fit, other.metrics.fit, "px")].filter(Boolean);
  return parts.length ? `${primary.label} vs ${other.label}: ${parts.join(" · ")}` : `${primary.label} vs ${other.label}: no comparable metrics yet.`;
}

export function strategyLabel(store, name) {
  return store.scene?.strategies.find((s) => s.name === name)?.label ?? name;
}

function viewCandidate(store, view, sample = view.gt_name ? store.gtByName(view.gt_name) : null) {
  const label = view.source === "photo" ? "Photo metadata pose" : "Dataset pose";
  return {
    key: "truth",
    kind: "truth",
    label,
    stat: `yaw ${formatNumber(view.true_extrinsics.yaw_deg, "deg")} · pitch ${formatNumber(view.true_extrinsics.pitch_deg, "deg")}`,
    metricText: "map fit -",
    metrics: { fit: null },
  };
}

function groundTruthCandidate(sample) {
  return {
    key: "gt-depth",
    kind: "ground_truth",
    label: "Original GT pose + source depth",
    stat: "PFM skyline reference",
    metricText: "map fit -",
    metrics: { fit: null },
  };
}

function solveCandidate(store, view, summary) {
  const fit = summary.metrics.contour_mae_px;
  const evidence = summary.evidence_source?.replaceAll("_", " ") ?? "unknown evidence";
  return {
    key: `solve:${summary.id}`,
    kind: "solve",
    id: summary.id,
    label: strategyLabel(store, summary.strategy),
    stat: `Solver pose · ${evidence}`,
    metricText: mapFitText(fit),
    metrics: {
      fit,
    },
  };
}

function mapFitText(value) {
  return `map fit ${formatNumber(value, "px")}`;
}

function hasMapFit(candidate) {
  const fit = candidate.metrics.fit;
  return fit !== null && fit !== undefined && !Number.isNaN(fit);
}

function metricDelta(label, a, b, unit) {
  if (a === null || a === undefined || b === null || b === undefined || Number.isNaN(a) || Number.isNaN(b)) {
    return "";
  }
  return `delta ${label} ${formatNumber(a - b, unit)}`;
}

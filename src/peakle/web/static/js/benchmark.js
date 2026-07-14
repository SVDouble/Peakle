"use strict";

const runSelect = document.querySelector("#runSelect");
const subsetTabs = document.querySelector("#subsetTabs");
const search = document.querySelector("#search");
const cards = document.querySelector("#cards");
const leaderboard = document.querySelector("#leaderboard");
const leaderboardHead = document.querySelector("#leaderboardHead");
const strata = document.querySelector("#strata");
const strataTitle = document.querySelector("#strataTitle");
const strataNote = document.querySelector("#strataNote");
const casesBody = document.querySelector("#cases");
const casesHead = document.querySelector("#casesHead");
const warning = document.querySelector("#warning");
const provenance = document.querySelector("#provenance");
const subsetCount = document.querySelector("#subsetCount");
const caseCount = document.querySelector("#caseCount");

const LABELS = {
  all: "All records",
  manual: "Manual labels",
  primary: "Ranking eligible",
  primary_height_a: "Primary clean",
  map_ab: "MAP_A + B",
  map_ab_height_a: "MAP_A + B + height",
  map_a: "MAP_A",
  map_a_height_a: "MAP_A + height",
  map_a_photo: "MAP_A + photo support",
  map_b: "MAP_A + B",
  map_proxy_5px: "Legacy proxy ≤5 px",
  map_proxy_10px: "Legacy proxy ≤10 px",
};

const ALGORITHMS = {
  "keep-prior": "Keep prior",
  horizon: "Horizon",
  contourdb: "Contour DB",
  cmaes: "CMA-ES",
  powell: "Powell",
  nelder: "Nelder–Mead",
  evolution: "Evolution",
  global: "Regional global",
  "render-pnp": "Render match + PnP",
  "skyline-atlas": "Skyline pose atlas",
};

const PRIORS = {
  raw_metadata: "Exact reference · retention",
  perturbed_metadata: "Perturbed metadata",
  position_only: "Position only",
  none: "Regional / no pose prior",
};

const EVIDENCE = {
  pfm_oracle: "PFM / source depth",
  photo_auto: "Automatic photo",
  photo_rgb: "Photo RGB",
};

let runs = [];
let summary = null;
let selectedSubset = "all";
let queryTimer = 0;
let loadGeneration = 0;
let summaryController = null;
let casesController = null;

function node(tag, attrs = {}, children = []) {
  const element = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") element.className = value;
    else if (key === "text") element.textContent = value;
    else element.setAttribute(key, value);
  }
  element.append(...children);
  return element;
}

function pct(value) {
  return value == null ? "—" : `${Math.round(100 * Number(value))}%`;
}

function num(value, digits = 1, suffix = "") {
  return value == null ? "—" : `${Number(value).toFixed(digits)}${suffix}`;
}

function signed(value, digits = 1, suffix = "") {
  if (value == null) return "—";
  const number = Number(value);
  return `${number > 0 ? "+" : ""}${number.toFixed(digits)}${suffix}`;
}

function originalMetadataAmbiguity(item) {
  const diagnostic = item.original_metadata_diagnostic;
  const delta = diagnostic?.available ? diagnostic.refined_minus_original : null;
  if (!delta) return null;
  return `non-ranking original Δ · ${num(delta.horizontal_position_m, 0, " m")} horiz · ${signed(delta.vertical_m, 0, " m")} vert · ${signed(delta.fov_deg, 1, "°")} FOV`;
}

function candidateValidationSummary(item) {
  const validation = item.candidate_validation;
  if (!validation) return null;
  if (validation.enabled === false) return "candidate holdout disabled · ablation";
  if (validation.passed === true) return "candidate holdout passed";
  const failures = Array.isArray(validation.failures) ? validation.failures.join(", ") : "gate failed";
  return `candidate holdout rejected · ${failures}`;
}

async function json(url, signal) {
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
  return response.json();
}

function renderRuns() {
  runSelect.replaceChildren(...runs.map((run) => {
    const kind = run.kind === "strategy_matrix"
      ? `${run.algorithm_count} strategies`
      : run.kind === "pose_atlas"
        ? `${run.track_count} atlas evidence tracks`
        : "legacy horizon";
    const status = run.error_count ? ` · ${run.error_count} ${run.kind === "pose_atlas" ? "track" : "sample"} errors` : "";
    return node("option", { value: run.id, text: `${run.created_at} · ${run.sample_count} samples · ${kind}${status}` });
  }));
  const preferred = runs.find((run) => run.recommended) ?? runs[0];
  if (preferred) runSelect.value = preferred.id;
}

function renderTabs() {
  const available = summary?.subsets ?? {};
  if (!(selectedSubset in available)) selectedSubset = summary?.default_subset ?? "all";
  subsetTabs.replaceChildren(...Object.keys(available).map((key) => {
    const label = LABELS[key] ?? key;
    const button = node("button", {
      type: "button",
      class: key === selectedSubset ? "active" : "",
      text: label,
      "aria-pressed": key === selectedSubset ? "true" : "false",
    });
    button.addEventListener("click", () => {
      selectedSubset = key;
      renderTabs();
      renderSummary();
      void loadCases(loadGeneration).catch(showError);
    });
    return button;
  }));
}

function renderProvenance() {
  const run = summary.run;
  const pills = run.kind === "pose_atlas"
    ? [
      ["Artifact", "pose-atlas study v2"],
      ["Samples", `${run.completed_sample_count}/${run.sample_count}`],
      ["Evidence tracks", run.track_count],
      ["Completed tracks", `${run.completed_track_count}/${run.attempted_case_count}`],
      ["Status", run.status],
      ["Compatibility", run.compatibility_policy],
    ]
    : [
      ["Artifact", run.kind === "strategy_matrix" ? `schema v${run.schema_version} matrix` : "legacy orientation"],
      ["Samples", `${run.completed_sample_count}/${run.sample_count}`],
      ["Strategies", run.algorithm_count],
      ["Attempted cells", run.attempted_case_count || "legacy"],
      ["Status", run.status],
      ["Compatibility", run.compatibility_policy],
    ];
  if (run.results_sha256) pills.push(["Artifact SHA", run.results_sha256.slice(0, 12)]);
  if (run.implementation_sha256) pills.push(["Implementation", run.implementation_sha256.slice(0, 12)]);
  if (run.git_sha) pills.push(["Git", `${run.git_sha.slice(0, 9)}${run.dirty_code ? "+dirty" : ""}`]);
  if (run.git_diff_sha256) pills.push(["Diff", run.git_diff_sha256.slice(0, 12)]);
  if (run.candidate_validation) pills.push(["Candidate gate", run.candidate_validation.enabled === false ? "disabled ablation" : "held-out + visibility"]);
  provenance.replaceChildren(...pills.map(([label, value]) => node("span", { class: "pill" }, [
    node("span", { text: `${label} ` }), node("strong", { text: String(value) }),
  ])));
  if (run.dirty_code) provenance.append(node("span", { class: "pill warn", text: "Dirty source snapshot" }));
  if (run.hash_verified === false) provenance.append(node("span", { class: "pill bad", text: "Artifact hash mismatch" }));
  if (!run.has_provenance) provenance.append(node("span", { class: "pill warn", text: "Provenance unavailable" }));
  const messages = summary.warnings ?? (summary.warning ? [summary.warning] : []);
  warning.hidden = messages.length === 0;
  warning.replaceChildren(...messages.map((message) => node("p", { text: message })));
}

function card(label, value, detail, className = "") {
  return node("article", { class: `card ${className}` }, [
    node("span", { class: "card-label", text: label }),
    node("strong", { class: "card-value number", text: value }),
    node("span", { class: "card-detail", text: detail }),
  ]);
}

function renderSummary() {
  if (summary.mode === "matrix") renderMatrixSummary();
  else if (summary.mode === "atlas") renderAtlasSummary();
  else renderLegacySummary();
}

function renderMatrixSummary() {
  const subset = summary.subsets[selectedSubset] ?? summary.subsets.all;
  const aggregates = subset.aggregates.filter((row) => row.attempted > 0);
  const eligible = aggregates.filter((row) => row.primary_attempts > 0);
  const errors = aggregates.reduce((total, row) => total + row.errors, 0);
  const abstentions = aggregates.reduce((total, row) => total + (row.abstained ?? 0), 0);
  const evidenceRejections = aggregates.reduce((total, row) => total + (row.evidence_rejected ?? 0), 0);
  const primaryAttempts = aggregates.reduce((total, row) => total + row.primary_attempts, 0);
  const primarySuccesses = eligible.reduce((total, row) => total + (row.primary_successes ?? 0), 0);
  const improvements = aggregates.reduce((total, row) => total + (row.improved_over_prior ?? 0), 0);
  const regressions = aggregates.reduce((total, row) => total + (row.regressed_from_prior ?? 0), 0);
  const yawImprovements = aggregates.reduce((total, row) => total + (row.yaw_improved_over_prior ?? 0), 0);
  const positionImprovements = aggregates.reduce((total, row) => total + (row.position_improved_over_prior ?? 0), 0);
  subsetCount.textContent = `${subset.sample_count} samples · ${subset.attempted_case_count} attempted · ${abstentions} abstained`;
  cards.replaceChildren(
    card("Ranking-eligible cells", String(primaryAttempts), "Manual MAP_A/B after recorded exclusions", "good"),
    card(
      "Joint successes",
      `${primarySuccesses} / ${primaryAttempts}`,
      "Position ≤100 m and yaw ≤5°; strategies are compared only within their evidence/prior condition",
      primarySuccesses ? "photo" : "bad",
    ),
    card(
      "Abstentions · errors",
      `${abstentions} · ${errors}`,
      `${evidenceRejections} evidence rejections; every non-success remains in the attempted denominator`,
      errors ? "bad" : abstentions ? "photo" : "",
    ),
    card(
      "Paired improvements",
      `${positionImprovements} pos · ${yawImprovements} yaw`,
      `${improvements} full-pose success flips · ${regressions} regressions against the identical prior`,
    ),
  );
  renderMatrixLeaderboard(aggregates);
  renderMatrixStrata();
}

function renderMatrixLeaderboard(aggregates) {
  const headers = ["Algorithm", "Evidence", "Prior", "Attempts", "Joint success", "Primary joint", "Position ≤100 m", "Yaw ≤5°", "Median position", "Median yaw", "Δ position", "Δ yaw", "Abstained", "Evidence rejected", "Errors", "Runtime"];
  leaderboardHead.replaceChildren(node("tr", {}, headers.map((label) => node("th", { text: label }))));
  const priorOrder = Object.keys(PRIORS);
  const evidenceOrder = Object.keys(EVIDENCE);
  const sorted = [...aggregates].sort((a, b) =>
    priorOrder.indexOf(a.prior_regime) - priorOrder.indexOf(b.prior_regime)
    || evidenceOrder.indexOf(a.evidence_track) - evidenceOrder.indexOf(b.evidence_track)
    || (b.primary_attempts > 0) - (a.primary_attempts > 0)
    || (b.primary_success_rate ?? -1) - (a.primary_success_rate ?? -1)
    || String(a.algorithm).localeCompare(String(b.algorithm))
  );
  leaderboard.replaceChildren(...sorted.map((row) => {
    const deltaClass = row.median_position_delta_vs_prior_m == null ? "" : row.median_position_delta_vs_prior_m <= 0 ? "success" : "failure";
    return node("tr", {}, [
      node("td", {}, [
        node("span", { class: "method", text: ALGORITHMS[row.algorithm] ?? row.algorithm }),
        node("span", {
          class: "sub",
          text: ["global", "render-pnp"].includes(row.algorithm)
            ? "experimental diagnostic; inspect recorded exclusions"
            : "compare within this evidence/prior condition",
        }),
      ]),
      node("td", { text: EVIDENCE[row.evidence_track] ?? row.evidence_track }),
      node("td", { text: PRIORS[row.prior_regime] ?? row.prior_regime }),
      node("td", { class: "number", text: String(row.attempted) }),
      node("td", { class: "number", text: pct(row.success_rate) }),
      node("td", { class: `number ${row.primary_attempts ? "success" : "muted"}`, text: row.primary_attempts ? `${pct(row.primary_success_rate)} · n=${row.primary_attempts}` : "excluded" }),
      node("td", { class: "number", text: pct(row.position_success_rate) }),
      node("td", { class: "number", text: pct(row.yaw_success_rate) }),
      node("td", { class: "number", text: num(row.median_horizontal_position_error_m, 1, " m") }),
      node("td", { class: "number", text: num(row.median_absolute_yaw_error_deg, 1, "°") }),
      node("td", { class: `number ${deltaClass}`, text: signed(row.median_position_delta_vs_prior_m, 1, " m") }),
      node("td", { class: `number ${(row.median_yaw_delta_vs_prior_deg ?? 0) <= 0 ? "success" : "failure"}`, text: signed(row.median_yaw_delta_vs_prior_deg, 1, "°") }),
      node("td", { class: `number ${row.abstained ? "failure" : ""}`, text: String(row.abstained ?? 0) }),
      node("td", { class: "number", text: String(row.evidence_rejected ?? 0) }),
      node("td", { class: `number ${row.errors ? "failure" : ""}`, text: String(row.errors) }),
      node("td", { class: "number", text: num(row.runtime_s, 1, " s") }),
    ]);
  }));
  if (!sorted.length) leaderboard.append(emptyRow(headers.length, "No attempted strategy cells in this subset."));
}

function renderMatrixStrata() {
  strataTitle.textContent = "Evaluation strata";
  strataNote.textContent = "MAP and raw-height gates are independent and provisional. Primary rows also require manual references and no strategy-specific exclusion.";
  const entries = Object.entries(summary.subsets);
  strata.replaceChildren(...entries.map(([key, subset]) => {
    const active = key === selectedSubset ? " active" : "";
    return node("button", { type: "button", class: `stratum stratum-button${active}` }, [
      node("div", { class: "stratum-head" }, [
        node("span", { class: "stratum-name", text: LABELS[key] ?? key }),
        node("span", { class: "muted number", text: `n=${subset.sample_count}` }),
      ]),
      node("span", { class: "sub", text: `${subset.attempted_case_count} attempted · ${subset.abstention_count ?? 0} abstained · ${subset.error_count} errors` }),
    ]);
  }));
  [...strata.querySelectorAll("button")].forEach((button, index) => {
    button.addEventListener("click", () => {
      selectedSubset = entries[index][0];
      renderTabs();
      renderSummary();
      void loadCases(loadGeneration).catch(showError);
    });
  });
}

function renderAtlasSummary() {
  const subset = summary.subsets[selectedSubset] ?? summary.subsets.all;
  const aggregates = subset.aggregates;
  const requested = aggregates.reduce((total, row) => total + row.requested, 0);
  const completed = aggregates.reduce((total, row) => total + row.completed, 0);
  const blindSuccesses = aggregates.reduce((total, row) => total + row.blind_winner_successes, 0);
  const oracleSuccesses = aggregates.reduce((total, row) => total + row.full_lattice_oracle_successes, 0);
  const rejected = aggregates.reduce((total, row) => total + row.evidence_rejected, 0);
  const errors = aggregates.reduce((total, row) => total + row.errors, 0);
  const missing = aggregates.reduce((total, row) => total + row.missing, 0);
  subsetCount.textContent = `${subset.sample_count} samples · ${completed}/${requested} completed tracks`;
  cards.replaceChildren(
    card("Completed evidence tracks", `${completed} / ${requested}`, `${rejected} evidence rejections · ${errors} errors · ${missing} missing; all remain in the denominator`, errors || missing ? "bad" : rejected ? "photo" : "good"),
    card("Blind-winner successes", `${blindSuccesses} / ${requested}`, "Estimator rank 1 · position ≤100 m and yaw ≤5°", blindSuccesses ? "photo" : "bad"),
    card("Full-lattice oracle ceiling", `${oracleSuccesses} / ${requested}`, "Evaluation-only reachability after the estimator score lattice was frozen", "good"),
    card("Selection gap", `+${Math.max(0, oracleSuccesses - blindSuccesses)}`, "Reachable targets missed by blind estimator selection; not additional solver wins"),
  );
  renderAtlasLeaderboard(aggregates);
  renderAtlasStrata(aggregates);
}

function renderAtlasLeaderboard(aggregates) {
  const headers = ["Evidence", "Requested", "Completed", "Blind success", "Oracle ceiling", "Blind position", "Oracle position", "Blind yaw", "Oracle yaw", "Oracle score rank", "Evaluation-only GT reach in top-100", "Runtime"];
  leaderboardHead.replaceChildren(node("tr", {}, headers.map((label) => node("th", { text: label }))));
  leaderboard.replaceChildren(...aggregates.map((row) => node("tr", {}, [
    node("td", {}, [
      node("span", { class: "method", text: EVIDENCE[row.track] ?? row.track }),
      node("span", { class: "sub", text: row.track === "pfm_oracle" ? "reference-pose source depth · analysis only" : "automatic photo skyline" }),
    ]),
    node("td", { class: "number", text: String(row.requested) }),
    node("td", { class: "number", text: String(row.completed) }),
    node("td", { class: "number", text: `${pct(row.blind_winner_success_rate)} · ${row.blind_winner_successes}/${row.requested}` }),
    node("td", { class: "number success", text: `${pct(row.full_lattice_oracle_success_rate)} · ${row.full_lattice_oracle_successes}/${row.requested}` }),
    node("td", { class: "number", text: num(row.median_blind_winner_horizontal_m, 1, " m") }),
    node("td", { class: "number", text: num(row.median_full_lattice_oracle_horizontal_m, 1, " m") }),
    node("td", { class: "number", text: num(row.median_blind_winner_yaw_deg, 1, "°") }),
    node("td", { class: "number", text: num(row.median_full_lattice_oracle_yaw_deg, 1, "°") }),
    node("td", { class: "number", text: num(row.median_full_lattice_oracle_estimator_rank, 0) }),
    node("td", { class: "number", text: `${row.shortlist_top_100_successes}/${row.requested}` }),
    node("td", { class: "number", text: num(row.runtime_s, 1, " s") }),
  ])));
  if (!aggregates.length) leaderboard.append(emptyRow(headers.length, "No atlas evidence tracks in this subset."));
}

function renderAtlasStrata(aggregates) {
  strataTitle.textContent = "Evidence tracks";
  strataNote.textContent = "Blind bars show estimator-selected success. Oracle bars show evaluation-only reachability in the same frozen score lattice.";
  strata.replaceChildren(...aggregates.map((row) => node("div", { class: "stratum" }, [
    node("div", { class: "stratum-head" }, [
      node("span", { class: "stratum-name", text: EVIDENCE[row.track] ?? row.track }),
      node("span", { class: "muted number", text: `n=${row.requested}` }),
    ]),
    node("div", { class: "stratum-bars" }, [
      bar("Blind", row.blind_winner_success_rate, ""),
      bar("Oracle ceiling", row.full_lattice_oracle_success_rate, "photo"),
    ]),
  ])));
}

function renderLegacySummary() {
  const subset = summary.subsets[selectedSubset] ?? summary.subsets.all;
  const oracle = subset.pfm_oracle;
  const photo = subset.photo_auto;
  subsetCount.textContent = `${subset.sample_count} samples`;
  cards.replaceChildren(
    card("PFM orientation success", pct(oracle.success_rate), `${oracle.successes}/${oracle.attempts} · errors included`, "good"),
    card("Photo orientation success", pct(photo.success_rate), `${photo.successes}/${photo.attempts} · errors included`, "photo"),
    card("Photo median yaw error", num(photo.median_abs_yaw_error_deg, 1, "°"), `PFM ${num(oracle.median_abs_yaw_error_deg, 1, "°")}`),
    card("Confidence calibration", "Retired", "GT-v2-derived CONFIRMED labels are ignored", "bad"),
  );
  const headers = ["Algorithm", "Evidence", "Prior", "Attempts", "Success", "Median yaw", "Median fit", "Missing / errors"];
  leaderboardHead.replaceChildren(node("tr", {}, headers.map((label) => node("th", { text: label }))));
  leaderboard.replaceChildren(
    legacyLeaderboardRow("PFM / source depth", oracle),
    legacyLeaderboardRow("Automatic photo skyline", photo),
  );
  renderLegacyStrata();
}

function legacyLeaderboardRow(evidence, track) {
  return node("tr", {}, [
    node("td", {}, [node("span", { class: "method", text: "Horizon" }), node("span", { class: "sub", text: "orientation only · legacy" })]),
    node("td", { text: evidence }),
    node("td", { text: "Known position; no yaw prior" }),
    node("td", { class: "number", text: String(track.attempts) }),
    node("td", { class: "number", text: pct(track.success_rate) }),
    node("td", { class: "number", text: num(track.median_abs_yaw_error_deg, 1, "°") }),
    node("td", { class: "number", text: num(track.median_fit_px, 1, " px") }),
    node("td", { class: "number", text: String(track.errors_or_missing) }),
  ]);
}

function renderLegacyStrata() {
  strataTitle.textContent = "Dataset gates";
  strataNote.textContent = "Old proxy subsets are retained for provenance only. MAP_A uses fixed-pose angular agreement; height is a separate raw-altitude check.";
  strata.replaceChildren(...Object.entries(summary.subsets).map(([key, subset]) => node("div", { class: "stratum" }, [
    node("div", { class: "stratum-head" }, [node("span", { class: "stratum-name", text: LABELS[key] ?? key }), node("span", { class: "muted", text: `n=${subset.sample_count}` })]),
    node("div", { class: "stratum-bars" }, [bar("PFM", subset.pfm_oracle.success_rate, ""), bar("Photo", subset.photo_auto.success_rate, "photo")]),
  ])));
}

function bar(label, value, className) {
  return node("div", {}, [
    node("div", { class: "stratum-head" }, [node("span", { class: "muted", text: label }), node("span", { class: "number", text: pct(value) })]),
    node("div", { class: `bar ${className}` }, [node("i", { style: `width:${Math.round(100 * (value ?? 0))}%` })]),
  ]);
}

async function loadRun() {
  if (!runSelect.value) return;
  loadGeneration += 1;
  const generation = loadGeneration;
  summaryController?.abort();
  casesController?.abort();
  summary = null;
  selectedSubset = "all";
  clearRunContent("Loading benchmark…");
  summaryController = new AbortController();
  const loaded = await json(`/api/bench/runs/${encodeURIComponent(runSelect.value)}/summary`, summaryController.signal);
  if (generation !== loadGeneration) return;
  summary = loaded;
  selectedSubset = summary.default_subset ?? "all";
  renderTabs();
  renderProvenance();
  renderSummary();
  await loadCases(generation);
}

async function loadCases(generation = loadGeneration) {
  if (!runSelect.value || !summary) return;
  casesController?.abort();
  clearCases("Loading samples…");
  casesController = new AbortController();
  const params = new URLSearchParams({ subset: selectedSubset, limit: "500", query: search.value.trim() });
  const result = await json(`/api/bench/runs/${encodeURIComponent(runSelect.value)}/cases?${params}`, casesController.signal);
  if (generation !== loadGeneration) return;
  caseCount.textContent = `${result.total} ${result.mode === "matrix" ? "strategy cells" : result.mode === "atlas" ? "sample tracks" : "samples"}`;
  if (result.mode === "matrix") renderMatrixCases(result.rows);
  else if (result.mode === "atlas") renderAtlasCases(result.rows);
  else renderLegacyCases(result.rows);
}

function clearRunContent(message) {
  subsetTabs.replaceChildren();
  provenance.replaceChildren();
  warning.hidden = true;
  warning.replaceChildren();
  cards.replaceChildren(card("Benchmark", "…", message));
  subsetCount.textContent = "";
  leaderboardHead.replaceChildren();
  leaderboard.replaceChildren(emptyRow(1, message));
  strata.replaceChildren();
  clearCases(message);
}

function clearCases(message) {
  caseCount.textContent = message;
  casesHead.replaceChildren();
  casesBody.replaceChildren(emptyRow(1, message));
}

function renderMatrixCases(rows) {
  const headers = ["Sample", "Strategy", "Evidence / prior", "Outcome", "Pose error", "Δ vs prior", "GT ↔ DEM", "Runtime"];
  casesHead.replaceChildren(node("tr", {}, headers.map((label) => node("th", { text: label }))));
  casesBody.replaceChildren(...rows.map((item) => {
    const compatibility = item.compatibility ?? {};
    const height = compatibility.height ?? {};
    const errors = item.errors ?? {};
    const delta = item.delta_vs_prior ?? {};
    const outcome = item.status === "error" ? item.error?.type ?? "Error" : item.status === "skipped" ? "Skipped" : item.success === true ? "Pass" : "Fail";
    const outcomeClass = item.success === true ? "success" : item.status === "skipped" ? "muted" : "failure";
    const exclusions = item.ranking_eligible ? "ranking eligible" : (item.ranking_exclusions ?? []).join(" · ") || item.skip_reason || "excluded";
    const ambiguity = originalMetadataAmbiguity(item);
    const candidateValidation = candidateValidationSummary(item);
    return node("tr", {}, [
      node("td", {}, [
        node("a", { class: "sample-link", href: `/gt?sample=${encodeURIComponent(item.name)}`, text: item.name }),
        node("span", { class: "sub", text: item.manual ? "manual reference" : "automatic reference" }),
        ...(ambiguity ? [node("span", { class: "sub", text: ambiguity })] : []),
      ]),
      node("td", {}, [node("span", { class: "method", text: ALGORITHMS[item.algorithm] ?? item.algorithm }), node("span", { class: "sub", text: exclusions })]),
      node("td", {}, [node("span", { text: EVIDENCE[item.evidence_track] ?? item.evidence_track }), node("span", { class: "sub", text: PRIORS[item.prior_regime] ?? item.prior_regime })]),
      node("td", {}, [
        node("span", { class: `outcome ${outcomeClass}`, text: outcome }),
        node("span", { class: "sub", text: item.outcome ?? item.skip_reason ?? "" }),
        ...(candidateValidation ? [node("span", { class: "sub", text: candidateValidation })] : []),
      ]),
      node("td", {}, [node("span", { class: "number", text: `${num(errors.horizontal_position_m, 1, " m")} · ${num(errors.yaw_deg, 1, "°")}` }), node("span", { class: "sub", text: "position · yaw" })]),
      node("td", {}, [node("span", { class: `number ${(delta.horizontal_position_m ?? 0) <= 0 ? "success" : "failure"}`, text: `${signed(delta.horizontal_position_m, 1, " m")} · ${signed(delta.yaw_deg, 1, "°")}` }), node("span", { class: "sub", text: "negative is improvement" })]),
      node("td", {}, [node("span", { class: "number", text: `${compatibility.tier ?? "—"} · ${num(compatibility.p90_deg, 2, "°")}` }), node("span", { class: "sub", text: `${height.tier ?? "no height gate"} · ${signed(height.raw_camera_clearance_m, 1, " m")}` })]),
      node("td", { class: "number", text: num(item.runtime_s, 1, " s") }),
    ]);
  }));
  if (!rows.length) casesBody.append(emptyRow(headers.length, "No strategy cells in this subset."));
}

function renderAtlasCases(rows) {
  const headers = ["Sample", "Evidence", "Blind winner", "Full-lattice oracle", "GT target reach (evaluation-only)", "Controlled prior", "GT ↔ DEM", "Runtime"];
  casesHead.replaceChildren(node("tr", {}, headers.map((label) => node("th", { text: label }))));
  casesBody.replaceChildren(...rows.map((item) => {
    const compatibility = item.compatibility ?? {};
    const height = compatibility.height ?? {};
    const archive = item.archive ?? {};
    const top100 = item.shortlist_top_100;
    const firstReach = item.shortlist_first_reach;
    const evidenceStatus = item.status === "ok" ? "complete" : item.status === "evidence_rejected" ? "evidence rejected" : item.status;
    return node("tr", {}, [
      node("td", {}, [
        node("a", { class: "sample-link", href: `/gt?sample=${encodeURIComponent(item.name)}`, text: item.name }),
        node("span", { class: "sub", text: item.manual ? "manual reference" : "automatic reference" }),
      ]),
      node("td", {}, [
        node("span", { class: "method", text: EVIDENCE[item.evidence_track] ?? item.evidence_track }),
        node("span", { class: "sub", text: evidenceStatus ?? "missing" }),
      ]),
      atlasCandidateCell(item.blind_winner, "estimator-selected", archive, item.reference_position_probe),
      atlasCandidateCell(item.full_lattice_oracle, "evaluation-only GT oracle", archive),
      node("td", {}, firstReach ? [
        node("span", { class: "number success", text: `Top ${firstReach.requested_k}` }),
        node("span", { class: "sub", text: `${firstReach.evaluated_k} evaluated · reached ≤100 m / 5°` }),
      ] : top100 ? [
        node("span", { class: "number failure", text: "Missed top 100" }),
        node("span", { class: "sub", text: `${top100.evaluated_k} evaluated · no target-successful mode` }),
      ] : [node("span", { class: "muted", text: "—" })]),
      node("td", {}, [
        node("span", { class: "number", text: `${num(item.prior_errors?.horizontal_position_m, 1, " m")} · ${num(item.prior_errors?.yaw_deg, 1, "°")}` }),
        node("span", { class: "sub", text: "reference-derived control · only east/north centres the grid" }),
        node("span", { class: "sub", text: "recorded yaw/pitch/altitude do not constrain scoring" }),
      ]),
      node("td", {}, [
        node("span", { class: "number", text: `${compatibility.tier ?? "—"} · ${num(compatibility.p90_deg, 2, "°")}` }),
        node("span", { class: "sub", text: `${height.tier ?? "no height gate"} · ${signed(height.raw_camera_clearance_m, 1, " m")}` }),
      ]),
      node("td", { class: "number", text: num(item.runtime_s, 1, " s") }),
    ]);
  }));
  if (!rows.length) casesBody.append(emptyRow(headers.length, "No pose-atlas sample tracks in this subset."));
}

function atlasCandidateCell(candidate, label, archive, referenceProbe = null) {
  if (!candidate) return node("td", {}, [node("span", { class: "muted", text: "—" }), node("span", { class: "sub", text: label })]);
  const errors = candidate.errors ?? {};
  const rankScope = candidate.estimator_rank_scope === "full_score_lattice" ? "full lattice" : "shortlist";
  const rankTotal = candidate.estimator_rank_scope === "full_score_lattice"
    ? archive?.full_lattice?.hypothesis_count
    : archive?.candidate_count;
  const details = [
    node("span", {
      class: `number ${candidate.reaches_target ? "success" : "failure"}`,
      text: `${num(errors.horizontal_position_m, 1, " m")} · ${num(errors.yaw_deg, 1, "°")}`,
    }),
    node("span", { class: "sub", text: `${label} · rank ${candidate.estimator_rank ?? "—"}${rankTotal ? `/${rankTotal}` : ""} in ${rankScope}` }),
  ];
  const delta = referenceProbe?.score_delta_reference_minus_blind_winner;
  if (delta != null) {
    const outcome = Number(delta) > 0 ? "loses" : "beats blind";
    details.push(node("span", {
      class: "sub",
      text: `reference-east/north probe ${outcome} by ${num(Math.abs(Number(delta)), 2, " px")} · yaw error ${num(referenceProbe?.errors?.yaw_deg, 1, "°")}`,
    }));
  }
  return node("td", {}, details);
}

function renderLegacyCases(rows) {
  const headers = ["Sample", "GT ↔ DEM", "PFM oracle", "Photo auto", "Extraction", "Reference"];
  casesHead.replaceChildren(node("tr", {}, headers.map((label) => node("th", { text: label }))));
  casesBody.replaceChildren(...rows.map((item) => {
    const compat = item.compatibility ?? {};
    const height = compat.height ?? {};
    const compatText = compat.proxy_px != null ? `${num(compat.proxy_px, 1, " px")} · ${compat.tier}` : `${num(compat.p90_deg, 2, "° p90")} · ${compat.tier ?? "—"}`;
    const ambiguity = originalMetadataAmbiguity(item);
    return node("tr", {}, [
      node("td", {}, [node("a", { class: "sample-link", href: `/gt?sample=${encodeURIComponent(item.name)}`, text: item.name })]),
      node("td", {}, [node("span", { class: "number", text: compatText }), node("span", { class: "sub", text: height.tier ? `${height.tier} · ${signed(height.raw_camera_clearance_m, 1, " m")}` : "height gate unavailable" })]),
      legacyTrackCell(item.pfm_oracle),
      legacyTrackCell(item.photo_auto),
      node("td", { class: "number", text: num(item.extraction_error_px, 1, " px") }),
      node("td", {}, [
        node("span", { class: "source-tag", text: item.manual ? "MANUAL" : "AUTO" }),
        ...(ambiguity ? [node("span", { class: "sub", text: ambiguity })] : []),
      ]),
    ]);
  }));
  if (!rows.length) casesBody.append(emptyRow(headers.length, "No samples in this subset."));
}

function legacyTrackCell(track) {
  if (!track) return node("td", { text: "—" });
  return node("td", {}, [
    node("span", { class: `number ${track.success ? "success" : "failure"}`, text: num(Math.abs(track.yaw_error_deg), 1, "°") }),
    node("span", { class: "sub", text: track.verdict ?? "UNCALIBRATED" }),
  ]);
}

function emptyRow(columns, message) {
  return node("tr", {}, [node("td", { colspan: String(columns), class: "empty", text: message })]);
}

runSelect.addEventListener("change", () => void loadRun().catch(showError));
search.addEventListener("input", () => {
  clearTimeout(queryTimer);
  queryTimer = setTimeout(() => void loadCases(loadGeneration).catch(showError), 180);
});

function showError(error) {
  if (error?.name === "AbortError") return;
  warning.hidden = false;
  warning.replaceChildren(node("p", { text: error.message ?? String(error) }));
}

(async () => {
  try {
    runs = await json("/api/bench/runs");
    renderRuns();
    if (!runs.length) throw new Error("No benchmark artifacts found. Run peakle.scripts.bench_pose_matrix first.");
    await loadRun();
  } catch (error) {
    showError(error);
  }
})();

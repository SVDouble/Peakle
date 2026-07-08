"use strict";

const MIN_FUZZY_SCORE = 0.62;

export function normalizeSearchText(value) {
  return String(value ?? "")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

export function fuzzySearchScore(query, weightedTexts) {
  const queryTokens = searchTokens(query);
  if (!queryTokens.length) {
    return 1;
  }
  let total = 0;
  for (const token of queryTokens) {
    let best = 0;
    for (const entry of weightedTexts) {
      const score = tokenScore(token, entry.text) * (entry.weight ?? 1);
      if (score > best) {
        best = score;
      }
    }
    if (best < tokenThreshold(token)) {
      return 0;
    }
    total += Math.min(1.2, best);
  }
  return total / queryTokens.length;
}

export function hasFuzzyMatch(query, weightedTexts) {
  return fuzzySearchScore(query, weightedTexts) >= MIN_FUZZY_SCORE;
}

function searchTokens(value) {
  return normalizeSearchText(value).split(/\s+/).filter(Boolean);
}

function tokenScore(queryToken, text) {
  const normalized = normalizeSearchText(text);
  if (!normalized) {
    return 0;
  }
  const candidates = [normalized, ...normalized.split(/\s+/).filter(Boolean)];
  return Math.max(...candidates.map((candidate) => candidateTokenScore(queryToken, candidate)));
}

function candidateTokenScore(queryToken, candidate) {
  if (!queryToken || !candidate) {
    return 0;
  }
  if (candidate === queryToken) {
    return 1;
  }
  if (candidate.startsWith(queryToken)) {
    return 0.94 - Math.min(0.18, (candidate.length - queryToken.length) / Math.max(candidate.length, 1));
  }
  if (queryToken.length >= 3 && candidate.includes(queryToken)) {
    return 0.82 - Math.min(0.15, (candidate.length - queryToken.length) / Math.max(candidate.length, 1));
  }
  const distance = levenshtein(queryToken, candidate);
  const ratio = 1 - distance / Math.max(queryToken.length, candidate.length, 1);
  const subsequence = subsequenceScore(queryToken, candidate);
  return Math.max(ratio, subsequence);
}

function tokenThreshold(token) {
  if (token.length <= 2) {
    return 0.9;
  }
  if (token.length <= 4) {
    return 0.68;
  }
  return MIN_FUZZY_SCORE;
}

function subsequenceScore(queryToken, candidate) {
  let cursor = 0;
  let first = -1;
  let last = -1;
  for (let i = 0; i < candidate.length && cursor < queryToken.length; i += 1) {
    if (candidate[i] === queryToken[cursor]) {
      if (first < 0) {
        first = i;
      }
      last = i;
      cursor += 1;
    }
  }
  if (cursor !== queryToken.length) {
    return 0;
  }
  const span = Math.max(1, last - first + 1);
  const compactness = queryToken.length / span;
  const coverage = queryToken.length / Math.max(candidate.length, queryToken.length);
  return 0.58 * compactness + 0.24 * coverage;
}

function levenshtein(a, b) {
  if (a === b) {
    return 0;
  }
  if (!a.length) {
    return b.length;
  }
  if (!b.length) {
    return a.length;
  }
  let previous = Array.from({ length: b.length + 1 }, (_value, index) => index);
  let current = new Array(b.length + 1);
  for (let i = 1; i <= a.length; i += 1) {
    current[0] = i;
    for (let j = 1; j <= b.length; j += 1) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      current[j] = Math.min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + cost);
    }
    [previous, current] = [current, previous];
  }
  return previous[b.length];
}

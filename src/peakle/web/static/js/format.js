"use strict";

// Small DOM/formatting helpers shared across panels.

const SVG_NS = "http://www.w3.org/2000/svg";

export function formatNumber(value, unit) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return `${Number(value).toFixed(1)} ${unit}`;
}

export function formatDistance(valueM) {
  if (valueM >= 1000) {
    return `${Number(valueM / 1000).toFixed(valueM % 1000 === 0 ? 0 : 1)} km`;
  }
  return `${Math.round(valueM)} m`;
}

export function svgElement(name, attributes) {
  const element = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attributes)) {
    element.setAttribute(key, value);
  }
  return element;
}

// Largest box of the given aspect ratio centered inside the container.
export function fitContainBox(containerWidth, containerHeight, aspect) {
  const width = Math.max(1, containerWidth);
  const height = Math.max(1, containerHeight);
  let boxWidth = width;
  let boxHeight = width / aspect;
  if (boxHeight > height) {
    boxHeight = height;
    boxWidth = height * aspect;
  }
  return {
    width: Math.max(1, Math.round(boxWidth)),
    height: Math.max(1, Math.round(boxHeight)),
    left: Math.round((width - boxWidth) / 2),
    top: Math.round((height - boxHeight) / 2),
  };
}

export function setBox(element, box) {
  element.style.left = `${box.left}px`;
  element.style.top = `${box.top}px`;
  element.style.width = `${box.width}px`;
  element.style.height = `${box.height}px`;
}

export function clearChildren(element) {
  element.replaceChildren();
}

// Minimal hyperscript helper. `attrs` may include `class`, `text`, `html`,
// `dataset`, event handlers (`onclick`, ...), and plain attributes/props.
export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined) {
      continue;
    }
    if (key === "class") {
      node.className = value;
    } else if (key === "text") {
      node.textContent = value;
    } else if (key === "html") {
      node.innerHTML = value;
    } else if (key === "dataset") {
      Object.assign(node.dataset, value);
    } else if (key.startsWith("on") && typeof value === "function") {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (key in node) {
      node[key] = value;
    } else {
      node.setAttribute(key, value);
    }
  }
  for (const child of [].concat(children)) {
    if (child !== null && child !== undefined && child !== false) {
      node.append(child);
    }
  }
  return node;
}

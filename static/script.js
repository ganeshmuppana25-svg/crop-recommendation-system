/* Crop Recommendation System - Phase 2 frontend.
   Visual layer: theme toggle, sliders, animations, states.
   Functional behaviour (API calls, validation, presets, history) unchanged
   from Phase 1: predictions always come from the Flask/Decision Tree backend. */

"use strict";

var API = {
  predict: "/api/predict",
  metrics: "/api/metrics",
  dataset: "/api/dataset",
  health: "/api/health"
};

var FEATURES = [
  { key: "N", label: "Nitrogen (N)", step: "1", dec: 0, icon: "i-flask" },
  { key: "P", label: "Phosphorus (P)", step: "1", dec: 0, icon: "i-flask" },
  { key: "K", label: "Potassium (K)", step: "1", dec: 0, icon: "i-flask" },
  { key: "temperature", label: "Temperature (\u00B0C)", step: "0.01", dec: 2, icon: "i-thermo" },
  { key: "humidity", label: "Humidity (%)", step: "0.01", dec: 2, icon: "i-drop" },
  { key: "ph", label: "Soil pH", step: "0.01", dec: 2, icon: "i-flask" },
  { key: "rainfall", label: "Rainfall (mm)", step: "0.01", dec: 2, icon: "i-rain" }
];

var HISTORY_KEY = "crop_history_v1";
var THEME_KEY = "crs_theme";
var STEPPER_MS = 750;          // short, real fetch runs in parallel
var CONFIDENCE_ANIM_MS = 900;  // counter duration

var validationRanges = {};
var classMeans = {};
var reducedMotion = false;
var confidenceRaf = null;

/* ------------------------- small helpers ------------------------- */

function $(id) { return document.getElementById(id); }

function prefersReducedMotion() {
  return window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function fetchJSON(url, options) {
  return fetch(url, options).then(function (res) {
    return res.json().then(function (data) {
      return { ok: res.ok, status: res.status, data: data };
    });
  });
}

function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, function (c) {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
  });
}

function fmt(v, dec) {
  var n = Number(v);
  if (isNaN(n)) return String(v);
  return n.toFixed(dec === undefined ? 2 : dec);
}

function delay(ms) {
  return new Promise(function (resolve) { setTimeout(resolve, ms); });
}

function icon(name) {
  return '<svg class="icon"><use href="#' + name + '"/></svg>';
}

/* ------------------------- theme (light / dark) ------------------------- */

function currentTheme() {
  return document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  try { localStorage.setItem(THEME_KEY, theme); } catch (e) { /* ignore */ }
}

function toggleTheme() {
  applyTheme(currentTheme() === "dark" ? "light" : "dark");
}

function initTheme() {
  // Head script already set the attribute; re-save so the choice persists.
  try {
    if (!localStorage.getItem(THEME_KEY)) {
      localStorage.setItem(THEME_KEY, currentTheme());
    }
  } catch (e) { /* ignore */ }
  $("theme-toggle").addEventListener("click", toggleTheme);
}

/* ------------------------- input building (sliders) ------------------------- */

function buildInputs() {
  var grid = $("inputs-grid");
  grid.innerHTML = "";
  FEATURES.forEach(function (f) {
    var div = document.createElement("div");
    div.className = "field";
    div.innerHTML =
      '<div class="field-head">' +
        '<span class="field-icon">' + icon(f.icon) + "</span>" +
        '<label for="in-' + f.key + '">' + escapeHTML(f.label) + "</label>" +
        '<span class="field-value" id="val-' + f.key + '">&ndash;</span>' +
      "</div>" +
      '<input type="range" id="rng-' + f.key + '" min="0" max="100" step="' + f.step + '" value="0" aria-label="' + escapeHTML(f.label) + ' slider" />' +
      '<div class="field-sub">' +
        '<input type="number" id="in-' + f.key + '" name="' + f.key + '" step="' + f.step + '" required />' +
        '<span class="range-hints"><span id="min-' + f.key + '"></span><span id="max-' + f.key + '"></span></span>' +
      "</div>";
    grid.appendChild(div);
  });

  // Two-way sync: slider <-> number <-> value badge
  FEATURES.forEach(function (f) {
    var rng = $("rng-" + f.key);
    var num = $("in-" + f.key);
    rng.addEventListener("input", function () {
      num.value = rng.value;
      updateFieldVisuals(f);
    });
    num.addEventListener("input", function () {
      var v = parseFloat(num.value);
      if (!isNaN(v) && rng.min !== "" && v >= +rng.min && v <= +rng.max) {
        rng.value = String(v);
      }
      updateFieldVisuals(f);
    });
  });
}

/* Slider fill gradient + live value badge */
function updateFieldVisuals(f) {
  var rng = $("rng-" + f.key);
  var num = $("in-" + f.key);
  var badge = $("val-" + f.key);
  if (!rng || !num || !badge) return;

  var min = rng.min === "" ? 0 : parseFloat(rng.min);
  var max = rng.max === "" ? 100 : parseFloat(rng.max);
  var v = parseFloat(num.value);

  if (isNaN(v)) {
    badge.textContent = "\u2013";
    rng.value = String(min);
  } else {
    badge.textContent = fmt(v, f.dec);
    var clamped = Math.min(Math.max(v, min), max);
    rng.value = String(clamped);
  }

  var pct = max > min ? ((parseFloat(rng.value) - min) / (max - min)) * 100 : 0;
  rng.style.background = "linear-gradient(to right, var(--primary) " + pct +
    "%, var(--track) " + pct + "%)";
}

/* Read the whole form (same keys/ranges as Phase 1 - unchanged contract) */
function readFormValues() {
  var values = {};
  var missing = [];
  FEATURES.forEach(function (f) {
    var input = $("in-" + f.key);
    var raw = (input.value || "").trim();
    if (raw === "") { missing.push(f.label); return; }
    var num = Number(raw);
    if (isNaN(num)) { missing.push(f.label); return; }
    values[f.key] = num;
  });
  return { values: values, missing: missing };
}

/* Set one field from presets/history: updates number, slider and badge */
function setFieldValue(key, value) {
  var num = $("in-" + key);
  if (!num || value === undefined || value === null) return;
  num.value = value;
  var f = null;
  for (var i = 0; i < FEATURES.length; i++) {
    if (FEATURES[i].key === key) { f = FEATURES[i]; break; }
  }
  if (f) updateFieldVisuals(f);
}

function applyRangesToInputs() {
  FEATURES.forEach(function (f) {
    var r = validationRanges[f.key];
    var rng = $("rng-" + f.key);
    var num = $("in-" + f.key);
    if (r && rng && num) {
      rng.min = r.min;
      rng.max = r.max;
      num.min = r.min;
      num.max = r.max;
      num.placeholder = fmt((r.min + r.max) / 2, f.dec);
      $("min-" + f.key).textContent = fmt(r.min, f.dec);
      $("max-" + f.key).textContent = fmt(r.max, f.dec);
      updateFieldVisuals(f);
    }
  });
}

/* ------------------------- dataset + metrics (KPIs) ------------------------- */

function renderDatasetInfo(d) {
  var ds = d.dataset || {};
  var items = [
    { icon: "i-flask", label: "Rows (samples)", value: ds.rows },
    { icon: "i-sliders", label: "Features", value: ds.n_features },
    { icon: "i-sprout", label: "Crop classes", value: ds.n_classes },
    { icon: "i-check", label: "Missing values", value: ds.missing_values },
    { icon: "i-check", label: "Duplicates removed", value: ds.duplicates_removed },
    { icon: "i-leaf", label: "Target column", value: ds.target }
  ];
  $("dataset-summary").innerHTML = items.map(function (it) {
    return '<div class="kpi-card">' +
      '<span class="kpi-head">' + icon(it.icon) + escapeHTML(it.label) + "</span>" +
      '<span class="kpi-value">' + escapeHTML(it.value) + "</span>" +
      '<span class="kpi-sub">verified during training</span>' +
      "</div>";
  }).join("");

  var classes = ds.classes || [];
  $("dataset-classes").textContent = classes.join(", ");

  var ranges = d.validation_ranges || {};
  $("dataset-ranges").innerHTML = FEATURES.map(function (f) {
    var r = ranges[f.key];
    if (!r) return "";
    return '<div class="summary-item"><span class="k">' + escapeHTML(f.label) +
      "</span>" + r.min + " to " + r.max + "</div>";
  }).join("");
}

/* Animate .imp-bar / .top3-bar widths 0 -> target (data-w holds the target %) */
function animateBars(container) {
  var bars = container.querySelectorAll("[data-w]");
  Array.prototype.forEach.call(bars, function (bar) {
    bar.style.width = "0%";
    var target = bar.getAttribute("data-w") + "%";
    if (reducedMotion) {
      bar.style.width = target;
    } else {
      // force reflow so the transition always runs from 0
      void bar.offsetWidth;
      window.requestAnimationFrame(function () { bar.style.width = target; });
    }
  });
}

function renderImportance(containerId, importance) {
  var el = $(containerId);
  var entries = Object.keys(importance || {}).map(function (k) {
    return [k, Number(importance[k])];
  }).sort(function (a, b) { return b[1] - a[1]; });
  if (!entries.length) { el.innerHTML = "<p class='muted-text'>Not available.</p>"; return; }
  // importance values are 0..1 -> percent computed exactly once here
  el.innerHTML = entries.map(function (e) {
    var pct = (e[1] * 100).toFixed(2);
    return '<div class="importance-row"><span class="imp-name">' + escapeHTML(e[0]) +
      '</span><div class="imp-track"><div class="imp-bar" data-w="' + pct + '"></div></div>' +
      '<span class="imp-pct">' + pct + "%</span></div>";
  }).join("");
  animateBars(el);
}

function renderMetrics(m) {
  var metrics = m.metrics || {};
  var cards = [
    { icon: "i-check", label: "Accuracy", value: metrics.accuracy },
    { icon: "i-check", label: "Precision (wtd)", value: metrics.precision_weighted },
    { icon: "i-check", label: "Recall (wtd)", value: metrics.recall_weighted },
    { icon: "i-check", label: "F1 Score (wtd)", value: metrics.f1_weighted }
  ];
  $("metrics-grid").innerHTML = cards.map(function (c) {
    var pct = c.value === undefined ? "--" : (c.value * 100).toFixed(2) + "%";
    return '<div class="kpi-card">' +
      '<span class="kpi-head">' + icon(c.icon) + escapeHTML(c.label) + "</span>" +
      '<span class="kpi-value">' + pct + "</span>" +
      '<span class="kpi-sub">on 440-sample test set</span>' +
      "</div>";
  }).join("");

  var model = m.model || {};
  var line = "Algorithm: " + (model.type || "DecisionTreeClassifier") +
    " &middot; Train rows: " + (model.train_rows || "--") +
    " &middot; Test rows: " + (model.test_rows || "--") +
    " &middot; Tree depth: " + (model.tree_depth || "--") +
    " &middot; Leaves: " + (model.tree_leaves || "--");
  var p = document.createElement("p");
  p.className = "muted-text section-note";
  p.innerHTML = line;
  $("metrics-grid").after(p);

  renderImportance("global-feature-importance", m.feature_importance);
}

function checkHealth() {
  fetchJSON(API.health).then(function (r) {
    var banner = $("health-banner");
    if (r.ok && r.data.model_loaded) {
      banner.innerHTML = '<span class="dot"></span>Model ready &middot; ' +
        escapeHTML(r.data.model_type) + " &middot; " + r.data.n_classes + " crops";
      banner.classList.add("ok");
    } else {
      banner.innerHTML = '<span class="dot"></span>Model not loaded &mdash; run train_model.py';
      banner.classList.add("bad");
    }
  }).catch(function () {
    var banner = $("health-banner");
    banner.innerHTML = '<span class="dot"></span>Cannot reach the Flask server';
    banner.classList.add("bad");
  });
}

/* ------------------------- prediction flow ------------------------- */

function setStepper(step) {
  // step: 1..3 active, 4 = all done
  var steps = document.querySelectorAll("#predict-stepper .step");
  Array.prototype.forEach.call(steps, function (el, i) {
    el.classList.remove("active", "done");
    if (i + 1 < step) el.classList.add("done");
    if (i + 1 === step) el.classList.add("active");
  });
}

function showStepper() {
  $("predict-stepper").classList.remove("hidden");
  $("result-empty").classList.add("hidden");
  $("result-content").classList.add("hidden");
  setStepper(1);
  if (!reducedMotion) {
    setTimeout(function () { setStepper(2); }, STEPPER_MS * 0.4);
  }
}

function hideStepper() {
  $("predict-stepper").classList.add("hidden");
}

/* Confidence counter: animates 0 -> actual exactly once, ends at the true value.
   data.confidence is already 0-100 from the API (used as-is, never re-scaled). */
function animateConfidence(finalConfidence) {
  var textEl = $("result-confidence");
  var barEl = $("confidence-bar");
  var cls = finalConfidence >= 75 ? "confidence-high" :
            finalConfidence >= 50 ? "confidence-mid" : "confidence-low";

  if (confidenceRaf) window.cancelAnimationFrame(confidenceRaf);
  textEl.className = "result-confidence " + cls;

  if (reducedMotion) {
    textEl.textContent = fmt(finalConfidence) + "%";
    barEl.style.width = finalConfidence + "%";
    return;
  }

  barEl.style.width = "0%";
  textEl.textContent = "0%";
  var t0 = null;
  function frame(ts) {
    if (t0 === null) t0 = ts;
    var p = Math.min((ts - t0) / CONFIDENCE_ANIM_MS, 1);
    var eased = 1 - Math.pow(1 - p, 3); // ease-out cubic
    // snap to the exact endpoint for the last fraction of a percent
    var done = p >= 1 || finalConfidence * eased > finalConfidence - 0.005;
    var current = done ? finalConfidence : finalConfidence * eased;
    textEl.textContent = fmt(current) + "%";
    barEl.style.width = current + "%";
    if (p < 1) {
      confidenceRaf = window.requestAnimationFrame(frame);
    } else {
      textEl.textContent = fmt(finalConfidence) + "%"; // exact endpoint
      barEl.style.width = finalConfidence + "%";
      confidenceRaf = null;
    }
  }
  confidenceRaf = window.requestAnimationFrame(frame);
}

function handlePredict(event) {
  event.preventDefault();
  var errorEl = $("form-error");
  errorEl.classList.add("hidden");

  var form = readFormValues();
  if (form.missing.length) {
    errorEl.textContent = "Please enter valid numbers for: " + form.missing.join(", ");
    errorEl.classList.remove("hidden");
    return;
  }

  var btn = $("recommend-btn");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span><span class="btn-label">Predicting&hellip;</span>';

  showStepper();
  $("result-section").classList.remove("hidden");

  var fetchPromise = fetchJSON(API.predict, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(form.values)
  });

  // Stepper is brief and real: fetch runs in parallel, never a fake long wait.
  Promise.all([fetchPromise, delay(reducedMotion ? 60 : STEPPER_MS)]).then(function (results) {
    var r = results[0];
    btn.disabled = false;
    btn.innerHTML = icon("i-wheat") + '<span class="btn-label">Recommend Crop</span>';
    hideStepper();
    if (r.ok && r.data.success) {
      renderResult(r.data);
      addHistoryItem(r.data);
    } else {
      $("result-empty").classList.remove("hidden");
      var msg = r.data.error || "Prediction failed.";
      if (r.data.details && r.data.details.length) {
        msg += "\n" + r.data.details.join("\n");
      }
      errorEl.textContent = msg;
      errorEl.classList.remove("hidden");
    }
  }).catch(function () {
    btn.disabled = false;
    btn.innerHTML = icon("i-wheat") + '<span class="btn-label">Recommend Crop</span>';
    hideStepper();
    $("result-empty").classList.remove("hidden");
    errorEl.textContent = "Could not reach the server. Is Flask running?";
    errorEl.classList.remove("hidden");
  });
}

function medallionAccent(crop) {
  var accents = ["", "accent-1", "accent-2", "accent-3"];
  var hash = 0;
  for (var i = 0; i < crop.length; i++) hash = (hash * 31 + crop.charCodeAt(i)) >>> 0;
  return accents[hash % accents.length];
}

function renderResult(data) {
  $("result-empty").classList.add("hidden");
  $("result-content").classList.remove("hidden");

  $("result-crop").textContent = data.crop;
  var medallion = $("crop-medallion");
  medallion.className = "crop-medallion " + medallionAccent(String(data.crop));

  animateConfidence(Number(data.confidence));

  $("top3-list").innerHTML = (data.top3 || []).map(function (t, i) {
    var pct = Number(t.confidence).toFixed(2); // 0-100 straight from the model
    return '<li class="top3-item">' +
      '<span class="rank rank-' + (i + 1) + '">' + (i + 1) + "</span>" +
      '<span class="top3-name">' + escapeHTML(t.crop) + "</span>" +
      '<div class="top3-track"><div class="top3-bar" data-w="' + pct + '"></div></div>' +
      '<span class="top3-pct">' + pct + "%</span>" +
      "</li>";
  }).join("");
  animateBars($("top3-list"));

  var summary = data.input_summary || {};
  $("input-summary").innerHTML = FEATURES.map(function (f) {
    return '<div class="summary-item"><span class="k">' + escapeHTML(f.label) +
      "</span>" + escapeHTML(summary[f.key]) + "</div>";
  }).join("");

  $("crop-info").textContent = data.crop_info || "";
  $("explanation").textContent = data.explanation || "";
  renderImportance("result-feature-importance", data.feature_importance);

  $("result-section").classList.remove("hidden");
  $("result-section").scrollIntoView({ behavior: reducedMotion ? "auto" : "smooth" });
}

/* ------------------------- presets ------------------------- */

function handlePreset(event) {
  var btn = event.target.closest(".preset-btn");
  if (!btn) return;
  var crop = btn.getAttribute("data-crop");
  var means = classMeans[crop];
  if (!means) return;
  FEATURES.forEach(function (f) {
    if (means[f.key] !== undefined) {
      setFieldValue(f.key, fmt(means[f.key], f.dec));
    }
  });
  $("form-error").classList.add("hidden");
}

/* ------------------------- reset ------------------------- */

function handleReset() {
  FEATURES.forEach(function (f) {
    var input = $("in-" + f.key);
    if (input) input.value = "";
    updateFieldVisuals(f); // clears badge + slider
  });
  $("result-content").classList.add("hidden");
  $("result-empty").classList.remove("hidden");
  $("predict-stepper").classList.add("hidden");
  $("form-error").classList.add("hidden");
  window.scrollTo({ top: 0, behavior: reducedMotion ? "auto" : "smooth" });
}

/* ------------------------- history (localStorage) ------------------------- */

function loadHistory() {
  try {
    var raw = localStorage.getItem(HISTORY_KEY);
    var list = raw ? JSON.parse(raw) : [];
    return Array.isArray(list) ? list : [];
  } catch (e) {
    return [];
  }
}

function saveHistory(list) {
  try { localStorage.setItem(HISTORY_KEY, JSON.stringify(list)); }
  catch (e) { /* storage blocked - ignore */ }
}

function addHistoryItem(data) {
  var list = loadHistory();
  list.unshift({
    ts: new Date().toISOString(),
    inputs: data.input_summary,
    crop: data.crop,
    confidence: data.confidence
  });
  if (list.length > 20) list = list.slice(0, 20);
  saveHistory(list);
  renderHistory();
}

function renderHistory() {
  var list = loadHistory();
  var el = $("history-list");
  if (!list.length) {
    el.innerHTML = '<p class="history-empty">No predictions yet. Your history will appear here.</p>';
    return;
  }
  el.innerHTML = list.map(function (item, i) {
    var d = new Date(item.ts);
    var when = isNaN(d.getTime()) ? item.ts : d.toLocaleString();
    var inputs = item.inputs || {};
    var brief = FEATURES.map(function (f) {
      return inputs[f.key] !== undefined ? f.key + "=" + inputs[f.key] : "";
    }).filter(Boolean).join(", ");
    return '<div class="history-item" style="animation-delay:' + Math.min(i * 0.05, 0.4) + 's">' +
      "<div><span class='crop'>" + icon("i-sprout") + " " + escapeHTML(item.crop) +
      '</span> <span class="meta">(' + fmt(item.confidence) + '%)</span><br>' +
      '<span class="meta">' + escapeHTML(when) + " &middot; " + escapeHTML(brief) +
      '</span></div><button type="button" class="secondary-btn small" data-restore="' +
      i + '">Restore</button></div>';
  }).join("");
}

function handleHistoryClick(event) {
  var btn = event.target.closest("[data-restore]");
  if (!btn) return;
  var list = loadHistory();
  var item = list[Number(btn.getAttribute("data-restore"))];
  if (!item || !item.inputs) return;
  FEATURES.forEach(function (f) {
    if (item.inputs[f.key] !== undefined) {
      setFieldValue(f.key, item.inputs[f.key]);
    }
  });
  window.scrollTo({ top: 0, behavior: reducedMotion ? "auto" : "smooth" });
}

function handleClearHistory() {
  if (!confirm("Clear all saved prediction history?")) return;
  saveHistory([]);
  renderHistory();
}

/* ------------------------- entrance reveal ------------------------- */

function initReveal() {
  var cards = document.querySelectorAll(".card");
  if (reducedMotion || !("IntersectionObserver" in window)) return; // stay visible
  Array.prototype.forEach.call(cards, function (card) { card.classList.add("reveal"); });
  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (entry.isIntersecting) {
        entry.target.classList.add("in-view");
        io.unobserve(entry.target);
      }
    });
  }, { threshold: 0.08 });
  Array.prototype.forEach.call(cards, function (card) { io.observe(card); });
}

/* ------------------------- init ------------------------- */

function init() {
  reducedMotion = prefersReducedMotion();
  initTheme();
  buildInputs();

  fetchJSON(API.dataset).then(function (r) {
    if (r.ok && r.data.success) {
      validationRanges = r.data.validation_ranges || {};
      classMeans = r.data.class_means || {};
      applyRangesToInputs();
      renderDatasetInfo(r.data);
    }
  }).catch(function () { /* page still usable for manual input */ });

  fetchJSON(API.metrics).then(function (r) {
    if (r.ok && r.data.success) renderMetrics(r.data);
  }).catch(function () { /* skeletons remain if metrics unavailable */ });

  checkHealth();
  renderHistory();
  initReveal();

  $("predict-form").addEventListener("submit", handlePredict);
  $("reset-btn").addEventListener("click", handleReset);
  $("clear-history-btn").addEventListener("click", handleClearHistory);
  $("history-list").addEventListener("click", handleHistoryClick);
  Array.prototype.forEach.call(
    document.querySelectorAll(".preset-btn"),
    function (btn) { btn.addEventListener("click", handlePreset); }
  );
}

document.addEventListener("DOMContentLoaded", init);

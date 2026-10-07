// Shared helpers for the standalone figure pages.
//
// Every figure is an HTML page embedded in the deck as an <iframe>. The deck
// forwards reveal.js fragment progress via postMessage ({type: "reveal-step"}),
// so a figure can build up in stages as the speaker advances. Opened on its
// own, a figure shows its final state (override with ?step=N).

(function () {
  const C = {
    text: "#0b0c0c",
    text2: "#505a5f",
    text3: "#6f777b",
    surface: "#ffffff",
    panel: "#f3f2f1",
    grid: "#e3e4e5",
    rule: "#b1b4b6",
    blue: "#1d70b8",
    darkBlue: "#003078",
    orange: "#f47738",
    turquoise: "#28a197",
    purple: "#912b88",
    lightBlue: "#5694ca",
    brown: "#b58840",
    red: "#d4351c",
    green: "#00703c",
    yellow: "#ffdd00",
  };

  // One colour per model, used identically on every slide.
  const MODEL = {
    v5_full: { label: "RIoT v5 (V5F)", short: "V5F", color: C.blue },
    v4r_full: { label: "Original thesis model (V4R)", short: "V4R", color: C.orange },
    e2_1_unify_iid: { label: "Unified encoding only (E2.1)", short: "E2.1", color: C.turquoise },
    loo_no_unify: { label: "V5F without unified encoding", short: "V5F − unified", color: C.purple },
    e1_6_complete_weather: { label: "Loss hygiene only (P2B)", short: "P2B", color: C.lightBlue },
    e3_3d_profile_inductive_zero: { label: "Best held-out screen (E3.3d)", short: "E3.3d", color: C.brown },
    naive_profile: { label: "Seasonal profile baseline", short: "Naïve profile", color: C.text2 },
  };

  // ── Step control ──────────────────────────────────────────────────────────
  const embedded = window.parent !== window;
  const params = new URLSearchParams(location.search);
  const urlStep = params.get("step");

  // Embedded figures are drawn at 1/ZOOM of the frame size and zoomed up, so
  // text and marks read from the back of a room at 1080p.
  const ZOOM = +(params.get("zoom") || (embedded ? 1.25 : 1));
  document.documentElement.style.zoom = ZOOM;
  let stepHandler = null;
  let maxSteps = 0;
  let currentStep = null;

  function applyStep(n) {
    if (!stepHandler) return;
    const s = Math.max(0, Math.min(maxSteps, n));
    if (s === currentStep) return;
    currentStep = s;
    stepHandler(s);
  }

  function onStep(handler, steps) {
    stepHandler = handler;
    maxSteps = steps;
    if (urlStep !== null) applyStep(+urlStep);
    else if (!embedded) applyStep(steps);
    else {
      applyStep(0);
      window.parent.postMessage({ type: "figure-ready" }, "*");
    }
  }

  window.addEventListener("message", (e) => {
    if (e.data && e.data.type === "reveal-step" && urlStep === null) applyStep(e.data.step);
  });

  // Toggle step classes: elements with data-step="k" appear once step >= k.
  function showSteps(root, step) {
    root.querySelectorAll("[data-step]").forEach((el) => {
      const k = +el.getAttribute("data-step");
      el.classList.toggle("step-hidden", step < k);
    });
  }

  // ── Tooltip ───────────────────────────────────────────────────────────────
  let tipEl = null;
  function tip() {
    if (!tipEl) {
      tipEl = document.createElement("div");
      tipEl.className = "tooltip";
      tipEl.setAttribute("role", "status");
      document.body.appendChild(tipEl);
    }
    return tipEl;
  }

  // rows: [{color, value, name}] — values lead, names follow. Built with
  // textContent only: labels come from data files.
  function showTip(event, title, rows) {
    const el = tip();
    el.replaceChildren();
    if (title) {
      const t = document.createElement("div");
      t.className = "tt-title";
      t.textContent = title;
      el.appendChild(t);
    }
    for (const r of rows) {
      const row = document.createElement("div");
      row.className = "tt-row";
      if (r.color) {
        const k = document.createElement("span");
        k.className = "tt-key";
        k.style.borderColor = r.color;
        row.appendChild(k);
      }
      const v = document.createElement("span");
      v.className = "tt-val";
      v.textContent = r.value;
      row.appendChild(v);
      if (r.name) {
        const n = document.createElement("span");
        n.className = "tt-name";
        n.textContent = r.name;
        row.appendChild(n);
      }
      el.appendChild(row);
    }
    // Pointer coordinates are in device pixels; the page is zoomed by ZOOM.
    const pad = 16;
    const rect = el.getBoundingClientRect();
    let x = event.clientX + pad;
    let y = event.clientY + pad;
    if (x + rect.width > window.innerWidth) x = event.clientX - rect.width - pad;
    if (y + rect.height > window.innerHeight) y = event.clientY - rect.height - pad;
    el.style.left = Math.max(0, x) / ZOOM + "px";
    el.style.top = Math.max(0, y) / ZOOM + "px";
    el.style.opacity = 1;
  }

  function hideTip() {
    if (tipEl) tipEl.style.opacity = 0;
  }

  // ── Small utilities ───────────────────────────────────────────────────────
  function size() {
    return { w: Math.floor(window.innerWidth / ZOOM), h: Math.floor(window.innerHeight / ZOOM) };
  }

  function runsByExperiment() {
    const map = {};
    for (const r of window.DATA_V5_RUNS.runs) map[r.experiment] = r;
    return map;
  }

  function legend(container, items) {
    const box = document.createElement("div");
    box.className = "legend";
    for (const it of items) {
      const item = document.createElement("span");
      item.className = "item";
      const sw = document.createElement("span");
      sw.className = "swatch-" + (it.shape || "line");
      if (it.shape === "line" || !it.shape) sw.style.borderColor = it.color;
      else sw.style.background = it.color;
      if (it.dashed) sw.style.borderTopStyle = "dashed";
      const label = document.createElement("span");
      label.textContent = it.label;
      item.append(sw, label);
      box.appendChild(item);
    }
    container.appendChild(box);
    return box;
  }

  const fmt = {
    pct: (v, d = 0) => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v * 100).toFixed(d) + "%",
    num: (v, d = 1) => (v == null ? "–" : v.toFixed(d)),
  };

  // Clean hairline axes: no domain line on the value axis, gridlines instead.
  function styleAxis(g, { grid = false, domain = true } = {}) {
    if (!domain) g.select(".domain").remove();
    if (grid) g.selectAll(".tick line").attr("stroke", C.grid);
    return g;
  }

  window.Fig = { C, MODEL, onStep, showSteps, showTip, hideTip, size, runsByExperiment, legend, fmt, styleAxis, embedded };
})();

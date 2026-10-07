// Shared layout and drawing for the pipeline schematics.
//
// Each audience page defines window.SCHEMATIC_CONTENT (all wording) and calls
// drawSchematic(). Everything is drawn as plain SVG with inline attributes, so
// the exported .svg and .pdf files are self-contained vector graphics.

(function () {
  const W = 1200;
  const H = 1020;
  const C = {
    text: "#0b0c0c",
    text2: "#505a5f",
    text3: "#6f777b",
    panel: "#f3f2f1",
    rule: "#b1b4b6",
    blue: "#1d70b8",
    darkBlue: "#003078",
    lightBlue: "#d2e2f1",
    paleBlue: "#e8f1f8",
    orange: "#f47738",
    red: "#d4351c",
    white: "#ffffff",
  };
  const FONT = "Arial, 'Helvetica Neue', Helvetica, 'Liberation Sans', sans-serif";
  const NS = "http://www.w3.org/2000/svg";

  // ── Primitives ────────────────────────────────────────────────────────────
  function el(tag, attrs, parent) {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) if (v != null) n.setAttribute(k, v);
    if (parent) parent.appendChild(n);
    return n;
  }

  function text(parent, x, y, str, o = {}) {
    const t = el("text", {
      x, y,
      "font-size": o.size || 16,
      "font-weight": o.weight || 400,
      fill: o.fill || C.text,
      "text-anchor": o.anchor || "start",
      "letter-spacing": o.spacing,
      "font-style": o.italic ? "italic" : null,
    }, parent);
    t.textContent = str;
    return t;
  }

  // Greedy word wrap using a measuring element (Arial metrics in the browser).
  let measurer = null;
  function measure(str, size, weight) {
    if (!measurer) {
      const svg = document.querySelector("svg");
      measurer = el("text", { x: -9999, y: -9999, "font-family": FONT }, svg);
    }
    measurer.setAttribute("font-size", size);
    measurer.setAttribute("font-weight", weight || 400);
    measurer.textContent = str;
    return measurer.getComputedTextLength();
  }
  function wrap(str, maxW, size, weight) {
    const words = str.split(" ");
    const lines = [];
    let line = "";
    for (const w of words) {
      const test = line ? line + " " + w : w;
      if (line && measure(test, size, weight) > maxW) { lines.push(line); line = w; } else line = test;
    }
    if (line) lines.push(line);
    return lines;
  }
  // Paragraph; returns the y of the next free line.
  function para(parent, x, y, str, maxW, o = {}) {
    const size = o.size || 16;
    const lh = o.lh || size * 1.32;
    wrap(str, maxW, size, o.weight).forEach((l, i) => text(parent, x, y + i * lh, l, o));
    return y + wrap(str, maxW, size, o.weight).length * lh;
  }

  function arrowHead(parent, x, y, size, fill, dir = 1) {
    el("path", { d: `M${x},${y - size}L${x + dir * size * 1.2},${y}L${x},${y + size}Z`, fill }, parent);
  }

  // ── Sensor symbols ────────────────────────────────────────────────────────
  const sym = {
    working(g, x, y, r = 7) {
      el("circle", { cx: x, cy: y, r, fill: C.blue, stroke: C.white, "stroke-width": 2 }, g);
    },
    intermittent(g, x, y, r = 7) {
      el("circle", { cx: x, cy: y, r: r + 6, fill: "none", stroke: C.blue, "stroke-width": 2, "stroke-dasharray": "3 3" }, g);
      el("circle", { cx: x, cy: y, r, fill: C.blue, stroke: C.white, "stroke-width": 2 }, g);
    },
    silent(g, x, y, r = 7) {
      el("circle", { cx: x, cy: y, r: r + 3, fill: C.white, stroke: C.text2, "stroke-width": 2 }, g);
      text(g, x, y + 5, "?", { size: 14, weight: 700, fill: C.text2, anchor: "middle" });
    },
    lost(g, x, y, r = 7) {
      el("circle", { cx: x, cy: y, r: r + 3, fill: C.white, stroke: C.text2, "stroke-width": 2 }, g);
      const d = r * 0.75;
      el("path", { d: `M${x - d},${y - d}L${x + d},${y + d}M${x + d},${y - d}L${x - d},${y + d}`, stroke: C.red, "stroke-width": 3, "stroke-linecap": "round" }, g);
    },
    unsensed(g, x, y, r = 7) {
      el("circle", { cx: x, cy: y, r: r + 3, fill: "none", stroke: C.text3, "stroke-width": 2, "stroke-dasharray": "4 3" }, g);
    },
    // Output state: an estimate (fill = level) with a confidence halo.
    estimate(g, x, y, level, halo, r = 7) {
      el("circle", { cx: x, cy: y, r: r + halo, fill: C.blue, "fill-opacity": 0.14 }, g);
      el("circle", { cx: x, cy: y, r, fill: level, stroke: C.white, "stroke-width": 2 }, g);
    },
    alert(g, x, y, s = 9) {
      el("path", { d: `M${x},${y - s}L${x + s * 1.1},${y + s * 0.8}L${x - s * 1.1},${y + s * 0.8}Z`, fill: C.orange, stroke: C.white, "stroke-width": 1.5, "stroke-linejoin": "round" }, g);
      text(g, x, y + s * 0.6, "!", { size: s * 1.25, weight: 700, fill: C.white, anchor: "middle" });
    },
  };

  // Sequential blue ramp for estimated activity.
  function ramp(t) {
    const a = [210, 226, 241], b = [0, 48, 120];
    const c = a.map((v, i) => Math.round(v + (b[i] - v) * t));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  }

  // ── Panels ────────────────────────────────────────────────────────────────
  function mapFrame(g, x, y, w, h, caption) {
    el("rect", { x, y, width: w, height: h, fill: C.panel, rx: 0 }, g);
    if (caption) text(g, x + 14, y + h - 14, caption, { size: 13, fill: C.text3 });
    return (p) => [x + 30 + p[0] * (w - 60), y + 26 + p[1] * (h - 70)];
  }

  function realityMap(g, x, y, w, h, N, caption) {
    const pos = mapFrame(g, x, y, w, h, caption);
    const state = (i) => Object.keys(N.states).find((k) => N.states[k].includes(i)) || "working";
    N.unsensed.forEach((p) => sym.unsensed(g, ...pos(p)));
    N.sensors.forEach((p, i) => sym[state(i)](g, ...pos(p)));
  }

  function pictureMap(g, x, y, w, h, N, caption) {
    const pos = mapFrame(g, x, y, w, h, caption);
    const all = [
      ...N.sensors.map((p, i) => ({ p, kind: Object.keys(N.states).find((k) => N.states[k].includes(i)) || "working" })),
      ...N.unsensed.map((p) => ({ p, kind: "unsensed" })),
    ];
    // The model links each place to its nearest neighbours.
    const seen = new Set();
    all.forEach((a, i) => {
      all.map((b, j) => ({ j, d: Math.hypot(a.p[0] - b.p[0], a.p[1] - b.p[1]) }))
        .filter((o) => o.j !== i).sort((m, n) => m.d - n.d).slice(0, 3)
        .forEach(({ j }) => {
          const key = [i, j].sort().join("-");
          if (seen.has(key)) return;
          seen.add(key);
          const [x1, y1] = pos(a.p), [x2, y2] = pos(all[j].p);
          el("line", { x1, y1, x2, y2, stroke: C.blue, "stroke-opacity": 0.25, "stroke-width": 1.5 }, g);
        });
    });
    const HALO = { working: 3, intermittent: 7, silent: 12, lost: 13, unsensed: 15 };
    all.forEach((a, i) => {
      // Smooth pseudo-activity surface: busier towards the centre-right of the field.
      const t = Math.max(0.12, Math.min(0.95, 1 - Math.hypot(a.p[0] - 0.62, a.p[1] - 0.6) * 1.3));
      sym.estimate(g, ...pos(a.p), ramp(t), HALO[a.kind]);
    });
    all.forEach((a) => { if (a.kind === "lost") { const [px, py] = pos(a.p); sym.alert(g, px + 13, py - 13, 8); } });
  }

  // ── Mini icons for the cards ──────────────────────────────────────────────
  const icon = {
    gaps(g, x, y, col) { // series with a gap the model reads, not fills
      const pts = [[0, 14], [8, 6], [16, 10], [24, 2]];
      pts.forEach(([dx, dy]) => el("circle", { cx: x + dx, cy: y + dy, r: 3, fill: col }, g));
      el("rect", { x: x + 30, y: y - 4, width: 16, height: 22, fill: "none", stroke: col, "stroke-width": 1.5, "stroke-dasharray": "3 2" }, g);
      [[52, 8], [60, 4]].forEach(([dx, dy]) => el("circle", { cx: x + dx, cy: y + dy, r: 3, fill: col }, g));
    },
    neighbours(g, x, y, col) {
      const c = [x + 32, y + 8];
      [[x + 6, y - 2], [x + 58, y - 2], [x + 14, y + 22], [x + 52, y + 22]].forEach(([a, b]) => {
        el("line", { x1: a, y1: b, x2: c[0], y2: c[1], stroke: col, "stroke-width": 1.5 }, g);
        el("circle", { cx: a, cy: b, r: 4, fill: col }, g);
      });
      el("circle", { cx: c[0], cy: c[1], r: 6, fill: "none", stroke: col, "stroke-width": 2, "stroke-dasharray": "3 2" }, g);
    },
    unsure(g, x, y, col, bandCol) {
      el("path", { d: `M${x},${y + 10}L${x + 26},${y + 8}L${x + 64},${y - 6}L${x + 64},${y + 26}L${x + 26},${y + 12}Z`, fill: bandCol || col, "fill-opacity": 0.3 }, g);
      el("path", { d: `M${x},${y + 10}L${x + 26},${y + 10}L${x + 64},${y + 10}`, stroke: col, "stroke-width": 2, fill: "none" }, g);
    },
  };

  // ── Main ──────────────────────────────────────────────────────────────────
  window.drawSchematic = function () {
    const K = window.SCHEMATIC_CONTENT;
    const N = window.SCHEMATIC_NODES;
    const svg = document.querySelector("svg");
    svg.setAttribute("xmlns", NS);
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("width", W);
    svg.setAttribute("height", H);
    svg.setAttribute("font-family", FONT);
    const root = el("g", {}, svg);
    el("rect", { x: 0, y: 0, width: W, height: H, fill: C.white }, root);

    // Title block with a GOV.UK-style accent bar.
    el("rect", { x: 40, y: 34, width: 8, height: 64, fill: C.blue }, root);
    text(root, 64, 62, K.title, { size: 30, weight: 700 });
    para(root, 64, 90, K.subtitle, 1060, { size: 16, fill: C.text2 });

    const colW = 340, gap = 50, top = 140;
    const colX = [40, 40 + colW + gap, 40 + 2 * (colW + gap)];

    // Stage headers
    K.stages.forEach((s, i) => {
      const x = colX[i];
      el("circle", { cx: x + 16, cy: top + 10, r: 16, fill: i === 1 ? C.darkBlue : C.blue }, root);
      text(root, x + 16, top + 16, String(i + 1), { size: 17, weight: 700, fill: C.white, anchor: "middle" });
      text(root, x + 42, top + 17, s.title, { size: 21, weight: 700 });
      para(root, x, top + 48, s.lede, colW, { size: 15, fill: C.text2, lh: 19 });
    });

    const mapY = 230, mapH = 250;
    // Column 1: the messy reality
    realityMap(root, colX[0], mapY, colW, mapH, N, K.mapCaption);
    // Column 3: the complete picture
    pictureMap(root, colX[2], mapY, colW, mapH, N, K.pictureCaption);

    // Column 2: the resilient model
    const eng = el("g", {}, root);
    const ey = mapY, eh = 540;
    el("rect", { x: colX[1], y: ey, width: colW, height: eh, fill: C.darkBlue }, eng);
    text(eng, colX[1] + 22, ey + 36, K.engine.title, { size: 20, weight: 700, fill: C.white });
    para(eng, colX[1] + 22, ey + 60, K.engine.lede, colW - 44, { size: 14, fill: "#c8d9ec", lh: 18 });
    const icons = [icon.gaps, icon.neighbours, icon.unsure];
    K.engine.principles.forEach((p, i) => {
      const py = ey + 128 + i * 122;
      el("line", { x1: colX[1] + 22, x2: colX[1] + colW - 22, y1: py - 16, y2: py - 16, stroke: C.white, "stroke-opacity": 0.2 }, eng);
      icons[i](eng, colX[1] + 22, py + 6, C.white, C.lightBlue);
      text(eng, colX[1] + 102, py + 10, p.title, { size: 17, weight: 700, fill: C.white });
      para(eng, colX[1] + 102, py + 32, p.text, colW - 124, { size: 14, fill: "#c8d9ec", lh: 18 });
    });
    el("rect", { x: colX[1] + 22, y: ey + eh - 54, width: colW - 44, height: 34, fill: C.white, "fill-opacity": 0.1 }, eng);
    text(eng, colX[1] + colW / 2, ey + eh - 32, K.engine.footnote, { size: 14, weight: 700, fill: C.white, anchor: "middle" });

    // Flow arrows between the columns, level with the maps.
    [[colX[0] + colW, colX[1]], [colX[1] + colW, colX[2]]].forEach(([a, b], i) => {
      const ay = mapY + mapH / 2;
      el("rect", { x: a + 6, y: ay - 7, width: b - a - 24, height: 14, fill: i === 0 ? C.rule : C.blue }, root);
      arrowHead(root, b - 20, ay, 14, i === 0 ? C.rule : C.blue);
    });

    // Lists under the two maps
    function list(x, y, items, kind) {
      items.forEach((it, i) => {
        const iy = y + i * 66;
        const g = el("g", {}, root);
        if (kind === "reality") sym[it.symbol](g, x + 14, iy + 8);
        else drawOutputIcon(g, x + 14, iy + 8, it.symbol);
        text(g, x + 40, iy + 8, it.title, { size: 15, weight: 700 });
        if (it.tag) {
          // Right-aligned tag on the title line; titles are kept short enough to leave room.
          const tw = measure(it.tag.toUpperCase(), 10, 700) + 12;
          el("rect", { x: x + colW - tw, y: iy - 5, width: tw, height: 17, fill: C.paleBlue }, g);
          text(g, x + colW - tw / 2, iy + 7.5, it.tag.toUpperCase(), { size: 10, weight: 700, fill: C.darkBlue, anchor: "middle", spacing: "0.05em" });
        }
        para(g, x + 40, iy + 27, it.text, colW - 40, { size: 13.5, fill: C.text2, lh: 17 });
      });
    }
    function drawOutputIcon(g, x, y, s) {
      if (s === "filled") sym.estimate(g, x, y, ramp(0.6), 6, 6);
      if (s === "between") { sym.estimate(g, x, y, ramp(0.4), 7, 6); el("circle", { cx: x, cy: y, r: 10, fill: "none", stroke: C.text3, "stroke-width": 1.5, "stroke-dasharray": "3 2" }, g); }
      if (s === "forecast") {
        el("path", { d: `M${x - 13},${y + 1}L${x - 2},${y + 1}L${x + 13},${y - 9}L${x + 13},${y + 11}L${x - 2},${y + 3}Z`, fill: C.blue, "fill-opacity": 0.25 }, g);
        el("path", { d: `M${x - 13},${y + 2}L${x + 13},${y + 1}`, stroke: C.blue, "stroke-width": 2.5 }, g);
      }
      if (s === "alert") sym.alert(g, x, y, 10);
    }
    list(colX[0], mapY + mapH + 26, K.failures, "reality");
    list(colX[2], mapY + mapH + 26, K.outputs, "outputs");

    // ── Bottom band: graceful degradation + audience box ───────────────────
    const by = 822;
    el("line", { x1: 40, x2: W - 40, y1: by - 18, y2: by - 18, stroke: C.rule }, root);
    const cw = 600, ch = 100;
    text(root, 40, by + 4, K.degradation.title, { size: 17, weight: 700 });
    const gx = 40, gy = by + 18, gw = 300;
    // axes
    el("line", { x1: gx, x2: gx + gw, y1: gy + ch, y2: gy + ch, stroke: C.rule, "stroke-width": 1.5 }, root);
    el("line", { x1: gx, x2: gx, y1: gy, y2: gy + ch, stroke: C.rule, "stroke-width": 1.5 }, root);
    text(root, gx + gw, gy + ch + 16, K.degradation.xLabel, { size: 12, fill: C.text3, anchor: "end" });
    text(root, gx + 6, gy + 10, K.degradation.yLabel, { size: 12, fill: C.text3 });
    // resilient: gentle decline with widening band
    const X = (t) => gx + 10 + t * (gw - 20);
    const Y = (v) => gy + ch - 6 - v * (ch - 26);
    const res = (t) => 0.92 - 0.38 * t * t;
    const band = (t) => 0.04 + 0.22 * t;
    let up = "", lo = "";
    for (let t = 0; t <= 1.0001; t += 0.05) { up += `${up ? "L" : "M"}${X(t)},${Y(Math.min(1, res(t) + band(t)))}`; }
    for (let t = 1; t >= -0.0001; t -= 0.05) { lo += `L${X(t)},${Y(Math.max(0, res(t) - band(t)))}`; }
    el("path", { d: up + lo + "Z", fill: C.blue, "fill-opacity": 0.16 }, root);
    let rl = "";
    for (let t = 0; t <= 1.0001; t += 0.05) rl += `${rl ? "L" : "M"}${X(t)},${Y(res(t))}`;
    el("path", { d: rl, fill: "none", stroke: C.blue, "stroke-width": 3 }, root);
    // brittle: holds, then falls off a cliff
    el("path", { d: `M${X(0)},${Y(0.94)}L${X(0.38)},${Y(0.93)}C${X(0.46)},${Y(0.9)} ${X(0.48)},${Y(0.1)} ${X(0.58)},${Y(0.04)}L${X(1)},${Y(0.03)}`, fill: "none", stroke: C.orange, "stroke-width": 3 }, root);
    // legend to the right of the chart
    const lx = gx + gw + 24;
    [[C.orange, K.degradation.brittle], [C.blue, K.degradation.resilient]].forEach(([col, s], i) => {
      const ly = gy + 10 + i * 62;
      el("line", { x1: lx, x2: lx + 22, y1: ly - 5, y2: ly - 5, stroke: col, "stroke-width": 3 }, root);
      para(root, lx + 30, ly, s, cw - gw - 64, { size: 13.5, fill: C.text2, lh: 17 });
    });

    // Audience box
    const ax = 40 + cw + 30, aw = W - 40 - ax;
    // Box height follows its content (chips may wrap onto a second row).
    const boxBg = el("rect", { x: ax, y: by - 6, width: aw, fill: C.paleBlue }, root);
    const boxBar = el("rect", { x: ax, y: by - 6, width: 6, fill: C.blue }, root);
    let boxEnd;
    text(root, ax + 22, by + 20, K.audience.title, { size: 17, weight: 700 });
    if (K.audience.chips) {
      let cx = ax + 22, cy = by + 40;
      K.audience.chips.forEach((c) => {
        const tw = measure(c, 13.5, 700) + 20;
        if (cx + tw > ax + aw - 16) { cx = ax + 22; cy += 32; }
        el("rect", { x: cx, y: cy, width: tw, height: 24, fill: C.white, stroke: C.blue, "stroke-width": 1.5 }, root);
        text(root, cx + tw / 2, cy + 16.5, c, { size: 13.5, weight: 700, fill: C.darkBlue, anchor: "middle" });
        cx += tw + 8;
      });
      boxEnd = para(root, ax + 22, cy + 46, K.audience.text, aw - 44, { size: 13.5, fill: C.text2, lh: 17 });
    } else {
      boxEnd = para(root, ax + 22, by + 44, K.audience.text, aw - 44, { size: 14, fill: C.text2, lh: 18 });
    }
    const boxH = Math.max(ch + 44, boxEnd - (by - 6) + 4);
    boxBg.setAttribute("height", boxH);
    boxBar.setAttribute("height", boxH);

    text(root, 40, H - 16, K.footer, { size: 11.5, fill: C.text3 });
    if (measurer) measurer.remove();
  };
})();

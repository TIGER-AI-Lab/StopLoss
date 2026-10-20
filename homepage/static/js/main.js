/* StopLoss project page: theme toggle, result charts, results table and copy buttons. */
(function () {
  "use strict";

  // ---------- results of the paper ----------

  // Table 1: [family, method, ΔCPS (%), monitoring ($), tier AUC, success retained (%), failure capture (%)]
  var SETTINGS = {
    vl: { bench: "SWE-bench Verified", exec: "GPT-5.6 Luna", runs: 425, successes: 306, failures: 119, rows: [
      ["sl", "Qwen2.5-7B-Instruct", -4.5, 0.44, 0.622, 99.0, 5.0],
      ["sl", "Qwen3-8B", -1.8, 0.43, 0.605, 97.7, 5.0],
      ["sl", "Qwen2.5-Coder-7B-Instruct", -5.2, 0.35, 0.612, 99.0, 5.9],
      ["judge", "GPT-5.6 Sol", 116.8, 33.20, 0.633, 91.8, 20.2],
      ["judge", "GPT-5.6 Terra", 38.6, 12.90, 0.623, 89.5, 24.4],
      ["judge", "GPT-5.6 Luna", 8.4, 4.18, 0.600, 90.8, 14.3]
    ]},
    vs: { bench: "SWE-bench Verified", exec: "Claude Sonnet 4.6", runs: 425, successes: 326, failures: 99, rows: [
      ["sl", "Qwen2.5-7B-Instruct", 0.2, 0.44, 0.561, 100.0, 0.0],
      ["sl", "Qwen3-8B", -1.1, 0.42, 0.563, 99.7, 2.0],
      ["sl", "Qwen2.5-Coder-7B-Instruct", 0.1, 0.35, 0.561, 100.0, 0.0],
      ["judge", "Claude Opus 5", 34.7, 127.13, 0.747, 95.7, 32.3],
      ["judge", "Claude Opus 4.6", 11.6, 37.07, 0.602, 99.1, 0.0],
      ["judge", "Claude Sonnet 4.6", 10.3, 37.97, 0.602, 97.5, 3.0]
    ]},
    pl: { bench: "SWE-bench Pro", exec: "GPT-5.6 Luna", runs: 213, successes: 144, failures: 69, rows: [
      ["sl", "Qwen2.5-7B-Instruct", -24.2, 0.20, 0.614, 72.2, 43.5],
      ["sl", "Qwen3-8B", -25.7, 0.21, 0.621, 84.0, 37.7],
      ["sl", "Qwen2.5-Coder-7B-Instruct", -30.9, 0.15, 0.620, 74.3, 50.7],
      ["judge", "GPT-5.6 Sol", 59.3, 15.36, 0.657, 86.8, 29.0],
      ["judge", "GPT-5.6 Terra", 24.9, 6.44, 0.651, 79.2, 27.5],
      ["judge", "GPT-5.6 Luna", 0.0, 2.03, 0.639, 87.5, 14.5]
    ]},
    ps: { bench: "SWE-bench Pro", exec: "Claude Sonnet 4.6", runs: 213, successes: 107, failures: 106, rows: [
      ["sl", "Qwen2.5-7B-Instruct", -30.6, 0.30, 0.579, 91.6, 19.8],
      ["sl", "Qwen3-8B", -48.1, 0.23, 0.608, 83.2, 31.1],
      ["sl", "Qwen2.5-Coder-7B-Instruct", -34.5, 0.23, 0.578, 89.7, 19.8],
      ["judge", "Claude Opus 5", -26.5, 34.59, 0.695, 95.3, 31.1],
      ["judge", "Claude Opus 4.6", -17.2, 23.98, 0.614, 97.2, 15.1],
      ["judge", "Claude Sonnet 4.6", -29.0, 19.84, 0.598, 83.2, 23.6]
    ]}
  };

  // Table 2: [backbone, untuned tier AUC, trained-ensemble tier AUC, gain as printed in the paper]
  var AUC = {
    vl: { step: 0.489, rows: [["Qwen2.5-7B-Instruct", 0.491, 0.622, 0.131], ["Qwen3-8B", 0.504, 0.605, 0.100], ["Qwen2.5-Coder-7B-Instruct", 0.437, 0.612, 0.175]] },
    vs: { step: 0.471, rows: [["Qwen2.5-7B-Instruct", 0.512, 0.561, 0.049], ["Qwen3-8B", 0.481, 0.563, 0.082], ["Qwen2.5-Coder-7B-Instruct", 0.455, 0.561, 0.106]] },
    pl: { step: 0.528, rows: [["Qwen2.5-7B-Instruct", 0.452, 0.614, 0.162], ["Qwen3-8B", 0.507, 0.621, 0.114], ["Qwen2.5-Coder-7B-Instruct", 0.433, 0.620, 0.187]] },
    ps: { step: 0.513, rows: [["Qwen2.5-7B-Instruct", 0.511, 0.579, 0.068], ["Qwen3-8B", 0.516, 0.608, 0.092], ["Qwen2.5-Coder-7B-Instruct", 0.478, 0.578, 0.100]] }
  };

  var MINUS = "−";
  var SVGNS = "http://www.w3.org/2000/svg";

  function signed(v, digits) {
    return (v < 0 ? MINUS : "+") + Math.abs(v).toFixed(digits);
  }
  function shortName(name) {
    return name.replace("-Instruct", "");
  }
  function methodName(row) {
    return row[0] === "sl" ? "StopLoss · " + row[1] : row[1];
  }

  // ---------- theme ----------

  var root = document.documentElement;
  var darkQuery = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
  function currentTheme() {
    return root.getAttribute("data-theme") || (darkQuery && darkQuery.matches ? "dark" : "light");
  }
  var toggle = document.querySelector(".theme-toggle");
  if (toggle) {
    toggle.addEventListener("click", function () {
      var next = currentTheme() === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("stoploss-theme", next); } catch (e) { /* storage unavailable */ }
    });
  }

  var nav = document.querySelector(".nav");
  function onScroll() {
    if (nav) nav.classList.toggle("scrolled", window.scrollY > 8);
  }
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  // ---------- svg helpers ----------

  function svgEl(tag, attrs, parent) {
    var node = document.createElementNS(SVGNS, tag);
    for (var key in attrs) node.setAttribute(key, attrs[key]);
    if (parent) parent.appendChild(node);
    return node;
  }
  function svgText(parent, x, y, str, cls, anchor) {
    var node = svgEl("text", { x: x, y: y, "class": cls }, parent);
    if (anchor) node.setAttribute("text-anchor", anchor);
    node.textContent = str;
    return node;
  }

  var measureCtx = document.createElement("canvas").getContext("2d");
  function textWidth(str, px, weight) {
    measureCtx.font = (weight || 400) + " " + px + "px Inter, ui-sans-serif, system-ui, sans-serif";
    return measureCtx.measureText(str).width;
  }

  // Horizontal bar from baseline x0 to data end x1, rounded only at the data end.
  function barPath(x0, x1, y, h, radius) {
    var len = Math.abs(x1 - x0);
    if (len < 0.5) return null;
    var r = Math.min(radius, len, h / 2);
    var dir = x1 > x0 ? 1 : -1;
    var xe = x1 - dir * r;
    return "M" + x0 + "," + y + "H" + xe + "Q" + x1 + "," + y + " " + x1 + "," + (y + r) +
      "V" + (y + h - r) + "Q" + x1 + "," + (y + h) + " " + xe + "," + (y + h) + "H" + x0 + "Z";
  }

  // ---------- tooltip ----------

  var tooltip = document.getElementById("tooltip");
  function fillTooltip(title, sub, pairs) {
    tooltip.textContent = "";
    var t = document.createElement("div");
    t.className = "tt-title";
    t.textContent = title;
    tooltip.appendChild(t);
    if (sub) {
      var s = document.createElement("div");
      s.className = "tt-sub";
      s.textContent = sub;
      tooltip.appendChild(s);
    }
    pairs.forEach(function (pair) {
      var row = document.createElement("div");
      row.className = "tt-row";
      var label = document.createElement("span");
      label.textContent = pair[0];
      var value = document.createElement("b");
      value.textContent = pair[1];
      row.appendChild(label);
      row.appendChild(value);
      tooltip.appendChild(row);
    });
  }
  function placeTooltip(x, y) {
    tooltip.hidden = false;
    var w = tooltip.offsetWidth, h = tooltip.offsetHeight;
    var left = x + 16, top = y + 16;
    if (left + w > window.innerWidth - 8) left = x - w - 16;
    if (top + h > window.innerHeight - 8) top = y - h - 16;
    tooltip.style.left = Math.max(8, left) + "px";
    tooltip.style.top = Math.max(8, top) + "px";
  }
  function hideTooltip() {
    tooltip.hidden = true;
  }
  function attachTooltip(node, content) {
    node.addEventListener("pointerenter", function (e) { content(); placeTooltip(e.clientX, e.clientY); });
    node.addEventListener("pointermove", function (e) { placeTooltip(e.clientX, e.clientY); });
    node.addEventListener("pointerleave", hideTooltip);
    node.addEventListener("focus", function () {
      content();
      var box = node.getBoundingClientRect();
      placeTooltip(box.left + box.width * 0.55, box.top + box.height * 0.5);
    });
    node.addEventListener("blur", hideTooltip);
  }

  // ---------- ΔCPS small multiples ----------

  var DELTA_DOMAIN = [-55, 125];   // shared by the four panels
  var DELTA_TICKS = [-50, 0, 50, 100];

  function renderDelta(container, key) {
    var s = SETTINGS[key];
    var W = Math.max(260, Math.floor(container.clientWidth));
    var labels = s.rows.map(function (r) { return r[0] === "sl" ? shortName(r[1]) : r[1]; });
    var labelW = Math.ceil(Math.max.apply(null, labels.map(function (l) { return textWidth(l, 12.5, 400); }))) + 14;
    var valueW = 52, rowH = 28, barH = 14, groupGap = 10, axisH = 20, top = 2;
    var x0 = labelW, x1 = W - valueW - 6;
    var scale = function (v) { return x0 + (v - DELTA_DOMAIN[0]) / (DELTA_DOMAIN[1] - DELTA_DOMAIN[0]) * (x1 - x0); };
    var rowsBottom = top + s.rows.length * rowH + groupGap;
    var H = rowsBottom + axisH;
    var svg = svgEl("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H, role: "group",
      "aria-label": "Change in cost per retained solve on " + s.bench + " with " + s.exec });

    var compact = x1 - x0 < 220;
    DELTA_TICKS.forEach(function (t) {
      var x = Math.round(scale(t)) + 0.5;
      svgEl("line", { x1: x, x2: x, y1: top, y2: rowsBottom, "class": t === 0 ? "zero" : "grid" }, svg);
      var label = t === 0 ? "0" : compact ? (t < 0 ? MINUS : "") + Math.abs(t) : signed(t, 0) + "%";
      svgText(svg, x, rowsBottom + 15, label, "tick", "middle");
    });
    var divY = Math.round(top + 3 * rowH + groupGap / 2) + 0.5;
    svgEl("line", { x1: 0, x2: W, y1: divY, y2: divY, "class": "divider" }, svg);

    s.rows.forEach(function (r, i) {
      var y = top + i * rowH + (i >= 3 ? groupGap : 0);
      var g = svgEl("g", { "class": "row", tabindex: 0, role: "img",
        "aria-label": methodName(r) + ": " + signed(r[2], 1) + "% change in cost per retained solve" }, svg);
      svgEl("rect", { x: -6, y: y, width: W + 12, height: rowH, rx: 7, "class": "hover" }, g);
      svgText(g, 0, y + rowH / 2 + 4.5, labels[i], "row-label" + (r[0] === "judge" ? " judge" : ""));
      var d = barPath(Math.round(scale(0)), scale(r[2]), y + (rowH - barH) / 2, barH, 4);
      if (d) svgEl("path", { d: d, "class": "bar " + r[0] }, g);
      svgText(g, W, y + rowH / 2 + 4.5, signed(r[2], 1) + "%", "value" + (r[0] === "sl" ? " strong" : ""), "end");
      attachTooltip(g, function () {
        fillTooltip(methodName(r), s.bench + " · " + s.exec, [
          ["ΔCPS", signed(r[2], 1) + "%"],
          ["Monitoring", "$" + r[3].toFixed(2)],
          ["Tier AUC", r[4].toFixed(3)],
          ["Success retained", r[5].toFixed(1) + "%"],
          ["Failure capture", r[6].toFixed(1) + "%"]
        ]);
      });
    });
    container.textContent = "";
    container.appendChild(svg);
  }

  // ---------- AUC dumbbells ----------

  var AUC_DOMAIN = [0.42, 0.65];
  var AUC_TICKS = [0.45, 0.5, 0.55, 0.6];

  function renderAuc(container, key) {
    var a = AUC[key], s = SETTINGS[key];
    var W = Math.max(260, Math.floor(container.clientWidth));
    var labels = a.rows.map(function (r) { return shortName(r[0]); });
    var labelW = Math.ceil(Math.max.apply(null, labels.map(function (l) { return textWidth(l, 12.5, 400); }))) + 14;
    var valueW = 52, rowH = 30, top = 18, axisH = 20;
    var x0 = labelW + 6, x1 = W - valueW - 10;
    var scale = function (v) { return x0 + (v - AUC_DOMAIN[0]) / (AUC_DOMAIN[1] - AUC_DOMAIN[0]) * (x1 - x0); };
    var rowsBottom = top + a.rows.length * rowH + 4;
    var H = rowsBottom + axisH;
    var svg = svgEl("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H, role: "group",
      "aria-label": "Tier AUC of the untuned backbones and the trained ensembles on " + s.bench + " with " + s.exec });

    (x1 - x0 < 170 ? [0.5, 0.6] : AUC_TICKS).forEach(function (t) {
      var x = Math.round(scale(t)) + 0.5;
      svgEl("line", { x1: x, x2: x, y1: top, y2: rowsBottom, "class": "grid" }, svg);
      svgText(svg, x, rowsBottom + 15, t.toFixed(2), "tick", "middle");
    });
    var xs = Math.round(scale(a.step)) + 0.5;
    svgEl("line", { x1: xs, x2: xs, y1: top - 4, y2: rowsBottom, "class": "ref" }, svg);
    svgText(svg, xs, top - 8, "step count " + a.step.toFixed(3), "ref-label", "middle");
    svgText(svg, W, top - 8, "gain", "ref-label", "end");

    a.rows.forEach(function (r, i) {
      var cy = top + i * rowH + rowH / 2 + 2;
      var g = svgEl("g", { "class": "row", tabindex: 0, role: "img",
        "aria-label": r[0] + ": tier AUC " + r[1].toFixed(3) + " untuned, " + r[2].toFixed(3) + " trained ensemble" }, svg);
      svgEl("rect", { x: -6, y: cy - rowH / 2, width: W + 12, height: rowH, rx: 7, "class": "hover" }, g);
      svgText(g, 0, cy + 4.5, labels[i], "row-label");
      svgEl("line", { x1: scale(r[1]), x2: scale(r[2]), y1: cy, y2: cy, "class": "db-line" }, g);
      svgEl("circle", { cx: scale(r[1]), cy: cy, r: 5, "class": "db-base" }, g);
      svgEl("circle", { cx: scale(r[2]), cy: cy, r: 6, "class": "db-trained" }, g);
      svgText(g, W, cy + 4.5, "+" + r[3].toFixed(3), "gain", "end");
      attachTooltip(g, function () {
        fillTooltip(r[0], s.bench + " · " + s.exec, [
          ["Untuned backbone", r[1].toFixed(3)],
          ["Trained ensemble", r[2].toFixed(3)],
          ["Gain", "+" + r[3].toFixed(3)],
          ["Step-count baseline", a.step.toFixed(3)]
        ]);
      });
    });
    container.textContent = "";
    container.appendChild(svg);
  }

  var charts = Array.prototype.slice.call(document.querySelectorAll(".chart[data-chart]"));
  function renderAll() {
    charts.forEach(function (c) {
      if (c.getAttribute("data-chart") === "delta") renderDelta(c, c.getAttribute("data-setting"));
      else renderAuc(c, c.getAttribute("data-setting"));
    });
  }
  renderAll();
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(renderAll);
  var lastWidths = charts.map(function (c) { return c.clientWidth; });
  var pending = false;
  window.addEventListener("resize", function () {
    if (pending) return;
    pending = true;
    window.requestAnimationFrame(function () {
      pending = false;
      var widths = charts.map(function (c) { return c.clientWidth; });
      if (widths.some(function (w, i) { return w !== lastWidths[i]; })) {
        lastWidths = widths;
        hideTooltip();
        renderAll();
      }
    });
  });

  // ---------- results table ----------

  var body = document.getElementById("results-body");
  var meta = document.getElementById("table-meta");
  var tabs = Array.prototype.slice.call(document.querySelectorAll(".tab[data-setting]"));

  function bestIndex(rows, col, lower) {
    var vals = rows.map(function (r) { return r[col]; });
    var best = lower ? Math.min.apply(null, vals) : Math.max.apply(null, vals);
    return vals.map(function (v) { return v === best; });
  }

  function renderTable(key) {
    var s = SETTINGS[key];
    var best = {
      2: bestIndex(s.rows, 2, true), 3: bestIndex(s.rows, 3, true),
      5: bestIndex(s.rows, 5, false), 6: bestIndex(s.rows, 6, false)
    };
    var fmt = {
      2: function (v) { return signed(v, 1); }, 3: function (v) { return v.toFixed(2); },
      4: function (v) { return v.toFixed(3); }, 5: function (v) { return v.toFixed(1); }, 6: function (v) { return v.toFixed(1); }
    };
    body.textContent = "";
    s.rows.forEach(function (r, i) {
      var tr = document.createElement("tr");
      if (r[0] === "sl") tr.className = "sl";
      else if (i > 0 && s.rows[i - 1][0] === "sl") tr.className = "first-judge";
      var name = document.createElement("td");
      var who = document.createElement("span");
      who.className = "who";
      who.appendChild(document.createElement("i"));
      who.appendChild(document.createTextNode(methodName(r)));
      name.appendChild(who);
      tr.appendChild(name);
      for (var col = 2; col <= 6; col++) {
        var td = document.createElement("td");
        td.textContent = fmt[col](r[col]);
        if (best[col] && best[col][i]) td.className = "best";
        tr.appendChild(td);
      }
      body.appendChild(tr);
    });
    meta.textContent = s.bench + " with " + s.exec + ". Cost and stopping columns: " + s.runs + " stopping runs (" +
      s.successes + " successes, " + s.failures + " failures); tier AUC uses every run with a verdict.";
    tabs.forEach(function (t) {
      var on = t.getAttribute("data-setting") === key;
      t.setAttribute("aria-selected", on ? "true" : "false");
      t.tabIndex = on ? 0 : -1;
    });
  }

  tabs.forEach(function (t, i) {
    t.addEventListener("click", function () { renderTable(t.getAttribute("data-setting")); });
    t.addEventListener("keydown", function (e) {
      var step = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
      if (!step) return;
      e.preventDefault();
      var next = tabs[(i + step + tabs.length) % tabs.length];
      next.focus();
      renderTable(next.getAttribute("data-setting"));
    });
  });
  if (body) renderTable("vl");

  // ---------- copy buttons ----------

  function copyText(str) {
    if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(str);
    return new Promise(function (resolve, reject) {
      var area = document.createElement("textarea");
      area.value = str;
      area.setAttribute("readonly", "");
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.appendChild(area);
      area.select();
      try { document.execCommand("copy") ? resolve() : reject(new Error("copy failed")); }
      catch (e) { reject(e); }
      finally { document.body.removeChild(area); }
    });
  }
  Array.prototype.forEach.call(document.querySelectorAll(".copy[data-copy]"), function (button) {
    var label = button.querySelector("span");
    button.addEventListener("click", function () {
      var source = document.getElementById(button.getAttribute("data-copy"));
      copyText(source.textContent).then(function () {
        label.textContent = "Copied";
      }, function () {
        label.textContent = "Press Ctrl+C";
      }).then(function () {
        window.setTimeout(function () { label.textContent = "Copy"; }, 1600);
      });
    });
  });
})();

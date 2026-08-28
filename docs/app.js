/* Tree surrogates for multi-fidelity BO — static demo.
   Every number here comes from results/*.json in this repository. */
(function () {
  "use strict";

  // Palette. Ink for the published baseline, a muted tone for the black-box
  // tree, the accent reserved for the interpretable method.
  var INK = "#16130f", GREY = "#8a8275", ACCENT = "#bf3b1e";
  var RULE = "#d8d0c0", RULE2 = "#c3b9a5", FAINT = "#7d766a";
  var MONO = "'IBM Plex Mono', ui-monospace, Menlo, monospace";

  // ── Depth-3 distilled tree (matches figures/08_distilled_tree.png)
  var TREE = {
    f: "cy", t: 9.20, label: "Cell line average yield",
    yes: { f: "ph", t: 6.46, label: "pH",
      yes: { f: "tp", t: 33.41, label: "Temperature",
        yes: { leaf: 12.00 }, no: { leaf: 15.86 } },
      no:  { f: "tp", t: 33.49, label: "Temperature",
        yes: { leaf: 2.74 },  no: { leaf: 7.05 } } },
    no: { f: "ph", t: 6.47, label: "pH",
      yes: { f: "tp", t: 33.81, label: "Temperature",
        yes: { leaf: 21.24 }, no: { leaf: 26.00 } },
      no:  { f: "cb", t: 70.21, label: "Cell line best result",
        yes: { leaf: 18.04 }, no: { leaf: 9.08 } } }
  };
  var LEAVES = [2.74, 7.05, 9.08, 12.00, 15.86, 18.04, 21.24, 26.00];
  var LEAF_MAX = 26.00;
  var BEST_INPUT = { cy: 20, ph: 6.2, tp: 36, cb: 60 };  // reaches the 26.0 leaf

  var UNITS = { cy: "", ph: "", tp: " °C", cb: "" };
  var DEC = { cy: 1, ph: 2, tp: 1, cb: 1 };

  var METHODS = [
    { key: "gp", name: "GP baseline", colour: INK, mean: 56.34, std: 5.64, on: true,
      finals: [57.9, 59.3, 54.5, 46.7, 57.7, 45.6, 57.3, 59.0, 61.6, 63.9], interp: false,
      blurb: "The published Gaussian process engine. Calibrated uncertainty by construction, but it returns a number with no readable reason." },
    { key: "bark", name: "MF-BARK", colour: GREY, mean: 51.21, std: 9.71, on: true,
      finals: [54.4, 52.2, 59.0, 55.1, 28.3, 57.2, 50.3, 58.7, 37.8, 58.9], interp: false,
      blurb: "A tree kernel sampled by MCMC. Two recipes count as similar when the forest keeps sorting them into the same leaf." },
    { key: "ng", name: "NGBoost ensemble", colour: ACCENT, mean: 49.81, std: 9.11, on: true,
      finals: [45.9, 54.3, 57.4, 39.4, 58.8, 52.1, 49.6, 28.7, 59.3, 52.6], interp: true,
      blurb: "Ten boosted models whose disagreement supplies the uncertainty. Each member stays a readable tree." }
  ];

  var RANDOM_BASE = 3.49;
  var RULES = [
    { key: "c15", label: "Cell line 15", sub: "recommended by reading the rules", v: 73.80, kind: "rule" },
    { key: "bo",  label: "Best recipe the optimiser found", sub: "over a full campaign", v: 59.33, kind: "bo" },
    { key: "c19", label: "Cell line 19", sub: "recommended by reading the rules", v: 49.40, kind: "rule" },
    { key: "c22", label: "Cell line 22", sub: "recommended by reading the rules", v: 44.72, kind: "rule" },
    { key: "rnd", label: "A cell line picked at random", sub: "mean of 30 runs", v: 3.49, sd: 6.81, kind: "base" },
    { key: "avg", label: "An average cell line", sub: "mean of 15 runs", v: 3.11, sd: 2.45, kind: "base" }
  ];
  var RMAX = 80, rsel = "c15";

  var CONV = null, showBands = true, cur = 59, nGrid = 60, playing = false, timer = null;

  var $ = function (id) { return document.getElementById(id); };
  var esc = function (s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;"); };
  var eur = function (n) { return "€" + Math.round(n).toLocaleString("en-GB"); };

  // Every section renders independently. A missing element skips its own
  // block instead of throwing and blanking the whole page.
  function have() {
    for (var i = 0; i < arguments.length; i++) if (!$(arguments[i])) return false;
    return true;
  }
  function on(id, ev, fn) { var e = $(id); if (e) e.addEventListener(ev, fn); }
  function txt(id, s) { var e = $(id); if (e) e.textContent = s; }
  function html(id, s) { var e = $(id); if (e) e.innerHTML = s; }

  function tick(x, y, s, anchor, fill) {
    return '<text x="' + x + '" y="' + y + '" font-family=' + JSON.stringify(MONO) +
           ' font-size="10" letter-spacing=".06em" fill="' + (fill || FAINT) +
           '" text-anchor="' + (anchor || "middle") + '">' + s + "</text>";
  }

  // ── Tree ──────────────────────────────────────────────────────
  function evaluate(v) {
    var node = TREE, path = [];
    while (!("leaf" in node)) {
      var goYes = v[node.f] <= node.t;
      path.push({ label: node.label, feature: node.f, value: v[node.f], threshold: node.t, yes: goYes });
      node = goYes ? node.yes : node.no;
    }
    return { value: node.leaf, path: path, used: path.map(function (p) { return p.feature; }) };
  }

  function fmt(f, x) { return x.toFixed(DEC[f]) + UNITS[f]; }

  // The eight leaves drawn on a shared axis, so the discrete output space
  // and the position of the current one are both visible at a glance.
  function leafScale(active) {
    var W = 720, H = 74, L = 4, R = 4, base = 34, xmax = 28;
    var X = function (v) { return L + (v / xmax) * (W - L - R); };
    var s = ['<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="The eight possible outputs of the tree, from 2.7 to 26.0 grams per litre. The current one is ' + active.toFixed(1) + '.">'];
    s.push('<line x1="' + L + '" y1="' + base + '" x2="' + (W - R) + '" y2="' + base + '" stroke="' + RULE2 + '" stroke-width="1"/>');
    LEAVES.forEach(function (v) {
      var sel = Math.abs(v - active) < 1e-9, x = X(v);
      s.push('<line x1="' + x + '" y1="' + (sel ? base - 20 : base - 9) + '" x2="' + x + '" y2="' + base +
             '" stroke="' + (sel ? ACCENT : RULE2) + '" stroke-width="' + (sel ? 2.5 : 1.5) + '"/>');
      s.push('<text x="' + x + '" y="' + (base + 17) + '" font-family=' + JSON.stringify(MONO) +
             ' font-size="' + (sel ? 12 : 10.5) + '" font-weight="' + (sel ? 500 : 400) +
             '" fill="' + (sel ? ACCENT : FAINT) + '" text-anchor="middle">' + v.toFixed(1) + "</text>");
      if (sel) s.push('<circle cx="' + x + '" cy="' + (base - 20) + '" r="3.5" fill="' + ACCENT + '"/>');
    });
    s.push(tick(X(0), base + 36, "0", "middle", RULE2));
    s.push(tick(X(28), base + 36, "28 g/L", "middle", RULE2));
    s.push("</svg>");
    html("leafbar", s.join(""));
  }

  function render() {
    if (!have("cy", "ph", "tp", "cb", "pathlist")) return;
    var v = { cy: +$("cy").value, ph: +$("ph").value, tp: +$("tp").value, cb: +$("cb").value };
    ["cy", "ph", "tp", "cb"].forEach(function (f) { txt(f + "-o", fmt(f, v[f])); });

    var r = evaluate(v);
    var rank = LEAVES.slice().reverse().indexOf(r.value) + 1;

    txt("leafnum", r.value.toFixed(1));
    txt("leafrank", rank === 1
      ? "The highest of the eight possible outputs."
      : "Output " + rank + " of 8. The highest the tree can return is " + LEAF_MAX.toFixed(1) + " g/L.");

    var unused = r.used.indexOf("cb") === -1;
    if ($("cb-wrap")) $("cb-wrap").classList.toggle("dim", unused);
    txt("cb-h", unused ? "Not used on this branch" : "Used to reach this leaf");

    var ol = $("pathlist");
    ol.textContent = "";
    r.path.forEach(function (p, i) {
      var li = document.createElement("li");
      li.innerHTML = '<span class="n">' + (i + 1) + "</span><span>" + esc(p.label) +
        " <code>" + fmt(p.feature, p.value) + "</code>" + (p.yes ? " &le; " : " &gt; ") +
        "<code>" + p.threshold.toFixed(DEC[p.feature]) + "</code> " +
        '<span class="go ' + (p.yes ? "yes" : "no") + '">' + (p.yes ? "left" : "right") + "</span></span>";
      ol.appendChild(li);
    });

    leafScale(r.value);
  }

  // ── Convergence, animated ─────────────────────────────────────
  function convChart() {
    var box = $("convchart");
    if (!box) return;
    if (!CONV) { box.innerHTML = '<p class="empty">Convergence data not available.</p>'; return; }

    var W = 900, H = 400, L = 46, R = 14, T = 16, B = 48;
    var pw = W - L - R, ph = H - T - B;
    var grid = CONV.grid, xmax = grid[grid.length - 1], ymax = 70;
    var X = function (c) { return L + (c / xmax) * pw; };
    var Y = function (v) { return T + ph - (v / ymax) * ph; };
    var idx = Math.max(0, Math.min(cur, grid.length - 1));
    var s = ['<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Best pilot titer against cumulative experimental cost for each selected method, averaged over ten seeds.">'];

    // sparse horizontal rules, baseline in ink
    [20, 40, 60].forEach(function (y) {
      s.push('<line x1="' + L + '" y1="' + Y(y) + '" x2="' + (L + pw) + '" y2="' + Y(y) + '" stroke="' + RULE + '" stroke-width="1"/>');
    });
    [0, 20, 40, 60].forEach(function (y) { s.push(tick(L - 10, Y(y) + 3.5, y, "end")); });
    s.push('<line x1="' + L + '" y1="' + Y(0) + '" x2="' + (L + pw) + '" y2="' + Y(0) + '" stroke="' + INK + '" stroke-width="1.2"/>');

    for (var c = 0; c <= xmax; c += 10000) {
      s.push('<line x1="' + X(c) + '" y1="' + Y(0) + '" x2="' + X(c) + '" y2="' + (Y(0) + 5) + '" stroke="' + RULE2 + '"/>');
      s.push(tick(X(c), Y(0) + 20, c === 0 ? "0" : (c / 1000) + "k"));
    }
    s.push(tick(L + pw, Y(0) + 38, "CUMULATIVE COST (€)", "end", RULE2));
    s.push(tick(L - 10, T - 4, "g/L", "end", RULE2));

    var heads = [];
    METHODS.forEach(function (m) {
      var d = CONV.methods[m.key];
      if (!d || !m.on) return;
      var pts = [], hi = [], lo = [], last = null;
      for (var i = 0; i <= idx; i++) {
        if (d.mean[i] === null) continue;
        pts.push(X(grid[i]) + "," + Y(d.mean[i]));
        hi.push(X(grid[i]) + "," + Y(Math.min(ymax, d.mean[i] + (d.std[i] || 0))));
        lo.unshift(X(grid[i]) + "," + Y(Math.max(0, d.mean[i] - (d.std[i] || 0))));
        last = { x: X(grid[i]), y: Y(d.mean[i]) };
      }
      if (!pts.length) return;
      if (showBands && pts.length > 1) {
        s.push('<polygon points="' + hi.concat(lo).join(" ") + '" fill="' + m.colour + '" opacity=".10"/>');
      }
      s.push('<polyline points="' + pts.join(" ") + '" fill="none" stroke="' + m.colour +
             '" stroke-width="1.9" stroke-linejoin="round" stroke-linecap="round"/>');
      heads.push({ m: m, p: last });
    });

    s.push('<line x1="' + X(grid[idx]) + '" y1="' + T + '" x2="' + X(grid[idx]) + '" y2="' + Y(0) +
           '" stroke="' + INK + '" stroke-width="1" stroke-dasharray="2 4" opacity=".5"/>');
    heads.forEach(function (h) {
      s.push('<circle cx="' + h.p.x + '" cy="' + h.p.y + '" r="4" fill="' + h.m.colour +
             '" stroke="#f6f3ec" stroke-width="1.5"/>');
    });
    s.push("</svg>");
    box.innerHTML = s.join("");

    html("live", METHODS.filter(function (m) { return m.on; }).map(function (m) {
      var d = CONV.methods[m.key];
      var v = d && d.mean[idx];
      return '<span class="lv" style="--c:' + m.colour + '"><b>' + m.name + "</b>" +
             (v === null || v === undefined ? "<i>—</i>" : "<i>" + v.toFixed(1) + "</i>") + "</span>";
    }).join(""));
    txt("scrubout", eur(grid[idx]));
    if ($("scrub") && $("scrub").value !== String(idx)) $("scrub").value = idx;
  }

  function setPlaying(go) {
    playing = go;
    if ($("play")) $("play").classList.toggle("on", go);
    txt("playlab", go ? "Pause" : (cur >= nGrid - 1 ? "Replay" : "Play"));
    if (timer) { clearInterval(timer); timer = null; }
    if (go) {
      timer = setInterval(function () {
        cur++;
        if (cur >= nGrid - 1) { cur = nGrid - 1; convChart(); setPlaying(false); return; }
        convChart();
      }, 90);
    }
  }

  function toggles() {
    if (!$("toggles")) return;
    $("toggles").innerHTML = METHODS.map(function (m) {
      return '<button type="button" class="tg' + (m.on ? " on" : "") + '" data-k="' + m.key +
        '" aria-pressed="' + m.on + '" style="--c:' + m.colour + '"><span class="sw"></span>' + m.name + "</button>";
    }).join("");
    Array.prototype.forEach.call($("toggles").querySelectorAll(".tg"), function (b) {
      b.addEventListener("click", function () {
        var m = METHODS.filter(function (x) { return x.key === b.dataset.k; })[0];
        if (m.on && METHODS.filter(function (x) { return x.on; }).length === 1) return;
        m.on = !m.on;
        b.classList.toggle("on", m.on);
        b.setAttribute("aria-pressed", String(m.on));
        convChart();
      });
    });
  }

  // ── Final performance (always all three) ──────────────────────
  function finalChart() {
    if (!$("chart")) return;
    var W = 900, ROW = 78, L = 0, R = 0, T = 26, B = 34;
    var H = T + METHODS.length * ROW + B;
    var pw = W - L - R, xmax = 70;
    var X = function (v) { return L + (v / xmax) * pw; };
    var s = ['<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Final pilot titer by method. GP baseline 56.3, MF-BARK 51.2, NGBoost ensemble 49.8 grams per litre, averaged over ten seeds.">'];

    METHODS.forEach(function (m, i) {
      var top = T + i * ROW, cy = top + 46;
      s.push('<line x1="' + L + '" y1="' + top + '" x2="' + (L + pw) + '" y2="' + top +
             '" stroke="' + (i === 0 ? INK : RULE) + '" stroke-width="' + (i === 0 ? 1.5 : 1) + '"/>');
      s.push('<text x="' + L + '" y="' + (top + 22) + '" font-family="\'IBM Plex Sans\',sans-serif" font-size="14" font-weight="500" fill="' + INK + '">' + m.name + "</text>");
      s.push('<text x="' + (L + pw) + '" y="' + (top + 22) + '" font-family=' + JSON.stringify(MONO) +
             ' font-size="15" font-weight="500" fill="' + m.colour + '" text-anchor="end">' + m.mean.toFixed(1) + "</text>");
      s.push('<line x1="' + X(m.mean - m.std) + '" y1="' + cy + '" x2="' + X(m.mean + m.std) + '" y2="' + cy +
             '" stroke="' + m.colour + '" stroke-width="1.5" opacity=".4"/>');
      m.finals.forEach(function (f) {
        s.push('<circle cx="' + X(f) + '" cy="' + cy + '" r="3.2" fill="' + m.colour + '" opacity=".55"/>');
      });
      s.push('<line x1="' + X(m.mean) + '" y1="' + (cy - 11) + '" x2="' + X(m.mean) + '" y2="' + (cy + 11) +
             '" stroke="' + m.colour + '" stroke-width="2.2"/>');
    });

    var base = T + METHODS.length * ROW;
    s.push('<line x1="' + L + '" y1="' + base + '" x2="' + (L + pw) + '" y2="' + base + '" stroke="' + INK + '" stroke-width="1.2"/>');
    for (var g = 0; g <= xmax; g += 10) {
      s.push('<line x1="' + X(g) + '" y1="' + base + '" x2="' + X(g) + '" y2="' + (base + 5) + '" stroke="' + RULE2 + '"/>');
      s.push(tick(X(g), base + 20, g));
    }
    s.push(tick(L + pw, base + 20, "FINAL PILOT TITER (g/L)", "end", RULE2));
    s.push("</svg>");
    html("chart", s.join(""));
  }

  function cards() {
    html("mcards", METHODS.map(function (m) {
      return '<div class="note"><h3><span class="sw" style="background:' + m.colour + '"></span>' +
        m.name + "</h3><p>" + m.blurb + '</p><span class="tag ' + (m.interp ? "yes" : "no") + '">' +
        (m.interp ? "Readable model" : "Black box") + "</span></div>";
    }).join(""));
  }

  // ── Rule validation ───────────────────────────────────────────
  function rulesUI() {
    if (!have("rchips", "rrows", "rcard")) return;

    $("rchips").innerHTML = RULES.map(function (r) {
      return '<button type="button" class="rc' + (r.key === rsel ? " on" : "") +
        '" data-k="' + r.key + '" aria-pressed="' + (r.key === rsel) + '">' + esc(r.label) + "</button>";
    }).join("");

    $("rrows").innerHTML = RULES.map(function (r, i) {
      return '<button type="button" class="rrow k-' + r.kind + (r.key === rsel ? " sel" : "") + '" data-k="' + r.key + '">' +
        '<span class="ri">' + ("0" + (i + 1)).slice(-2) + "</span>" +
        '<span class="rlab">' + esc(r.label) + "<small>" + esc(r.sub) + "</small></span>" +
        '<span class="rtrack"><i style="width:' + (r.v / RMAX * 100).toFixed(1) + '%"></i></span>' +
        '<span class="rv">' + r.v.toFixed(1) + "</span></button>";
    }).join("");

    Array.prototype.forEach.call(document.querySelectorAll(".rc,.rrow"), function (b) {
      b.addEventListener("click", function () { rsel = b.dataset.k; rulesUI(); });
    });

    var r = RULES.filter(function (x) { return x.key === rsel; })[0];
    txt("rname", r.label);
    txt("rsub", r.sub + (r.sd ? ", spread ±" + r.sd.toFixed(1) : ""));
    txt("rval", r.v.toFixed(1));
    $("rcard").className = "rcard k-" + r.kind;
    txt("rmul", r.key === "rnd"
      ? "This is the random baseline"
      : (r.v / RANDOM_BASE).toFixed(1) + "× the random baseline");
  }

  // ── Wiring ────────────────────────────────────────────────────
  ["cy", "ph", "tp", "cb"].forEach(function (id) { on(id, "input", render); });
  on("best", "click", function () {
    Object.keys(BEST_INPUT).forEach(function (k) { if ($(k)) $(k).value = BEST_INPUT[k]; });
    render();
  });
  on("bands", "change", function () { showBands = this.checked; convChart(); });
  on("play", "click", function () {
    if (playing) { setPlaying(false); return; }
    if (cur >= nGrid - 1) cur = 0;
    setPlaying(true);
  });
  on("scrub", "input", function () {
    setPlaying(false);
    cur = +this.value;
    txt("playlab", cur >= nGrid - 1 ? "Replay" : "Play");
    convChart();
  });

  [render, toggles, finalChart, cards, rulesUI].forEach(function (f) {
    try { f(); } catch (e) { if (window.console) console.error(e); }
  });

  fetch("data/convergence.json?v=3")
    .then(function (r) { return r.ok ? r.json() : null; })
    .then(function (j) {
      CONV = j;
      if (j) {
        nGrid = j.grid.length;
        cur = nGrid - 1;
        if ($("scrub")) $("scrub").max = nGrid - 1;
      }
      convChart();
    })
    .catch(function () { convChart(); });
})();

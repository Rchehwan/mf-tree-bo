/* Tree surrogates for multi-fidelity BO — static demo.
   Every number below is taken from results/*.json in this repository. */
(function () {
  "use strict";

  // ── Depth-3 distilled tree (the reported run; matches figures/08_distilled_tree.png)
  // Features: cy = clone average yield, ph = pH, tp = temperature, cb = clone best titer
  var TREE = {
    f: "cy", t: 9.20, label: "Clone average yield",
    yes: { f: "ph", t: 6.46, label: "pH",
      yes: { f: "tp", t: 33.41, label: "Temperature",
        yes: { leaf: 12.00 }, no: { leaf: 15.86 } },
      no:  { f: "tp", t: 33.49, label: "Temperature",
        yes: { leaf: 2.74 },  no: { leaf: 7.05 } } },
    no: { f: "ph", t: 6.47, label: "pH",
      yes: { f: "tp", t: 33.81, label: "Temperature",
        yes: { leaf: 21.24 }, no: { leaf: 26.00 } },
      no:  { f: "cb", t: 70.21, label: "Clone best titer",
        yes: { leaf: 18.04 }, no: { leaf: 9.08 } } }
  };
  var LEAVES = [12.00, 15.86, 2.74, 7.05, 21.24, 26.00, 18.04, 9.08];
  var LEAF_MIN = 2.74, LEAF_MAX = 26.00;

  var UNITS = { cy: "", ph: "", tp: " °C", cb: "" };
  var DEC   = { cy: 1, ph: 2, tp: 1, cb: 1 };

  // ── Reported results (results/*.json)
  var METHODS = [
    { key: "gp",  name: "GP baseline", colour: "#0f5fa8", mean: 56.34, std: 5.64,
      finals: [57.9, 59.3, 54.5, 46.7, 57.7, 45.6, 57.3, 59.0, 61.6, 63.9],
      mix: [8.7, 2.4, 17.4], interp: false,
      blurb: "The published Gaussian process engine. Calibrated uncertainty by construction, but returns a number with no readable reason." },
    { key: "bark", name: "MF-BARK", colour: "#6a5acd", mean: 51.21, std: 9.71,
      finals: [54.4, 52.2, 59.0, 55.1, 28.3, 57.2, 50.3, 58.7, 37.8, 58.9],
      mix: [58.6, 2.7, 12.7], interp: false,
      blurb: "A tree kernel sampled by MCMC. Two recipes are similar when the forest keeps sorting them into the same leaf." },
    { key: "ng", name: "NGBoost ensemble", colour: "#0f8f7a", mean: 49.81, std: 9.11,
      finals: [45.9, 54.3, 57.4, 39.4, 58.8, 52.1, 49.6, 28.7, 59.3, 52.6],
      mix: [9.3, 0.0, 18.7], interp: true,
      blurb: "Ten boosted models whose disagreement supplies the uncertainty. Each member stays a readable tree." }
  ];

  var VALIDATION = {
    recommended: [ { clone: 19, v: 49.40 }, { clone: 15, v: 73.80 }, { clone: 22, v: 44.72 } ],
    bo_best: 59.33, random: 3.49, avg_clone: 3.11
  };

  var $ = function (id) { return document.getElementById(id); };

  // ── Tree evaluation ────────────────────────────────────────────
  function evaluate(v) {
    var node = TREE, path = [], leafIndex = 0;
    while (!("leaf" in node)) {
      var x = v[node.f], goYes = x <= node.t;
      path.push({ label: node.label, feature: node.f, value: x, threshold: node.t, yes: goYes });
      leafIndex = (leafIndex << 1) | (goYes ? 0 : 1);
      node = goYes ? node.yes : node.no;
    }
    return { value: node.leaf, path: path, index: leafIndex, used: path.map(function (p) { return p.feature; }) };
  }

  function fmt(f, x) { return x.toFixed(DEC[f]) + UNITS[f]; }

  function shade(v) {
    var t = (v - LEAF_MIN) / (LEAF_MAX - LEAF_MIN);
    t = Math.max(0, Math.min(1, t));
    // pale olive (low titer) -> deep green (high titer)
    return "rgb(" + Math.round(46 + (1 - t) * 120) + "," +
                    Math.round(120 + (1 - t) * 55) + "," +
                    Math.round(72 + (1 - t) * 40) + ")";
  }

  function render() {
    var v = { cy: +$("cy").value, ph: +$("ph").value, tp: +$("tp").value, cb: +$("cb").value };
    ["cy", "ph", "tp", "cb"].forEach(function (f) { $(f + "-o").textContent = fmt(f, v[f]); });

    var r = evaluate(v);

    $("leafnum").textContent = r.value.toFixed(1);
    $("leafcard").style.background = shade(r.value);
    var rank = LEAVES.slice().sort(function (a, b) { return b - a; }).indexOf(r.value) + 1;
    $("leafrank").textContent = "Highest-yielding leaf: rank " + rank + " of 8";

    // clone-best control is only consulted on one branch
    $("cb-wrap").classList.toggle("dim", r.used.indexOf("cb") === -1);
    $("cb-h").textContent = r.used.indexOf("cb") === -1
      ? "Not used on this branch" : "Used to reach this leaf";

    var ol = $("pathlist");
    ol.textContent = "";
    r.path.forEach(function (p, i) {
      var li = document.createElement("li");
      var n = document.createElement("span");
      n.className = "n"; n.textContent = String(i + 1);
      var txt = document.createElement("span");
      var cmp = p.yes ? " ≤ " : " > ";
      txt.innerHTML = p.label + " <code>" + fmt(p.feature, p.value) + "</code>" + cmp +
        "<code>" + p.threshold.toFixed(DEC[p.feature]) + "</code> — " +
        (p.yes ? '<span class="yes">left</span>' : '<span class="no">right</span>');
      li.appendChild(n); li.appendChild(txt);
      ol.appendChild(li);
    });
  }

  // ── Comparison chart (inline SVG, no dependency) ───────────────
  function chart() {
    var W = 640, H = 300, L = 108, R = 22, T = 16, B = 42;
    var maxV = 70, plotW = W - L - R, plotH = H - T - B;
    var rowH = plotH / METHODS.length;
    var x = function (val) { return L + (val / maxV) * plotW; };
    var svg = ['<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Final pilot titer by method. GP baseline 56.3, MF-BARK 51.2, NGBoost ensemble 49.8 grams per litre, each averaged over ten seeds.">'];

    for (var g = 0; g <= maxV; g += 10) {
      svg.push('<line x1="' + x(g) + '" y1="' + T + '" x2="' + x(g) + '" y2="' + (T + plotH) + '" stroke="#eef1f6"/>');
      svg.push('<text x="' + x(g) + '" y="' + (H - 16) + '" font-size="11" fill="#6b7a90" text-anchor="middle">' + g + '</text>');
    }
    svg.push('<text x="' + (L + plotW / 2) + '" y="' + (H - 2) + '" font-size="11" fill="#6b7a90" text-anchor="middle">final pilot titer (g/L)</text>');

    METHODS.forEach(function (m, i) {
      var cy = T + rowH * i + rowH / 2, bh = 22;
      svg.push('<text x="' + (L - 10) + '" y="' + (cy + 4) + '" font-size="12.5" fill="#0e1726" text-anchor="end" font-weight="600">' + m.name + '</text>');
      svg.push('<rect x="' + L + '" y="' + (cy - bh / 2) + '" width="' + (x(m.mean) - L) + '" height="' + bh + '" rx="5" fill="' + m.colour + '" opacity=".22"/>');
      // std whisker
      svg.push('<line x1="' + x(m.mean - m.std) + '" y1="' + cy + '" x2="' + x(m.mean + m.std) + '" y2="' + cy + '" stroke="' + m.colour + '" stroke-width="2" opacity=".5"/>');
      m.finals.forEach(function (f) {
        svg.push('<circle cx="' + x(f) + '" cy="' + cy + '" r="3.4" fill="' + m.colour + '" opacity=".62"/>');
      });
      svg.push('<line x1="' + x(m.mean) + '" y1="' + (cy - bh / 2 - 3) + '" x2="' + x(m.mean) + '" y2="' + (cy + bh / 2 + 3) + '" stroke="' + m.colour + '" stroke-width="3"/>');
      svg.push('<text x="' + (x(m.mean) + 9) + '" y="' + (cy + 4) + '" font-size="12.5" font-weight="700" fill="' + m.colour + '">' + m.mean.toFixed(1) + '</text>');
    });
    svg.push('</svg>');
    $("chart").innerHTML = svg.join("");
  }

  function cards() {
    $("mcards").innerHTML = METHODS.map(function (m) {
      return '<div class="mcard"><h3><span class="dot" style="background:' + m.colour + '"></span>' + m.name + "</h3>" +
        "<p>" + m.blurb + "</p>" +
        '<span class="tag ' + (m.interp ? "yes" : "no") + '">' +
        (m.interp ? "Readable model" : "Black box") + "</span></div>";
    }).join("");
  }

  function validation() {
    var max = 80;
    var rows = VALIDATION.recommended.map(function (r) {
      return { lab: "Recommended cell line " + r.clone, sub: "chosen by reading the model", v: r.v, base: false };
    });
    rows.push({ lab: "Best recipe the optimiser found", sub: "over a full campaign", v: VALIDATION.bo_best, base: false });
    rows.push({ lab: "Randomly chosen cell line", sub: "mean of 30 runs", v: VALIDATION.random, base: true });
    rows.push({ lab: "Average cell line", sub: "mean of 15 runs", v: VALIDATION.avg_clone, base: true });

    $("valgrid").innerHTML = rows.map(function (r) {
      return '<div class="vrow' + (r.base ? " base" : "") + '">' +
        '<div class="lab">' + r.lab + "<small>" + r.sub + "</small></div>" +
        '<div class="val">' + r.v.toFixed(1) + " <span style=\"font-size:12px;font-weight:500;color:#6b7a90\">g/L</span></div>" +
        '<div class="vbar"><i style="width:' + Math.min(100, (r.v / max) * 100) + '%"></i></div></div>';
    }).join("");
  }

  ["cy", "ph", "tp", "cb"].forEach(function (id) {
    $(id).addEventListener("input", render);
  });

  render(); chart(); cards(); validation();
})();

/* ---------------------------------------------------------------------------
   Page behaviour: charts drawn from the study's own result files, a hero
   animation of the sparse matrix being reconstructed, and a PMF that actually
   trains in the browser.

   No chart library. Everything is hand-drawn SVG so the marks obey the same
   rules as the paper's matplotlib figures (thin marks, recessive grid, direct
   labels) and so the page stays a single self-contained file.
--------------------------------------------------------------------------- */

(() => {
  "use strict";

  const DATA = window.SITE_DATA || {};
  const R = DATA.results || {};
  const SVG_NS = "http://www.w3.org/2000/svg";

  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const SERIES = () => [1, 2, 3, 4, 5, 6].map((i) => css(`--series-${i}`));

  const fmt = (v, digits = 4) => (v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(digits));
  const el = (tag, attrs = {}, parent = null) => {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (parent) parent.appendChild(node);
    return node;
  };
  const html = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };

  /* ------------------------------------------------------------- tooltip */

  const tip = html("div", "tooltip");
  document.body.appendChild(tip);

  function showTip(event, text) {
    tip.textContent = text;
    tip.classList.add("show");
    const pad = 14;
    const rect = tip.getBoundingClientRect();
    let x = event.clientX + pad;
    let y = event.clientY + pad;
    if (x + rect.width > window.innerWidth - 8) x = event.clientX - rect.width - pad;
    if (y + rect.height > window.innerHeight - 8) y = event.clientY - rect.height - pad;
    tip.style.left = `${x}px`;
    tip.style.top = `${y}px`;
  }
  const hideTip = () => tip.classList.remove("show");

  function hoverable(node, text) {
    node.addEventListener("mousemove", (e) => showTip(e, text));
    node.addEventListener("mouseleave", hideTip);
    node.style.cursor = "crosshair";
  }

  /* --------------------------------------------------------- chart frame */

  let clipSeq = 0;

  function frame(mount, { width = 640, height = 260, pad = { t: 16, r: 18, b: 34, l: 46 } } = {}) {
    mount.innerHTML = "";
    const svg = el("svg", {
      viewBox: `0 0 ${width} ${height}`,
      class: "chart",
      preserveAspectRatio: "xMidYMid meet",
      role: "img",
    }, mount);

    // Every data mark is drawn inside this clip. A confidence band computed from
    // a stratum with three observations is wider than the whole rating scale,
    // and without a clip its path runs thousands of units past the viewBox and
    // paints over the rest of the page.
    const id = `plot-clip-${++clipSeq}`;
    const defs = el("defs", {}, svg);
    const clip = el("clipPath", { id }, defs);
    el("rect", {
      x: pad.l, y: pad.t - 2, width: width - pad.l - pad.r, height: height - pad.t - pad.b + 4,
    }, clip);

    const iw = width - pad.l - pad.r;
    const ih = height - pad.t - pad.b;
    return { svg, width, height, pad, iw, ih, clipId: id };
  }

  function axes(f, { xTicks, yTicks, xLabel, yLabel, yFmt = (v) => v.toFixed(2) }) {
    const { svg, pad, iw, ih } = f;
    for (const t of yTicks) {
      const y = pad.t + ih - t.p * ih;
      el("line", { x1: pad.l, y1: y, x2: pad.l + iw, y2: y, stroke: css("--grid"), "stroke-width": 1 }, svg);
      const label = el("text", {
        x: pad.l - 7, y: y + 3.5, "text-anchor": "end",
        fill: css("--ink-3"), "font-size": 10, "font-family": css("--sans"),
      }, svg);
      label.textContent = yFmt(t.v);
    }
    el("line", {
      x1: pad.l, y1: pad.t + ih, x2: pad.l + iw, y2: pad.t + ih,
      stroke: css("--axis"), "stroke-width": 1,
    }, svg);

    for (const t of xTicks) {
      const x = pad.l + t.p * iw;
      const label = el("text", {
        x, y: pad.t + ih + 15, "text-anchor": "middle",
        fill: css("--ink-3"), "font-size": 10, "font-family": css("--sans"),
      }, svg);
      label.textContent = t.label;
    }

    if (xLabel) {
      const node = el("text", {
        x: pad.l + iw / 2, y: f.height - 3, "text-anchor": "middle",
        fill: css("--ink-2"), "font-size": 10.5, "font-family": css("--sans"),
      }, svg);
      node.textContent = xLabel;
    }
    if (yLabel) {
      const node = el("text", {
        x: 11, y: pad.t + ih / 2, "text-anchor": "middle",
        transform: `rotate(-90 11 ${pad.t + ih / 2})`,
        fill: css("--ink-2"), "font-size": 10.5, "font-family": css("--sans"),
      }, svg);
      node.textContent = yLabel;
    }
  }

  const ticksFrom = (lo, hi, n = 5) =>
    Array.from({ length: n }, (_, i) => {
      const v = lo + ((hi - lo) * i) / (n - 1);
      return { v, p: (v - lo) / (hi - lo || 1) };
    });

  function legend(mount, entries) {
    const box = html("div", "legend");
    for (const [label, colour] of entries) {
      const span = html("span");
      const sw = html("span", "swatch");
      sw.style.background = colour;
      span.appendChild(sw);
      span.appendChild(document.createTextNode(label));
      box.appendChild(span);
    }
    mount.appendChild(box);
  }

  /* ------------------------------------------------- 1. model comparison */

  function chartComparison(mountId, payload, tableId) {
    const mount = document.getElementById(mountId);
    if (!mount || !payload) return void hideSection(mountId);

    const rows = [...payload.table].sort((a, b) => b.test_rmse - a.test_rmse);
    const best = Math.min(...rows.map((r) => r.test_rmse));
    const lo = 0.85;
    const hi = Math.max(...rows.map((r) => r.test_rmse + (r.test_rmse_ci95 || 0))) + 0.02;

    const rowH = 26;
    const f = frame(mount, { height: rows.length * rowH + 46, pad: { t: 10, r: 54, b: 30, l: 132 } });
    const { svg, pad, iw, ih } = f;
    const x = (v) => pad.l + ((v - lo) / (hi - lo)) * iw;

    for (const t of ticksFrom(lo, hi, 5)) {
      el("line", { x1: x(t.v), y1: pad.t, x2: x(t.v), y2: pad.t + ih, stroke: css("--grid"), "stroke-width": 1 }, svg);
      const label = el("text", {
        x: x(t.v), y: pad.t + ih + 15, "text-anchor": "middle",
        fill: css("--ink-3"), "font-size": 10, "font-family": css("--sans"),
      }, svg);
      label.textContent = t.v.toFixed(2);
    }

    rows.forEach((row, i) => {
      const isBaseline = /mean|Bias baseline/i.test(row.label);
      const y = pad.t + i * (ih / rows.length) + 4;
      const h = ih / rows.length - 8;
      const colour = row.test_rmse === best ? css("--series-3") : isBaseline ? css("--ink-3") : css("--series-1");

      const bar = el("rect", {
        x: pad.l, y, width: Math.max(1, x(row.test_rmse) - pad.l), height: h,
        fill: colour, rx: 2,
      }, svg);
      hoverable(bar, `${row.label}\nRMSE ${fmt(row.test_rmse)} ± ${fmt(row.test_rmse_ci95)}\nMAE ${fmt(row.test_mae)}`);

      if (row.test_rmse_ci95) {
        el("line", {
          x1: x(row.test_rmse - row.test_rmse_ci95), x2: x(row.test_rmse + row.test_rmse_ci95),
          y1: y + h / 2, y2: y + h / 2, stroke: css("--ink-2"), "stroke-width": 1,
        }, svg);
      }

      const name = el("text", {
        x: pad.l - 8, y: y + h / 2 + 3.5, "text-anchor": "end",
        fill: css("--ink"), "font-size": 11, "font-family": css("--sans"),
        "font-weight": row.test_rmse === best ? 640 : 400,
      }, svg);
      name.textContent = row.label;

      const value = el("text", {
        x: x(row.test_rmse) + 6, y: y + h / 2 + 3.5,
        fill: css("--ink-2"), "font-size": 10, "font-family": css("--mono"),
      }, svg);
      value.textContent = fmt(row.test_rmse);
    });

    if (tableId) buildComparisonTable(tableId, rows, best);
  }

  function buildComparisonTable(tableId, rows, best) {
    const mount = document.getElementById(tableId);
    if (!mount) return;
    const ordered = [...rows].sort((a, b) => a.test_rmse - b.test_rmse);
    const table = html("table");
    table.innerHTML =
      "<thead><tr><th>Model</th><th>Test RMSE</th><th>95% CI</th><th>Test MAE</th>" +
      "<th>Train RMSE</th><th>Fit (s)</th></tr></thead>";
    const body = html("tbody");
    for (const row of ordered) {
      const tr = html("tr");
      if (row.test_rmse === best) tr.classList.add("is-best");
      if (/mean|Bias baseline/i.test(row.label)) tr.classList.add("is-baseline");
      tr.innerHTML =
        `<td>${row.label}</td>` +
        `<td class="num">${fmt(row.test_rmse)}</td>` +
        `<td class="num">±${fmt(row.test_rmse_ci95)}</td>` +
        `<td class="num">${fmt(row.test_mae)}</td>` +
        `<td class="num">${fmt(row.train_rmse)}</td>` +
        `<td class="num">${fmt(row.fit_seconds, 1)}</td>`;
      body.appendChild(tr);
    }
    table.appendChild(body);
    mount.innerHTML = "";
    mount.appendChild(table);
  }

  /* -------------------------------------------------- 2. learning curves */

  function chartConvergence(mountId, legendId) {
    const mount = document.getElementById(mountId);
    const payload = R["convergence_ml-100k"];
    if (!mount || !payload) return void hideSection(mountId);

    const traces = Object.entries(payload.traces);
    const f = frame(mount, { height: 270, pad: { t: 14, r: 16, b: 38, l: 48 } });
    const { svg, pad, iw, ih } = f;

    const allVals = traces.flatMap(([, t]) => [...t.train_rmse, ...t.val_rmse]).filter(Number.isFinite);
    const lo = Math.min(...allVals) - 0.02;
    const hi = Math.min(Math.max(...allVals) + 0.02, 1.35);
    const maxEpoch = Math.max(...traces.map(([, t]) => Math.max(...t.epoch)));

    axes(f, {
      xTicks: ticksFrom(0, maxEpoch, 5).map((t) => ({ ...t, label: Math.round(t.v) })),
      yTicks: ticksFrom(lo, hi, 5),
      xLabel: "Epoch",
      yLabel: "RMSE",
      yFmt: (v) => v.toFixed(2),
    });

    const X = (e) => pad.l + (e / maxEpoch) * iw;
    const Y = (v) => pad.t + ih - ((v - lo) / (hi - lo)) * ih;
    const colours = SERIES();
    const entries = [];

    traces.forEach(([label, trace], i) => {
      const colour = colours[i % colours.length];
      entries.push([`${label} — validation`, colour]);

      const path = trace.epoch
        .map((e, k) => `${k ? "L" : "M"}${X(e).toFixed(1)},${Y(trace.val_rmse[k]).toFixed(1)}`)
        .join("");
      el("path", { d: path, fill: "none", stroke: colour, "stroke-width": 2 }, svg);

      if (i === 0) {
        const trainPath = trace.epoch
          .map((e, k) => `${k ? "L" : "M"}${X(e).toFixed(1)},${Y(trace.train_rmse[k]).toFixed(1)}`)
          .join("");
        el("path", {
          d: trainPath, fill: "none", stroke: colour, "stroke-width": 1.4,
          "stroke-dasharray": "4 3", opacity: 0.75,
        }, svg);
        entries.push([`${label} — training`, colour]);

        const bestIdx = trace.val_rmse.indexOf(Math.min(...trace.val_rmse.filter(Number.isFinite)));
        if (bestIdx >= 0) {
          el("line", {
            x1: X(trace.epoch[bestIdx]), y1: pad.t, x2: X(trace.epoch[bestIdx]), y2: pad.t + ih,
            stroke: css("--ink-3"), "stroke-width": 1, "stroke-dasharray": "3 3",
          }, svg);
          const note = el("text", {
            x: X(trace.epoch[bestIdx]) + 5, y: pad.t + 11,
            fill: css("--ink-2"), "font-size": 9.5, "font-family": css("--sans"),
          }, svg);
          note.textContent = `best epoch ${trace.epoch[bestIdx]}`;
        }
      }
    });

    const legendMount = document.getElementById(legendId);
    if (legendMount) { legendMount.innerHTML = ""; legend(legendMount, entries); }
  }

  /* ----------------------------------------------------- 3. line charts */

  function lineChart(mountId, legendId, spec) {
    const mount = document.getElementById(mountId);
    if (!mount || !spec || !spec.series.length || !spec.x.length) return void hideSection(mountId);

    const f = frame(mount, { height: spec.height || 250, pad: { t: 14, r: 20, b: 40, l: 50 } });
    const { svg, pad, iw, ih } = f;

    const values = spec.series.flatMap((s) => s.values).filter(Number.isFinite);
    const spread = Math.max(...values) - Math.min(...values);
    const lo = Math.min(...values) - spread * 0.12 - 0.002;
    const hi = Math.max(...values) + spread * 0.12 + 0.002;

    axes(f, {
      xTicks: spec.x.map((v, i) => ({ p: spec.x.length === 1 ? 0.5 : i / (spec.x.length - 1), label: spec.labels[i] })),
      yTicks: ticksFrom(lo, hi, 5),
      xLabel: spec.xLabel,
      yLabel: spec.yLabel,
      yFmt: spec.yFmt || ((v) => v.toFixed(3)),
    });

    const X = (i) => pad.l + (spec.x.length === 1 ? 0.5 : i / (spec.x.length - 1)) * iw;
    const Y = (v) => pad.t + ih - ((v - lo) / (hi - lo)) * ih;
    const plot = el("g", { "clip-path": `url(#${f.clipId})` }, f.svg);
    const colours = SERIES();
    const entries = [];

    spec.series.forEach((series, si) => {
      const colour = series.colour || colours[si % colours.length];
      entries.push([series.label, colour]);

      const points = series.values
        .map((v, i) => (Number.isFinite(v) ? [X(i), Y(v)] : null))
        .filter(Boolean);
      if (!points.length) return;

      if (series.band) {
        const upper = series.values.map((v, i) => [X(i), Y(v + (series.band[i] || 0))]);
        const lower = series.values.map((v, i) => [X(i), Y(v - (series.band[i] || 0))]).reverse();
        el("path", {
          d: [...upper, ...lower].map(([x, y], k) => `${k ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join("") + "Z",
          fill: colour, opacity: 0.14, stroke: "none",
        }, plot);
      }

      el("path", {
        d: points.map(([x, y], k) => `${k ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join(""),
        fill: "none", stroke: colour, "stroke-width": 2,
        "stroke-dasharray": series.dashed ? "5 3" : "none",
      }, plot);

      series.values.forEach((v, i) => {
        if (!Number.isFinite(v)) return;
        const dot = el("circle", {
          cx: X(i), cy: Y(v), r: 4, fill: colour,
          stroke: css("--surface"), "stroke-width": 1.5,
        }, plot);
        const note = spec.notes && spec.notes[i] ? `\n${spec.notes[i]}` : "";
        hoverable(dot, `${series.label}\n${spec.labels[i]}: ${fmt(v, spec.digits ?? 4)}${note}`);
      });
    });

    const legendMount = document.getElementById(legendId);
    if (legendMount) { legendMount.innerHTML = ""; legend(legendMount, entries); }
  }

  /* ---------------------------------------------------- 4. scalability */

  function chartScalability(mountId) {
    const mount = document.getElementById(mountId);
    const payload = R.scalability;
    if (!mount || !payload) return void hideSection(mountId);

    const rows = payload.table;
    const fit = payload.loglog_fit;
    const f = frame(mount, { height: 260, pad: { t: 16, r: 20, b: 40, l: 56 } });
    const { svg, pad, iw, ih } = f;

    const lx = rows.map((r) => Math.log10(r.n_train_ratings));
    const ly = rows.map((r) => Math.log10(r.seconds_per_epoch));
    const xlo = Math.min(...lx) - 0.12, xhi = Math.max(...lx) + 0.12;
    const ylo = Math.min(...ly) - 0.2, yhi = Math.max(...ly) + 0.2;

    const X = (v) => pad.l + ((v - xlo) / (xhi - xlo)) * iw;
    const Y = (v) => pad.t + ih - ((v - ylo) / (yhi - ylo)) * ih;

    axes(f, {
      xTicks: ticksFrom(xlo, xhi, 4).map((t) => ({ ...t, label: `10${sup(Math.round(t.v * 10) / 10)}` })),
      yTicks: ticksFrom(ylo, yhi, 4),
      xLabel: "Training ratings N (log scale)",
      yLabel: "Seconds / epoch (log)",
      yFmt: (v) => (10 ** v).toFixed(v < -1 ? 3 : 2),
    });

    // Fitted power law, and an exact-linear reference through the first point.
    const fitPath = [xlo, xhi].map((v, k) =>
      `${k ? "L" : "M"}${X(v).toFixed(1)},${Y(fit.intercept / Math.LN10 + fit.slope * v).toFixed(1)}`).join("");
    el("path", { d: fitPath, stroke: css("--ink-3"), "stroke-width": 1.6, "stroke-dasharray": "6 3", fill: "none" }, svg);

    const refIntercept = ly[0] - 1.0 * lx[0];
    const refPath = [xlo, xhi].map((v, k) =>
      `${k ? "L" : "M"}${X(v).toFixed(1)},${Y(refIntercept + v).toFixed(1)}`).join("");
    el("path", { d: refPath, stroke: css("--axis"), "stroke-width": 1.2, "stroke-dasharray": "2 3", fill: "none" }, svg);

    const datasets = [...new Set(rows.map((r) => r.dataset))];
    const colours = SERIES();
    rows.forEach((row, i) => {
      const colour = colours[datasets.indexOf(row.dataset) % colours.length];
      const dot = el("circle", {
        cx: X(lx[i]), cy: Y(ly[i]), r: 5, fill: colour,
        stroke: css("--surface"), "stroke-width": 1.5,
      }, svg);
      hoverable(dot, `${row.dataset}\nN = ${row.n_train_ratings.toLocaleString()}\n` +
        `${(row.seconds_per_epoch * 1000).toFixed(1)} ms/epoch\n${row.microseconds_per_rating} µs/rating`);
    });

    // Lead with the within-dataset exponents; the pooled one is secondary
    // because it mixes growth in N with growth in the entity counts.
    const perDataset = payload.per_dataset_fit || {};
    const within = Object.entries(perDataset)
      .map(([name, f]) => `${name} ${f.slope.toFixed(3)}`)
      .join("  ·  ");

    const caption = el("text", {
      x: pad.l + 6, y: pad.t + 11,
      fill: css("--ink"), "font-size": 10.5, "font-family": css("--sans"), "font-weight": 500,
    }, svg);
    caption.textContent = within ? `within dataset: ${within}` : "";

    const sub = el("text", {
      x: pad.l + 6, y: pad.t + 24,
      fill: css("--ink-3"), "font-size": 9.5, "font-family": css("--sans"),
    }, svg);
    sub.textContent = `pooled ${fit.slope.toFixed(3)} [${fit.slope_ci_low.toFixed(3)}, ${fit.slope_ci_high.toFixed(3)}], R² = ${fit.r_squared.toFixed(3)}`;

    const legendMount = document.getElementById("legend-scalability");
    if (legendMount) {
      legendMount.innerHTML = "";
      legend(legendMount, [
        ...datasets.map((d, i) => [d, colours[i % colours.length]]),
        ["fitted power law", css("--ink-3")],
        ["exact linear (slope 1)", css("--axis")],
      ]);
    }
  }

  const sup = (n) => String(n).replace(/[-\d.]/g, (c) => ({ "-": "⁻", ".": "·", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" }[c] || c));

  /* --------------------------------------------------------- 5. verdicts */

  function buildVerdicts(mountId) {
    const mount = document.getElementById(mountId);
    const payload = R["claim_verification_ml-100k"];
    if (!mount || !payload) return void hideSection(mountId);

    mount.innerHTML = "";
    for (const claim of payload.claims) {
      const card = html("div", "verdict");
      card.dataset.verdict = claim.verdict;

      const stripe = html("div", "verdict-stripe");
      const body = html("div", "verdict-body");
      body.appendChild(html("span", "verdict-tag", claim.verdict));
      body.appendChild(html("div", "verdict-claim", `${claim.claim_id}. ${claim.claim}`));
      if (claim.finding) body.appendChild(html("p", "verdict-finding", claim.finding));

      card.appendChild(stripe);
      card.appendChild(body);
      mount.appendChild(card);
    }
  }

  /* ------------------------------------------------------- 6. hero canvas */

  function heroMatrix(canvasId) {
    const canvas = document.getElementById(canvasId);
    const spec = DATA.matrix;
    if (!canvas || !spec) return;

    const ctx = canvas.getContext("2d");
    const cell = 7;
    const gap = 1;
    canvas.width = spec.cols * cell;
    canvas.height = spec.rows * cell;

    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let phase = 0;

    // A rank-2 stand-in for the reconstruction. It is *not* a fitted model, so
    // it must not pretend to predict specific ratings — but it should not be
    // noise either, since the point being made is that a factorisation fills
    // gaps with low-rank structure. Row and column effects give exactly that.
    const rowEffect = spec.grid.map((row) => {
      const seen = row.filter(Boolean);
      return seen.length ? seen.reduce((a, b) => a + b, 0) / seen.length : 3.5;
    });
    const colEffect = spec.grid[0].map((_, c) => {
      const seen = spec.grid.map((row) => row[c]).filter(Boolean);
      return seen.length ? seen.reduce((a, b) => a + b, 0) / seen.length : 3.5;
    });

    function draw(sweep) {
      ctx.fillStyle = css("--surface-sunk");
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      for (let r = 0; r < spec.rows; r++) {
        for (let c = 0; c < spec.cols; c++) {
          const value = spec.grid[r][c];
          const x = c * cell, y = r * cell;
          const w = cell - gap, h = cell - gap;

          if (value) {
            // Ratings are integers, so the arm is kept short (±1.2 rather than
            // ±2) to stop 3s and 4s -- the bulk of the data -- washing out
            // against the light midpoint.
            ctx.fillStyle = divergingColour(value, 3.5, 1.2);
            ctx.globalAlpha = 1;
            ctx.fillRect(x, y, w, h);
          } else if (c < sweep) {
            // Held back so the real observations still read first.
            ctx.fillStyle = divergingColour((rowEffect[r] + colEffect[c]) / 2, 3.5, 1.2);
            ctx.globalAlpha = 0.32;
            ctx.fillRect(x, y, w, h);
          }
        }
      }
      ctx.globalAlpha = 1;

      if (sweep > 0 && sweep < spec.cols) {
        ctx.fillStyle = css("--accent");
        ctx.fillRect(sweep * cell - 1, 0, 2, canvas.height);
      }
    }

    if (reduced) { draw(spec.cols); return; }

    (function animate() {
      phase = (phase + 0.55) % (spec.cols + 90);
      draw(Math.min(spec.cols, phase));
      requestAnimationFrame(animate);
    })();
  }

  /* ------------------------------------------------------ colour scales */

  const hexToRgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
  const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));

  // Diverging pair on a white page: the midpoint is a light neutral so cells at
  // the mean recede toward the surface and only deviations carry ink. (On the
  // earlier dark ground the midpoint had to be a mid-grey instead.)
  const COOL = hexToRgb("#005db7");   // low ratings — the page accent
  const MID = hexToRgb("#eceae4");    // neutral midpoint
  const WARM = hexToRgb("#c0392b");   // high ratings

  /**
   * Continuous diverging scale, anchored on the data's own centre.
   *
   * Two things this gets right that a five-step discrete ramp does not.
   * Predictions cluster tightly around the mean, so rounding them to integers
   * turns tiny differences into large colour jumps and the panel reads as
   * noise; interpolating continuously lets the real row/column structure show.
   * And the midpoint must be the *observed* mean (3.7 on this slice), not the
   * scale's arithmetic middle — anchoring at 3 pushes almost every cell onto
   * the warm arm and wastes half the scale.
   */
  function divergingColour(value, centre = 3.7, halfRange = 1.1) {
    const t = Math.max(-1, Math.min(1, (value - centre) / halfRange));
    const rgb = t < 0 ? mix(MID, COOL, -t) : mix(MID, WARM, t);
    return `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
  }

  /* --------------------------------------------------- 7. browser trainer */

  function liveTrainer() {
    const demo = DATA.demo;
    const canvas = document.getElementById("demo-matrix");
    const curveMount = document.getElementById("demo-curve");
    if (!demo || !canvas || !curveMount) return;

    const ctx = canvas.getContext("2d");
    const nUsers = demo.nUsers, nItems = demo.nItems;
    const users = Int32Array.from(demo.users);
    const items = Int32Array.from(demo.items);
    const ratings = Float64Array.from(demo.ratings);
    const n = users.length;

    // A fixed 80/20 split, drawn once so every run is comparable.
    const isTest = new Uint8Array(n);
    for (let i = 0; i < n; i++) isTest[i] = i % 5 === 0 ? 1 : 0;
    const trainIdx = [], testIdx = [];
    for (let i = 0; i < n; i++) (isTest[i] ? testIdx : trainIdx).push(i);

    const ui = {
      factors: document.getElementById("ctl-factors"),
      lr: document.getElementById("ctl-lr"),
      reg: document.getElementById("ctl-reg"),
      factorsOut: document.getElementById("out-factors"),
      lrOut: document.getElementById("out-lr"),
      regOut: document.getElementById("out-reg"),
      epoch: document.getElementById("out-epoch"),
      train: document.getElementById("out-train"),
      test: document.getElementById("out-test"),
      run: document.getElementById("btn-run"),
      reset: document.getElementById("btn-reset"),
    };

    let state = null;
    let running = false;
    let history = [];

    function hyper() {
      return {
        d: Number(ui.factors.value),
        lr: Number(ui.lr.value) / 10000,
        reg: Number(ui.reg.value) / 1000,
      };
    }

    function syncLabels() {
      const h = hyper();
      ui.factorsOut.textContent = h.d;
      ui.lrOut.textContent = h.lr.toFixed(4);
      ui.regOut.textContent = h.reg.toFixed(3);
    }

    function reset() {
      const { d } = hyper();
      let mean = 0;
      for (const i of trainIdx) mean += ratings[i];
      mean /= trainIdx.length;

      // Deterministic PRNG so a given control setting always replays identically.
      let seed = 20250817;
      const rand = () => {
        seed = (seed * 1664525 + 1013904223) >>> 0;
        return seed / 4294967296 - 0.5;
      };

      // Initialisation scale matters more here than in the offline trainer. Both
      // factor sets start near zero, so the gradient on each is proportional to
      // the other and the model barely moves until the factors grow. At the
      // 0.05 scale used offline (which has momentum and hundreds of epochs to
      // spare) this demo sits flat for ~20 epochs before anything happens.
      // INIT_SCALE = 0.5 puts it on a visible trajectory within five.
      const INIT_SCALE = 0.5;

      state = {
        d, mean,
        U: Float64Array.from({ length: nUsers * d }, () => rand() * INIT_SCALE),
        V: Float64Array.from({ length: nItems * d }, () => rand() * INIT_SCALE),
        epoch: 0,
      };
      history = [];
      update();
    }

    function epoch() {
      const { d, lr, reg } = { ...hyper(), d: state.d };
      const { U, V, mean } = state;

      for (let k = trainIdx.length - 1; k > 0; k--) {
        const j = (Math.random() * (k + 1)) | 0;
        [trainIdx[k], trainIdx[j]] = [trainIdx[j], trainIdx[k]];
      }

      for (const idx of trainIdx) {
        const uo = users[idx] * d, io = items[idx] * d;
        let dot = 0;
        for (let f = 0; f < d; f++) dot += U[uo + f] * V[io + f];
        const err = dot - (ratings[idx] - mean);
        for (let f = 0; f < d; f++) {
          const u = U[uo + f], v = V[io + f];
          U[uo + f] = u - lr * (err * v + reg * u);
          V[io + f] = v - lr * (err * u + reg * v);
        }
      }
      state.epoch++;
    }

    function score(indices) {
      const { U, V, d, mean } = state;
      let sum = 0;
      for (const idx of indices) {
        const uo = users[idx] * d, io = items[idx] * d;
        let dot = 0;
        for (let f = 0; f < d; f++) dot += U[uo + f] * V[io + f];
        const predicted = Math.min(5, Math.max(1, dot + mean));
        const e = predicted - ratings[idx];
        sum += e * e;
      }
      return Math.sqrt(sum / indices.length);
    }

    function drawMatrix() {
      const rows = 70, cols = 120;
      const cw = canvas.width / cols, ch = canvas.height / rows;
      const { U, V, d, mean } = state;

      ctx.fillStyle = css("--surface-sunk");
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      for (let r = 0; r < rows; r++) {
        const uo = r * d;
        for (let c = 0; c < cols; c++) {
          const io = c * d;
          let dot = 0;
          for (let f = 0; f < d; f++) dot += U[uo + f] * V[io + f];
          const predicted = Math.min(5, Math.max(1, dot + mean));
          ctx.fillStyle = divergingColour(predicted, mean, 1.1);
          ctx.fillRect(c * cw, r * ch, Math.ceil(cw), Math.ceil(ch));
        }
      }
    }

    function drawCurve() {
      const f = frame(curveMount, { width: 520, height: 190, pad: { t: 12, r: 14, b: 32, l: 46 } });
      const { svg, pad, iw, ih } = f;
      if (history.length < 2) {
        const note = el("text", {
          x: pad.l + iw / 2, y: pad.t + ih / 2, "text-anchor": "middle",
          fill: css("--ink-3"), "font-size": 12, "font-family": css("--sans"),
        }, svg);
        note.textContent = "Press Train to start";
        return;
      }

      const all = history.flatMap((h) => [h.train, h.test]);
      const lo = Math.min(...all) - 0.03, hi = Math.max(...all) + 0.03;
      const maxEpoch = history[history.length - 1].epoch;

      axes(f, {
        xTicks: ticksFrom(0, maxEpoch, 4).map((t) => ({ ...t, label: Math.round(t.v) })),
        yTicks: ticksFrom(lo, hi, 4),
        xLabel: "Epoch",
        yLabel: "RMSE",
        yFmt: (v) => v.toFixed(2),
      });

      const X = (e) => pad.l + (maxEpoch ? e / maxEpoch : 0) * iw;
      const Y = (v) => pad.t + ih - ((v - lo) / (hi - lo)) * ih;

      for (const [key, colour] of [["train", css("--series-1")], ["test", css("--series-2")]]) {
        el("path", {
          d: history.map((h, i) => `${i ? "L" : "M"}${X(h.epoch).toFixed(1)},${Y(h[key]).toFixed(1)}`).join(""),
          fill: "none", stroke: colour, "stroke-width": 2,
        }, svg);
      }

      const best = history.reduce((a, b) => (b.test < a.test ? b : a));
      el("circle", { cx: X(best.epoch), cy: Y(best.test), r: 4.5, fill: css("--series-2"), stroke: css("--surface"), "stroke-width": 1.5 }, svg);
    }

    function update() {
      ui.epoch.textContent = state.epoch;
      const train = score(trainIdx), test = score(testIdx);
      ui.train.textContent = fmt(train, 4);
      ui.test.textContent = fmt(test, 4);
      if (!history.length || history[history.length - 1].epoch !== state.epoch) {
        history.push({ epoch: state.epoch, train, test });
      }
      drawMatrix();
      drawCurve();
    }

    function loop() {
      if (!running) return;
      epoch();
      update();
      if (state.epoch >= 40) { running = false; ui.run.textContent = "Train"; return; }
      // rAF rather than setTimeout: it yields a frame between epochs so the
      // canvas actually repaints, and it suspends when the tab is hidden.
      requestAnimationFrame(loop);
    }

    ui.run.addEventListener("click", () => {
      running = !running;
      ui.run.textContent = running ? "Pause" : "Train";
      if (running) loop();
    });
    ui.reset.addEventListener("click", () => { running = false; ui.run.textContent = "Train"; reset(); });
    for (const control of [ui.factors, ui.lr, ui.reg]) {
      control.addEventListener("input", () => {
        syncLabels();
        if (control === ui.factors) { running = false; ui.run.textContent = "Train"; reset(); }
      });
    }

    canvas.width = 600;
    canvas.height = 350;
    syncLabels();
    reset();
  }

  /* ------------------------------------------------------------ helpers */

  function hideSection(mountId) {
    const node = document.getElementById(mountId);
    const container = node && node.closest("[data-optional]");
    if (container) container.classList.add("hidden");
  }

  function fillText() {
    for (const node of document.querySelectorAll("[data-fill]")) {
      const path = node.dataset.fill.split(".");
      let value = { R, DATA };
      for (const key of path) value = value && value[key];
      if (value !== undefined && value !== null) {
        node.textContent = node.dataset.digits ? fmt(value, Number(node.dataset.digits)) : value;
      }
    }
  }

  /* --------------------------------------------------------------- boot */

  function render() {
    fillText();
    heroMatrix("hero-matrix");
    chartComparison("chart-comparison", R["model_comparison_ml-100k"], "table-comparison");
    chartConvergence("chart-convergence", "legend-convergence");
    buildVerdicts("verdict-list");
    chartScalability("chart-scalability");

    const dim = R["latent_dimension_ml-100k"];
    if (dim) {
      lineChart("chart-dimension", "legend-dimension", {
        x: dim.table.map((r) => r.n_factors),
        labels: dim.table.map((r) => `d=${r.n_factors}`),
        xLabel: "Latent dimensionality d",
        yLabel: "RMSE",
        series: [
          { label: "Test (early stopping)", values: dim.table.map((r) => r.test_rmse), band: dim.table.map((r) => r.test_rmse_ci95) },
          { label: "Test (no early stopping)", values: dim.table.map((r) => r.test_rmse_no_early_stop), dashed: true },
          { label: "Train", values: dim.table.map((r) => r.train_rmse) },
        ],
      });
    } else hideSection("chart-dimension");

    const reg = R["regularisation_ml-100k"];
    if (reg) {
      lineChart("chart-regularisation", "legend-regularisation", {
        x: reg.table.map((r) => r.reg),
        labels: reg.table.map((r) => `λ=${r.reg}`),
        xLabel: "Regularisation strength λ",
        yLabel: "RMSE",
        series: [
          { label: "Train RMSE", values: reg.table.map((r) => r.train_rmse) },
          { label: "Test RMSE", values: reg.table.map((r) => r.test_rmse), band: reg.table.map((r) => r.test_rmse_ci95) },
        ],
      });
    } else hideSection("chart-regularisation");

    const cold = R["cold_start_ml-100k"];
    if (cold) {
      const models = Object.keys(cold.table[0]).filter(
        (k) => k !== "bin" && k !== "n_test_ratings" && !k.endsWith("ci95")
      );

      // A stratum needs enough test observations for its interval to mean
      // anything. MovieLens guarantees 20 ratings per user, so the lowest bin
      // fills only when a split strands one user's ratings in test: it holds
      // three, and its interval spans more than the rating scale.
      const MIN_RELIABLE = 50;
      const build = (table) => ({
        x: table.map((_, i) => i),
        labels: table.map((r) => r.bin),
        notes: table.map((r) => `n = ${r.n_test_ratings.toLocaleString()}`),
        xLabel: "Training ratings by the user",
        yLabel: "Test RMSE",
        height: 235,
        series: models.map((m) => ({
          label: m,
          values: table.map((r) => r[m]),
          band: table.map((r) => (r.n_test_ratings >= MIN_RELIABLE ? r[`${m} ci95`] || 0 : 0)),
        })),
      });

      lineChart("chart-coldstart", "legend-coldstart", build(cold.table));
      lineChart(
        "chart-coldstart-zoom", null,
        build(cold.table.filter((r) => r.n_test_ratings >= MIN_RELIABLE))
      );

      const note = document.getElementById("coldstart-note");
      const thin = cold.table.filter((r) => r.n_test_ratings < MIN_RELIABLE);
      if (note && thin.length) {
        note.textContent =
          `The ${thin.map((r) => r.bin).join(", ")} stratum holds only ` +
          `${thin[0].n_test_ratings} test ratings across all seeds, so its interval is ` +
          `wider than the rating scale and is suppressed. The right-hand panel drops it.`;
      }
    } else {
      hideSection("chart-coldstart");
    }

    liveTrainer();
  }

  // The page commits to a single light theme (see styles.css), so there is no
  // theme-change redraw to wire up; charts read their colours once at draw time.
  document.addEventListener("DOMContentLoaded", render);
})();

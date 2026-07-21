(function () {
  "use strict";
  const DATA = JSON.parse(document.getElementById("app-data").textContent);
  const SVGNS = "http://www.w3.org/2000/svg";

  const METRICS = [
    { key: "accuracy",         label: "Accuracy",        short: "Acc"  },
    { key: "precision_macro",  label: "Precision (macro)", short: "Prec" },
    { key: "recall_macro",     label: "Recall (macro)",   short: "Rec"  },
    { key: "f1_macro",         label: "F1 (macro)",       short: "F1"   },
    { key: "roc_auc_ovo_macro",label: "ROC AUC (OvO)",    short: "AUC"  },
  ];

  const EXPERIMENTS = [
    { id: "amusement", label: "Hold‑out: Amusement", short: "Amusement", mode: "class_holdout", novel_class: "amusement" },
    { id: "baseline",  label: "Hold‑out: Baseline",  short: "Baseline",  mode: "class_holdout", novel_class: "baseline"  },
    { id: "stress",    label: "Hold‑out: Stress",    short: "Stress",    mode: "class_holdout", novel_class: "stress"    },
    { id: "loso",      label: "LOSO (15 subjects)",       short: "LOSO",      mode: "loso",          novel_class: null        },
  ];

  function findRun(mode, novel_class, use_film) {
    return DATA.runs.find(r => r.mode === mode && r.novel_class === novel_class && r.use_film === use_film);
  }
  function val(run, metricKey) {
    const m = run.metrics || run.mean;
    return m ? m[metricKey] : undefined;
  }
  function std(run, metricKey) {
    return run.std ? run.std[metricKey] : null;
  }
  const fmtPct = v => (v == null ? "—" : (v * 100).toFixed(1) + "%");
  const fmt3   = v => (v == null ? "—" : v.toFixed(3));
  const fmtParams = n => n.toLocaleString("en-US");

  let currentMetric = "accuracy";

  // ---------------- config strip ----------------
  (function renderConfig() {
    const c = DATA.config;
    const items = [
      ["Dataset", "WESAD"],
      ["Trials / experiment", c.n_trials],
      ["Search epochs", c.search_epochs],
      ["Final epochs", c.epochs],
      ["r_minutes_max", c.r_minutes_max + " min"],
      ["Seed", c.seed],
    ];
    const el = document.getElementById("config-strip");
    el.innerHTML = items.map(([k, v]) => `<span class="chip">${k} <b>${v}</b></span>`).join("");
  })();

  // ---------------- metric toggle ----------------
  (function renderToggle() {
    const el = document.getElementById("metric-toggle");
    el.innerHTML = METRICS.map(m =>
      `<button data-metric="${m.key}" role="tab" aria-selected="${m.key === currentMetric}">${m.label}</button>`
    ).join("");
    el.querySelectorAll("button").forEach(btn => {
      btn.classList.toggle("active", btn.dataset.metric === currentMetric);
      btn.addEventListener("click", () => {
        currentMetric = btn.dataset.metric;
        el.querySelectorAll("button").forEach(b => {
          b.classList.toggle("active", b === btn);
          b.setAttribute("aria-selected", b === btn);
        });
        renderAll();
      });
    });
  })();

  // ---------------- tooltip ----------------
  const tt = document.getElementById("tooltip");
  function showTooltip(evt, html) {
    tt.innerHTML = html;
    tt.classList.add("show");
    moveTooltip(evt);
  }
  function moveTooltip(evt) {
    const pad = 14;
    let x = evt.clientX + pad, y = evt.clientY + pad;
    const rect = tt.getBoundingClientRect();
    if (x + rect.width + 12 > window.innerWidth) x = evt.clientX - rect.width - pad;
    if (y + rect.height + 12 > window.innerHeight) y = evt.clientY - rect.height - pad;
    tt.style.left = x + "px";
    tt.style.top = y + "px";
  }
  function hideTooltip() { tt.classList.remove("show"); }

  // ---------------- svg helpers ----------------
  function svgEl(tag, attrs) {
    const e = document.createElementNS(SVGNS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }
  function scaleLinear(domain, range) {
    const [d0, d1] = domain, [r0, r1] = range;
    return x => r0 + (x - d0) / (d1 - d0) * (r1 - r0);
  }

  // ================= KPI tiles =================
  function renderKPIs() {
    const lp = findRun("loso", null, false), lf = findRun("loso", null, true);
    const dAcc = val(lf, "accuracy") - val(lp, "accuracy");

    let f1PlainSum = 0, f1FilmSum = 0;
    EXPERIMENTS.filter(e => e.mode === "class_holdout").forEach(e => {
      f1PlainSum += val(findRun("class_holdout", e.novel_class, false), "f1_macro");
      f1FilmSum  += val(findRun("class_holdout", e.novel_class, true),  "f1_macro");
    });
    const dF1 = (f1FilmSum - f1PlainSum) / 3;

    let maxParamDiffPct = 0;
    EXPERIMENTS.forEach(e => {
      const rp = findRun(e.mode, e.novel_class, false), rf = findRun(e.mode, e.novel_class, true);
      const diff = Math.abs(rp.params - rf.params) / ((rp.params + rf.params) / 2) * 100;
      maxParamDiffPct = Math.max(maxParamDiffPct, diff);
    });

    const filmRuns = DATA.runs.filter(r => r.use_film);
    let rEffSum = 0, rEffN = 0;
    filmRuns.forEach(r => {
      const m = r.metrics || r.mean;
      if (m && m.effective_r_minutes != null) { rEffSum += m.effective_r_minutes; rEffN++; }
    });
    const rEffAvg = rEffSum / rEffN;

    const tiles = [
      {
        label: "Ablation runs", value: DATA.runs.length,
        sub: `3 hold‑out classes × 2 arms, + LOSO × 2 arms`,
      },
      {
        label: "LOSO accuracy, FiLM − Plain", value: (dAcc >= 0 ? "+" : "") + (dAcc * 100).toFixed(2) + " pt",
        sub: `${fmtPct(val(lp,"accuracy"))} plain → ${fmtPct(val(lf,"accuracy"))} film, mean over 15 subjects`,
        deltaClass: Math.abs(dAcc) < 0.005 ? "flat" : (dAcc > 0 ? "good" : "bad"),
      },
      {
        label: "Hold‑out F1, avg FiLM − Plain", value: (dF1 >= 0 ? "+" : "") + (dF1 * 100).toFixed(2) + " pt",
        sub: "averaged across amusement / baseline / stress hold‑outs",
        deltaClass: Math.abs(dF1) < 0.005 ? "flat" : (dF1 > 0 ? "good" : "bad"),
      },
      {
        label: "Parameter match", value: "≤ " + maxParamDiffPct.toFixed(2) + "%",
        sub: "largest plain↔FiLM parameter‑count gap across all 4 matched pairs",
      },
      {
        label: "Learned baseline window", value: rEffAvg.toFixed(1) + " min",
        sub: `avg. effective duration the FiLM selector converged on, of a ${DATA.config.r_minutes_max.toFixed(0)} min cap`,
      },
    ];

    const el = document.getElementById("kpi-grid");
    el.innerHTML = tiles.map(t => `
      <div class="kpi">
        <div class="k-label">${t.label}</div>
        <div class="k-value mono">${t.value}</div>
        <div class="k-sub">${t.sub}</div>
        ${t.deltaClass ? `<div class="k-delta ${t.deltaClass}">${t.deltaClass === "good" ? "↑" : t.deltaClass === "bad" ? "↓" : "≈"} vs plain</div>` : ""}
      </div>
    `).join("");
  }

  // ================= main grouped bar chart =================
  function renderMainChart() {
    const container = document.getElementById("main-chart");
    container.innerHTML = "";
    const W = 1000, H = 340;
    const marginL = 40, marginR = 16, marginT = 16, marginB = 40;
    const plotW = W - marginL - marginR, plotH = H - marginT - marginB;

    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", height: H, role: "img",
      "aria-label": "Grouped bar chart comparing plain and FiLM arms across hold-out and LOSO splits" });
    container.appendChild(svg);

    const y = scaleLinear([0, 1], [marginT + plotH, marginT]);
    // gridlines + y labels
    [0, 0.2, 0.4, 0.6, 0.8, 1.0].forEach(t => {
      svg.appendChild(svgEl("line", { x1: marginL, x2: marginL + plotW, y1: y(t), y2: y(t), class: "grid-line" }));
      const lbl = svgEl("text", { x: marginL - 8, y: y(t) + 3, class: "axis-label", "text-anchor": "end" });
      lbl.textContent = t.toFixed(1);
      svg.appendChild(lbl);
    });
    svg.appendChild(svgEl("line", { x1: marginL, x2: marginL + plotW, y1: y(0), y2: y(0), class: "baseline-line" }));

    const groupW = plotW / EXPERIMENTS.length;
    const barW = Math.min(58, groupW * 0.28);
    const barGap = 6;

    EXPERIMENTS.forEach((exp, gi) => {
      const gx = marginL + gi * groupW + groupW / 2;
      const rp = findRun(exp.mode, exp.novel_class, false);
      const rf = findRun(exp.mode, exp.novel_class, true);
      const vp = val(rp, currentMetric), vf = val(rf, currentMetric);
      const sp = std(rp, currentMetric), sf = std(rf, currentMetric);

      [{ run: rp, v: vp, s: sp, cls: "bar-plain", dx: -barGap / 2 - barW, label: "Plain" },
       { run: rf, v: vf, s: sf, cls: "bar-film",  dx:  barGap / 2,        label: "FiLM"  }
      ].forEach(b => {
        if (b.v == null) return;
        const x0 = gx + b.dx;
        const y0 = y(b.v);
        const h = y(0) - y0;
        const rect = svgEl("rect", {
          x: x0, y: y0, width: barW, height: Math.max(h, 1),
          rx: 4, ry: 4, class: b.cls, style: "cursor:pointer"
        });
        svg.appendChild(rect);
        rect.addEventListener("mousemove", e => {
          const rows = [
            [`${METRICS.find(m=>m.key===currentMetric).label}`, `<b>${fmt3(b.v)}</b>`],
          ];
          if (b.s != null) rows.push(["std across folds", `<b>&plusmn;${fmt3(b.s)}</b>`]);
          rows.push(["hidden_dim", `<b>${b.run.config.hidden_dim}</b>`]);
          rows.push(["params", `<b>${fmtParams(b.run.params)}</b>`]);
          showTooltip(e, `<div class="tt-title">${exp.label} · ${b.label}</div>` +
            rows.map(r => `<div class="tt-row"><span>${r[0]}</span>${r[1]}</div>`).join(""));
        });
        rect.addEventListener("mouseleave", hideTooltip);

        // error whisker
        if (b.s != null) {
          const yTop = y(Math.min(1, b.v + b.s));
          const yBot = y(Math.max(0, b.v - b.s));
          const cx = x0 + barW / 2;
          svg.appendChild(svgEl("line", { x1: cx, x2: cx, y1: yTop, y2: yBot, stroke: "var(--text-primary)", "stroke-width": 1.4, opacity: 0.55 }));
          svg.appendChild(svgEl("line", { x1: cx - 5, x2: cx + 5, y1: yTop, y2: yTop, stroke: "var(--text-primary)", "stroke-width": 1.4, opacity: 0.55 }));
          svg.appendChild(svgEl("line", { x1: cx - 5, x2: cx + 5, y1: yBot, y2: yBot, stroke: "var(--text-primary)", "stroke-width": 1.4, opacity: 0.55 }));
        }

        // direct label
        const lbl = svgEl("text", { x: x0 + barW / 2, y: y0 - 6, class: "val-label", "text-anchor": "middle" });
        lbl.textContent = b.v.toFixed(3);
        svg.appendChild(lbl);
      });

      const glabel = svgEl("text", { x: gx, y: marginT + plotH + 22, class: "grp-label", "text-anchor": "middle" });
      glabel.textContent = exp.label;
      svg.appendChild(glabel);
    });
  }

  // ================= LOSO subject chart =================
  function renderSubjectChart() {
    const container = document.getElementById("subject-chart");
    container.innerHTML = "";
    const rp = findRun("loso", null, false), rf = findRun("loso", null, true);
    const subjects = Object.keys(rp.per_fold).sort((a, b) => parseInt(a.slice(1)) - parseInt(b.slice(1)));

    const barW = 12, barGap = 4, groupGap = 22;
    const groupW = barW * 2 + barGap + groupGap;
    const marginL = 40, marginR = 16, marginT = 16, marginB = 34;
    const plotW = subjects.length * groupW;
    const plotH = 260;
    const W = marginL + plotW + marginR, H = marginT + plotH + marginB;

    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: Math.max(W, 640), height: H, role: "img",
      "aria-label": "Per-subject LOSO accuracy, plain vs FiLM" });
    container.appendChild(svg);
    container.style.minWidth = "640px";

    const y = scaleLinear([0, 1], [marginT + plotH, marginT]);
    [0, 0.2, 0.4, 0.6, 0.8, 1.0].forEach(t => {
      svg.appendChild(svgEl("line", { x1: marginL, x2: marginL + plotW, y1: y(t), y2: y(t), class: "grid-line" }));
      const lbl = svgEl("text", { x: marginL - 8, y: y(t) + 3, class: "axis-label", "text-anchor": "end" });
      lbl.textContent = t.toFixed(1);
      svg.appendChild(lbl);
    });
    svg.appendChild(svgEl("line", { x1: marginL, x2: marginL + plotW, y1: y(0), y2: y(0), class: "baseline-line" }));

    // mean reference lines
    const meanP = rp.mean[currentMetric], meanF = rf.mean[currentMetric];
    [["Plain mean", meanP, "var(--series-plain)"], ["FiLM mean", meanF, "var(--series-film)"]].forEach(([label, v, color]) => {
      if (v == null) return;
      const line = svgEl("line", { x1: marginL, x2: marginL + plotW, y1: y(v), y2: y(v), stroke: color, "stroke-width": 1.3, "stroke-dasharray": "3,3", opacity: 0.65 });
      svg.appendChild(line);
    });

    subjects.forEach((subj, i) => {
      const gx = marginL + i * groupW + groupGap / 2;
      const vp = rp.per_fold[subj][currentMetric];
      const vf = rf.per_fold[subj][currentMetric];
      [{ v: vp, cls: "bar-plain", x: gx, label: "Plain" }, { v: vf, cls: "bar-film", x: gx + barW + barGap, label: "FiLM" }].forEach(b => {
        if (b.v == null) return;
        const y0 = y(b.v);
        const rect = svgEl("rect", { x: b.x, y: y0, width: barW, height: Math.max(y(0) - y0, 1), rx: 3, ry: 3, class: b.cls, style: "cursor:pointer" });
        svg.appendChild(rect);
        rect.addEventListener("mousemove", e => {
          showTooltip(e, `<div class="tt-title">${subj} · ${b.label}</div>` +
            `<div class="tt-row"><span>${METRICS.find(m=>m.key===currentMetric).label}</span><b>${fmt3(b.v)}</b></div>`);
        });
        rect.addEventListener("mouseleave", hideTooltip);
      });
      const glabel = svgEl("text", { x: gx + barW + barGap / 2, y: marginT + plotH + 18, class: "grp-label", "text-anchor": "middle", "font-size": "10.5" });
      glabel.textContent = subj;
      svg.appendChild(glabel);
    });
  }

  // ================= delta grid =================
  function renderDeltaGrid() {
    const container = document.getElementById("delta-chart");
    container.innerHTML = "";
    container.style.minWidth = "620px";

    // compute all deltas to find shared scale
    const rows = EXPERIMENTS.map(exp => {
      const rp = findRun(exp.mode, exp.novel_class, false);
      const rf = findRun(exp.mode, exp.novel_class, true);
      const deltas = METRICS.map(m => {
        const vp = val(rp, m.key), vf = val(rf, m.key);
        return (vp == null || vf == null) ? null : vf - vp;
      });
      return { exp, deltas };
    });
    let maxAbs = 0;
    rows.forEach(r => r.deltas.forEach(d => { if (d != null) maxAbs = Math.max(maxAbs, Math.abs(d)); }));
    maxAbs = maxAbs * 1.2 || 0.1;

    const grid = document.createElement("div");
    grid.className = "delta-grid";
    grid.appendChild(Object.assign(document.createElement("div"), { className: "col-head", textContent: "" }));
    METRICS.forEach(m => {
      const h = document.createElement("div");
      h.className = "col-head";
      h.textContent = m.short;
      grid.appendChild(h);
    });

    rows.forEach(r => {
      const rh = document.createElement("div");
      rh.className = "row-head";
      rh.innerHTML = `${r.exp.short}<small>${r.exp.mode === "loso" ? "mean over 15 folds" : "class hold‑out"}</small>`;
      grid.appendChild(rh);

      METRICS.forEach((m, mi) => {
        const cell = document.createElement("div");
        cell.className = "delta-cell";
        const d = r.deltas[mi];
        const axis = document.createElement("div");
        axis.className = "axis";
        cell.appendChild(axis);
        if (d != null) {
          const pct = Math.min(1, Math.abs(d) / maxAbs) * 42; // px half-width max
          const bar = document.createElement("div");
          bar.className = "bar " + (d >= 0 ? "pos" : "neg");
          bar.style.width = pct + "px";
          if (d < 0) bar.style.left = `calc(50% - ${pct}px)`;
          cell.appendChild(bar);
          const valEl = document.createElement("div");
          valEl.className = "val";
          valEl.textContent = (d >= 0 ? "+" : "") + d.toFixed(3);
          valEl.style.color = d >= 0 ? "var(--good)" : "var(--critical)";
          valEl.style[d >= 0 ? "left" : "right"] = "50%";
          valEl.style[d >= 0 ? "marginLeft" : "marginRight"] = "3px";
          if (d < 0) valEl.style.textAlign = "right";
          cell.appendChild(valEl);
          cell.style.cursor = "pointer";
          cell.addEventListener("mousemove", e => {
            const rp = findRun(r.exp.mode, r.exp.novel_class, false);
            const rf = findRun(r.exp.mode, r.exp.novel_class, true);
            showTooltip(e, `<div class="tt-title">${r.exp.label} · ${m.label}</div>` +
              `<div class="tt-row"><span>Plain</span><b>${fmt3(val(rp, m.key))}</b></div>` +
              `<div class="tt-row"><span>FiLM</span><b>${fmt3(val(rf, m.key))}</b></div>` +
              `<div class="tt-row"><span>Δ (FiLM − Plain)</span><b style="color:${d>=0?'#7CD87C':'#ff9a8a'}">${(d>=0?"+":"")+d.toFixed(3)}</b></div>`);
          });
          cell.addEventListener("mouseleave", hideTooltip);
        }
        grid.appendChild(cell);
      });
    });

    container.appendChild(grid);
  }

  // ================= scatter: metric vs params =================
  function renderScatter() {
    const container = document.getElementById("scatter-chart");
    container.innerHTML = "";
    const W = 1000, H = 380;
    const marginL = 44, marginR = 20, marginT = 16, marginB = 40;
    const plotW = W - marginL - marginR, plotH = H - marginT - marginB;

    const params = DATA.runs.map(r => r.params);
    const pMin = Math.min(...params), pMax = Math.max(...params);
    const pad = (pMax - pMin) * 0.12 || pMax * 0.05;
    const x = scaleLinear([pMin - pad, pMax + pad], [marginL, marginL + plotW]);
    const y = scaleLinear([0, 1], [marginT + plotH, marginT]);

    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", height: H, role: "img",
      "aria-label": "Scatter plot of metric value versus trainable parameter count" });
    container.appendChild(svg);

    [0, 0.2, 0.4, 0.6, 0.8, 1.0].forEach(t => {
      svg.appendChild(svgEl("line", { x1: marginL, x2: marginL + plotW, y1: y(t), y2: y(t), class: "grid-line" }));
      const lbl = svgEl("text", { x: marginL - 8, y: y(t) + 3, class: "axis-label", "text-anchor": "end" });
      lbl.textContent = t.toFixed(1);
      svg.appendChild(lbl);
    });
    // x ticks
    const xTicks = 5;
    for (let i = 0; i <= xTicks; i++) {
      const v = pMin - pad + (i / xTicks) * (pMax - pMin + 2 * pad);
      const xx = x(v);
      svg.appendChild(svgEl("line", { x1: xx, x2: xx, y1: marginT, y2: marginT + plotH, class: "grid-line", opacity: 0.5 }));
      const lbl = svgEl("text", { x: xx, y: marginT + plotH + 18, class: "axis-label", "text-anchor": "middle" });
      lbl.textContent = (v / 1e6).toFixed(1) + "M";
      svg.appendChild(lbl);
    }
    const xAxisLbl = svgEl("text", { x: marginL + plotW / 2, y: H - 4, class: "axis-label", "text-anchor": "middle" });
    xAxisLbl.textContent = "Trainable parameters";
    svg.appendChild(xAxisLbl);

    // connecting lines between matched pairs
    EXPERIMENTS.forEach(exp => {
      const rp = findRun(exp.mode, exp.novel_class, false), rf = findRun(exp.mode, exp.novel_class, true);
      const vp = val(rp, currentMetric), vf = val(rf, currentMetric);
      if (vp == null || vf == null) return;
      svg.appendChild(svgEl("line", {
        x1: x(rp.params), y1: y(vp), x2: x(rf.params), y2: y(vf),
        stroke: "var(--muted)", "stroke-width": 1, "stroke-dasharray": "3,3", opacity: 0.5
      }));
    });

    // FiLM drawn as solid dots first, Plain drawn as hollow rings on top — some
    // matched pairs (e.g. LOSO) land within a pixel of each other on this scale,
    // and a transparent-fill ring lets the point underneath stay visible and
    // separately hoverable instead of one arm fully occluding the other.
    function drawPoint(exp, useFilm) {
      const r = findRun(exp.mode, exp.novel_class, useFilm);
      const v = val(r, currentMetric);
      if (v == null) return;
      const cx = x(r.params), cy = y(v);
      const label = useFilm ? "FiLM" : "Plain";
      const attrs = useFilm
        ? { cx, cy, r: 5.5, fill: "var(--series-film)", stroke: "var(--surface)", "stroke-width": 1.3 }
        : { cx, cy, r: 8, fill: "none", stroke: "var(--series-plain)", "stroke-width": 2.4 };
      const circle = svgEl("circle", Object.assign(attrs, { style: "cursor:pointer" }));
      svg.appendChild(circle);
      circle.addEventListener("mousemove", e => {
        showTooltip(e, `<div class="tt-title">${exp.label} · ${label}</div>` +
          `<div class="tt-row"><span>${METRICS.find(m=>m.key===currentMetric).label}</span><b>${fmt3(v)}</b></div>` +
          `<div class="tt-row"><span>Parameters</span><b>${fmtParams(r.params)}</b></div>` +
          `<div class="tt-row"><span>hidden_dim</span><b>${r.config.hidden_dim}</b></div>`);
      });
      circle.addEventListener("mouseleave", hideTooltip);
    }
    EXPERIMENTS.forEach(exp => drawPoint(exp, true));
    EXPERIMENTS.forEach(exp => drawPoint(exp, false));
  }

  // ================= data table =================
  const TABLE_COLS = [
    { key: "experiment", label: "Experiment", sort: (a,b) => a.expLabel.localeCompare(b.expLabel) },
    { key: "arm", label: "Arm", sort: (a,b) => Number(a.use_film) - Number(b.use_film) },
    { key: "hidden_dim", label: "Hidden", sort: (a,b) => a.config.hidden_dim - b.config.hidden_dim },
    { key: "params", label: "Params", sort: (a,b) => a.params - b.params },
    { key: "accuracy", label: "Accuracy" },
    { key: "precision_macro", label: "Precision" },
    { key: "recall_macro", label: "Recall" },
    { key: "f1_macro", label: "F1" },
    { key: "roc_auc_ovo_macro", label: "ROC AUC" },
    { key: "effective_r_minutes", label: "Learned baseline" },
  ];
  METRICS.forEach(m => {
    const c = TABLE_COLS.find(c => c.key === m.key);
    c.sort = (a, b) => {
      const av = a.metrics ? a.metrics[m.key] : (a.mean ? a.mean[m.key] : -1);
      const bv = b.metrics ? b.metrics[m.key] : (b.mean ? b.mean[m.key] : -1);
      return (av == null ? -1 : av) - (bv == null ? -1 : bv);
    };
  });

  let tableSort = { key: null, dir: 1 };

  function tableRows() {
    return DATA.runs.map(r => {
      const exp = EXPERIMENTS.find(e => e.mode === r.mode && e.novel_class === r.novel_class);
      return Object.assign({}, r, { expLabel: exp ? exp.label : r.key });
    });
  }

  function renderTableHead() {
    const head = document.getElementById("table-head");
    head.innerHTML = "";
    TABLE_COLS.forEach(c => {
      const th = document.createElement("th");
      th.tabIndex = 0;
      th.innerHTML = c.label + (tableSort.key === c.key ? `<span class="arrow">${tableSort.dir > 0 ? "▲" : "▼"}</span>` : "");
      th.addEventListener("click", () => {
        if (tableSort.key === c.key) tableSort.dir *= -1; else { tableSort.key = c.key; tableSort.dir = 1; }
        renderTable();
      });
      head.appendChild(th);
    });
  }

  function renderTableBody() {
    let rows = tableRows();
    const col = TABLE_COLS.find(c => c.key === tableSort.key);
    if (col && col.sort) {
      rows = rows.slice().sort((a, b) => col.sort(a, b) * tableSort.dir);
    } else {
      rows.sort((a, b) => a.expLabel.localeCompare(b.expLabel) || Number(a.use_film) - Number(b.use_film));
    }
    const body = document.getElementById("table-body");
    body.innerHTML = rows.map(r => {
      const m = r.metrics || r.mean;
      const s = r.std;
      const cell = key => {
        if (!m || m[key] == null) return "—";
        const base = m[key].toFixed(3);
        return s && s[key] != null ? `${base} <span style="color:var(--muted)">&plusmn;${s[key].toFixed(3)}</span>` : base;
      };
      const rEff = m && m.effective_r_minutes != null ? m.effective_r_minutes.toFixed(2) + " min" : "—";
      return `<tr>
        <td>${r.expLabel}</td>
        <td><span class="tag ${r.use_film ? "film" : "plain"}">${r.use_film ? "FiLM" : "Plain"}</span></td>
        <td>${r.config.hidden_dim}</td>
        <td>${fmtParams(r.params)}</td>
        <td>${cell("accuracy")}</td>
        <td>${cell("precision_macro")}</td>
        <td>${cell("recall_macro")}</td>
        <td>${cell("f1_macro")}</td>
        <td>${cell("roc_auc_ovo_macro")}</td>
        <td>${rEff}</td>
      </tr>`;
    }).join("");
  }

  function renderTable() { renderTableHead(); renderTableBody(); }

  // ================= wire up + resize =================
  function renderAll() {
    renderKPIs();
    renderMainChart();
    renderSubjectChart();
    renderDeltaGrid();
    renderScatter();
    renderTable();
  }

  document.addEventListener("mousemove", e => { if (tt.classList.contains("show")) moveTooltip(e); });
  window.addEventListener("resize", () => { renderMainChart(); renderSubjectChart(); renderScatter(); });

  renderAll();
})();

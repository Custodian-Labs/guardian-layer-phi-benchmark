// Dashboard logic. Reads /data/index.json (list of benchmark runs) and per-run
// summary + sample documents. All data is bundled at build time by
// `scripts/publish_results.py`; the dashboard is a pure static site.

const state = {
  manifest: null,
  currentBenchmark: null,
  rows: [],
  samples: [],
  mode: "type",
  metric: "f1",
  prChart: null,
  leakChart: null,
};

const $ = (sel) => document.querySelector(sel);

async function fetchJSON(path) {
  const resp = await fetch(path);
  if (!resp.ok) throw new Error(`${path} -> ${resp.status}`);
  return resp.json();
}

function isDark() { return document.documentElement.classList.contains("dark"); }

function applyChartTheme() {
  if (!window.Chart) return;
  const dark = isDark();
  Chart.defaults.color = dark ? "rgb(203 213 225)" : "rgb(51 65 85)";
  Chart.defaults.borderColor = dark ? "rgba(148, 163, 184, 0.2)" : "rgba(148, 163, 184, 0.3)";
}

function setupThemeToggle() {
  const btn = $("#theme-toggle");
  if (!btn) return;
  btn.addEventListener("click", () => {
    const next = isDark() ? "light" : "dark";
    document.documentElement.classList.toggle("dark", next === "dark");
    localStorage.setItem("theme", next);
    applyChartTheme();
    if (state.rows.length) renderCharts();
  });
}

async function init() {
  setupThemeToggle();
  applyChartTheme();

  try {
    state.datasets_meta = await fetchJSON("./data/datasets_meta.json");
  } catch { state.datasets_meta = {}; }

  try {
    state.systems_meta = await fetchJSON("./data/systems_meta.json");
  } catch { state.systems_meta = {}; }

  // Load summaries + samples for every run upfront so we can build the
  // cross-benchmark overview and ranking-proof sections.
  try {
    state.manifest = await fetchJSON("./data/index.json");
    state.allSummaries = {};
    state.allSamples = {};
    await Promise.all(state.manifest.runs.map(async (r) => {
      try {
        state.allSummaries[r.id] = await fetchJSON(`./data/${r.summary_path}`);
      } catch {}
      if (r.samples_path) {
        try {
          state.allSamples[r.id] = await fetchJSON(`./data/${r.samples_path}`);
        } catch {}
      }
    }));
    renderOverview();
    setupRankingProof();
  } catch {
    document.body.insertAdjacentHTML("afterbegin",
      `<div class="bg-amber-100 text-amber-900 px-6 py-3 text-sm">
        No benchmark data yet. Run <code>python scripts/run_benchmark.py …</code>
        then <code>python scripts/publish_results.py</code> to populate <code>web/data/</code>.
      </div>`);
    return;
  }

  const benchSelect = $("#benchmark-select");
  state.manifest.runs.forEach((r) => {
    const opt = document.createElement("option");
    opt.value = r.id;
    opt.textContent = `${r.benchmark} · ${r.timestamp} · ${r.mode || "type"} (${r.n_docs} docs)`;
    benchSelect.appendChild(opt);
  });

  $("#last-updated").textContent = `Last updated ${state.manifest.generated_at}`;

  benchSelect.addEventListener("change", (e) => loadRun(e.target.value));
  $("#mode-select").addEventListener("change", (e) => { state.mode = e.target.value; render(); });
  $("#metric-select").addEventListener("change", (e) => { state.metric = e.target.value; render(); });

  document.querySelectorAll("th[data-sort]").forEach((th) => {
    th.addEventListener("click", () => {
      const key = th.dataset.sort;
      state.rows.sort((a, b) => {
        const av = a[key], bv = b[key];
        if (typeof av === "number") return bv - av;
        return String(av).localeCompare(String(bv));
      });
      renderTable();
    });
  });

  if (state.manifest.runs.length) {
    benchSelect.value = state.manifest.runs[0].id;
    loadRun(state.manifest.runs[0].id);
  }
}

async function loadRun(runId) {
  state.currentBenchmark = runId;
  const run = state.manifest.runs.find((r) => r.id === runId);
  state.rows = await fetchJSON(`./data/${run.summary_path}`);
  try {
    state.samples = await fetchJSON(`./data/${run.samples_path}`);
  } catch {
    state.samples = [];
  }
  render();
}

function render() {
  renderDatasetIntro();
  renderExampleIO();
  renderCards();
  renderTable();
  renderCharts();
  renderDrilldown();
}

function renderExampleIO() {
  const root = $("#example-io");
  if (!root) return;
  if (!state.samples?.length) {
    root.innerHTML = `
      <div class="px-4 sm:px-6 py-4">
        <h2 class="text-sm sm:text-base font-semibold">Worked example · input → detection → masked output</h2>
        <p class="text-xs text-slate-500 dark:text-slate-400 mt-1">No samples bundled for this run; per-document text is only published for synthetic / open-license benchmarks.</p>
      </div>`;
    return;
  }

  // Pick the first sample with predictions (drop the empty placeholder rows).
  const sample = state.samples.find((s) => s.predictions && Object.keys(s.predictions).length) || state.samples[0];
  if (!sample || !sample.text) {
    root.innerHTML = `
      <div class="px-4 sm:px-6 py-4">
        <h2 class="text-sm sm:text-base font-semibold">Worked example</h2>
        <p class="text-xs text-slate-500 dark:text-slate-400 mt-1">Sample doc has no text payload.</p>
      </div>`;
    return;
  }

  // Within the sample, pick the system with the highest per-doc F1 for this drilldown.
  const sysEntries = Object.entries(sample.predictions);
  sysEntries.sort((a, b) => (b[1].score?.f1 ?? 0) - (a[1].score?.f1 ?? 0));
  const [bestName, bestPred] = sysEntries[0] || ["—", { spans: [], score: {} }];
  const bestMeta = sysMeta(bestName);

  const goldSpans = sample.gold_spans || [];
  const predSpans = bestPred.spans || [];
  const maskedText = applyMask(sample.text, predSpans);
  const predJson = predSpans.map((s) => ({
    start: s.start, end: s.end, label: s.label, text: sample.text.slice(s.start, s.end),
  }));
  const predJsonStr = JSON.stringify({ spans: predJson }, null, 2);

  root.innerHTML = `
    <div class="px-4 sm:px-6 py-3 sm:py-4 border-b border-slate-200 dark:border-slate-800">
      <h2 class="text-sm sm:text-base font-semibold">Worked example · input → detection → masked output</h2>
      <p class="text-xs text-slate-500 dark:text-slate-400 mt-1">
        One real document from this benchmark, taken through the strongest system on this doc
        (<span class="font-medium text-slate-700 dark:text-slate-300">${escapeHtml(bestMeta.display)}</span>,
        <code class="text-[11px]">${escapeHtml(bestMeta.model_id)}</code>,
        per-doc F1 ${pct(bestPred.score?.f1)}).
      </p>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-px bg-slate-200 dark:bg-slate-800">
      <!-- 1. Input -->
      <div class="bg-white dark:bg-slate-900 p-4">
        <div class="flex items-center justify-between mb-2">
          <span class="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">1 · Input (with gold PHI)</span>
          <span class="metric-pill">${sample.gold_n ?? goldSpans.length} gold span${goldSpans.length === 1 ? "" : "s"}</span>
        </div>
        <div class="doc-text bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded p-3 max-h-72 overflow-y-auto">${highlight(sample.text, goldSpans, "gold")}</div>
      </div>

      <!-- 2. System output (JSON) -->
      <div class="bg-white dark:bg-slate-900 p-4">
        <div class="flex items-center justify-between mb-2">
          <span class="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">2 · Detector output</span>
          <span class="metric-pill ${bestPred.score?.f1 > 0.8 ? "good" : "bad"}">F1 ${pct(bestPred.score?.f1)}</span>
        </div>
        <pre class="doc-text bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded p-3 max-h-72 overflow-y-auto text-[11px]">${escapeHtml(predJsonStr)}</pre>
      </div>

      <!-- 3. After masking -->
      <div class="bg-white dark:bg-slate-900 p-4">
        <div class="flex items-center justify-between mb-2">
          <span class="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">3 · Masked output (safe to send)</span>
          <span class="metric-pill good">[LABEL] placeholders</span>
        </div>
        <div class="doc-text bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded p-3 max-h-72 overflow-y-auto">${escapeHtml(maskedText)}</div>
      </div>
    </div>
  `;
}

function applyMask(text, spans) {
  if (!spans || !spans.length) return text;
  const sorted = [...spans].sort((a, b) => a.start - b.start);
  let out = "", cursor = 0;
  for (const s of sorted) {
    if (s.start < cursor) continue;
    out += text.slice(cursor, s.start);
    out += `[${s.label}]`;
    cursor = s.end;
  }
  out += text.slice(cursor);
  return out;
}

function renderDatasetIntro() {
  const root = $("#dataset-intro");
  const run = state.manifest?.runs?.find((r) => r.id === state.currentBenchmark);
  const bench = run?.benchmark;
  const meta = state.datasets_meta?.[bench];
  if (!meta) { root.innerHTML = ""; return; }

  const entityList = (meta.entity_types || []).map((e) => `<code class="metric-pill">${escapeHtml(e)}</code>`).join(" ");
  root.innerHTML = `
    <div class="flex items-start justify-between gap-6 flex-wrap">
      <div class="flex-1 min-w-[280px]">
        <div class="text-xs uppercase tracking-wide text-slate-500">Benchmark</div>
        <h2 class="text-lg font-semibold">${escapeHtml(meta.title)}</h2>
        <div class="text-sm text-slate-600">${escapeHtml(meta.subtitle)}</div>
        <p class="mt-2 text-sm text-slate-700">${escapeHtml(meta.why_we_include_it || "")}</p>
        <div class="mt-3 grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
          <div><span class="text-slate-500">Language:</span> ${escapeHtml(meta.language || "?")}</div>
          <div><span class="text-slate-500">Size:</span> ${escapeHtml(meta.size || "?")}</div>
          <div><span class="text-slate-500">License:</span> ${escapeHtml(meta.license || "?")}</div>
          <div><span class="text-slate-500">DUA needed:</span> ${meta.dua_required ? "<span class='text-rose-700'>yes</span>" : "<span class='text-emerald-700'>no</span>"}</div>
        </div>
        <div class="mt-3 text-xs"><span class="text-slate-500">PHI density:</span> ${escapeHtml(meta.phi_density || "?")}</div>
        <div class="mt-2 text-xs flex flex-wrap gap-1">${entityList}</div>
        ${meta.url ? `<div class="mt-2 text-xs"><a class="text-indigo-600 hover:underline" href="${meta.url}" target="_blank" rel="noopener">${escapeHtml(meta.url)}</a></div>` : ""}
      </div>
      <div class="flex-1 min-w-[280px]">
        <div class="text-xs uppercase tracking-wide text-slate-500 mb-1">Example excerpt</div>
        <pre class="doc-text bg-slate-50 border border-slate-200 rounded p-3 max-h-44 overflow-y-auto">${escapeHtml(meta.example || "")}</pre>
      </div>
    </div>
  `;
}

function sysMeta(name) {
  return state.systems_meta?.[name] || { display: name, model_id: name, provider: "—", size: "—", access: "—" };
}

function sysDisplay(name) {
  return sysMeta(name).display || name;
}

function renderCards() {
  const best = (key, lowerBetter = false) => {
    const sorted = [...state.rows].sort((a, b) => lowerBetter ? a[key] - b[key] : b[key] - a[key]);
    return sorted[0];
  };
  const cards = [
    { title: "Best F1", row: best("f1"), key: "f1", fmt: pct },
    { title: "Best Recall", row: best("recall"), key: "recall", fmt: pct },
    { title: "Lowest Leakage", row: best("leakage_rate", true), key: "leakage_rate", fmt: pct },
    { title: "Lowest Char Leak", row: best("char_leakage_rate", true), key: "char_leakage_rate", fmt: pct },
  ];
  $("#summary-cards").innerHTML = cards.map((c) => `
    <div class="bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-800 p-4 shadow-sm">
      <div class="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">${c.title}</div>
      <div class="mt-1 text-2xl font-semibold">${c.row ? c.fmt(c.row[c.key]) : "—"}</div>
      <div class="mt-1 text-xs text-slate-500 dark:text-slate-400 truncate" title="${c.row ? escapeHtml(sysMeta(c.row.system).model_id) : ""}">${c.row ? escapeHtml(sysDisplay(c.row.system)) : ""}</div>
    </div>
  `).join("");
}

function _bestByCol(key, lowerBetter = false) {
  if (!state.rows.length) return null;
  const vals = state.rows.map((r) => r[key]).filter((v) => v != null && !Number.isNaN(v));
  if (!vals.length) return null;
  return lowerBetter ? Math.min(...vals) : Math.max(...vals);
}

function renderTable() {
  const tbody = $("#comparison-tbody");
  const metric = state.metric;

  // Best value per column (used to colour each cell independently).
  const bestP = _bestByCol("precision");
  const bestR = _bestByCol("recall");
  const bestF1 = _bestByCol("f1");
  const bestLeak = _bestByCol("leakage_rate", true);
  const bestCharLeak = _bestByCol("char_leakage_rate", true);

  const cell = (val, best, fmt = pct) => {
    if (val == null) return `<td class="px-3 sm:px-4 py-2 sm:py-2.5 text-right tabular-nums">—</td>`;
    const isBest = best != null && Math.abs(val - best) < 1e-9;
    return `<td class="px-3 sm:px-4 py-2 sm:py-2.5 text-right tabular-nums ${isBest ? "cell-best font-semibold" : ""}">${fmt(val)}</td>`;
  };

  const lowerBetter = metric.includes("leakage");
  const metricVals = state.rows.map((r) => r[metric] ?? 0);
  const metricBest = lowerBetter ? Math.min(...metricVals) : Math.max(...metricVals);

  tbody.innerHTML = state.rows.map((r) => {
    const meta = sysMeta(r.system);
    const isRowBest = Math.abs(r[metric] - metricBest) < 1e-9;
    return `
      <tr class="${isRowBest ? "highlight-row" : ""}">
        <td class="px-3 sm:px-4 py-2 sm:py-2.5">
          <div class="font-medium flex items-center gap-1.5 flex-wrap">
            ${escapeHtml(meta.display || r.system)}
            ${meta.note ? `<span class="metric-pill bad" title="${escapeHtml(meta.note)}">out-of-domain</span>` : ""}
          </div>
          <div class="text-xs text-slate-500 dark:text-slate-400 font-mono"><code>${escapeHtml(meta.model_id || "")}</code></div>
          <div class="text-[11px] text-slate-400 dark:text-slate-500 mt-0.5">${escapeHtml(meta.size || "")} · ${escapeHtml(meta.access || "")}</div>
        </td>
        ${cell(r.precision, bestP)}
        ${cell(r.recall, bestR)}
        ${cell(r.f1, bestF1)}
        ${cell(r.leakage_rate, bestLeak)}
        ${cell(r.char_leakage_rate, bestCharLeak)}
        <td class="px-3 sm:px-4 py-2 sm:py-2.5 text-right text-slate-500 dark:text-slate-400">${r.n_docs}</td>
      </tr>
    `;
  }).join("");

  $("#row-meta").textContent = `${state.rows.length} systems · row highlight = best ${metric}, cell highlight = best per column`;
}

function renderCharts() {
  const labels = state.rows.map((r) => sysDisplay(r.system));
  const recall = state.rows.map((r) => r.recall);
  const precision = state.rows.map((r) => r.precision);
  const leak = state.rows.map((r) => r.leakage_rate);
  const charLeak = state.rows.map((r) => r.char_leakage_rate);

  if (state.prChart) state.prChart.destroy();
  state.prChart = new Chart($("#pr-chart"), {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "Precision", data: precision, backgroundColor: "rgba(99,102,241,0.7)" },
        { label: "Recall",    data: recall,    backgroundColor: "rgba(16,185,129,0.7)" },
      ],
    },
    options: { responsive: true, scales: { y: { min: 0, max: 1 } } },
  });

  if (state.leakChart) state.leakChart.destroy();
  state.leakChart = new Chart($("#leak-chart"), {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "Span leakage",     data: leak,     backgroundColor: "rgba(244,63,94,0.55)" },
        { label: "Char leakage",     data: charLeak, backgroundColor: "rgba(244,114,182,0.55)" },
      ],
    },
    options: { responsive: true, scales: { y: { min: 0, max: 1 } } },
  });
}

function renderDrilldown() {
  const root = $("#drilldown");
  if (!state.samples.length) {
    root.innerHTML = `<p class="text-sm text-slate-500">No per-document samples bundled for this run. Pass <code>--publish-samples</code> to <code>scripts/publish_results.py</code> when the dataset license allows it.</p>`;
    return;
  }
  root.innerHTML = state.samples.map((doc) => {
    const systems = Object.entries(doc.predictions);
    return `
      <div class="border border-slate-200 rounded-md p-4">
        <div class="flex items-center justify-between text-xs text-slate-500 mb-2">
          <span>doc <code>${doc.doc_id}</code></span>
          <span>${doc.gold_n} gold PHI spans</span>
        </div>
        <div class="doc-text mb-3">${highlight(doc.text, doc.gold_spans, "gold")}</div>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
          ${systems.map(([sysName, p]) => `
            <div class="border border-slate-100 rounded p-3">
              <div class="flex items-center justify-between mb-1">
                <span class="text-xs font-semibold">${sysName}</span>
                <span class="metric-pill ${p.score.f1 > 0.8 ? "good" : "bad"}">F1 ${pct(p.score.f1)}</span>
              </div>
              <div class="doc-text">${highlight(doc.text, p.spans, sysName)}</div>
            </div>
          `).join("")}
        </div>
      </div>
    `;
  }).join("");
}

function highlight(text, spans, system) {
  if (!spans || !spans.length) return escapeHtml(text);
  const sorted = [...spans].sort((a, b) => a.start - b.start);
  let out = "", cursor = 0;
  for (const s of sorted) {
    if (s.start < cursor) continue;
    out += escapeHtml(text.slice(cursor, s.start));
    out += `<span class="phi-span" data-system="${system}" title="${escapeHtml(s.label || "")}">${escapeHtml(text.slice(s.start, s.end))}</span>`;
    cursor = s.end;
  }
  out += escapeHtml(text.slice(cursor));
  return out;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;",
  }[c]));
}

function pct(x) {
  if (x === undefined || x === null || Number.isNaN(x)) return "—";
  return (x * 100).toFixed(1) + "%";
}

// ───────────────────────────────────────────────────────────────
// NEW: cross-benchmark overview heatmap
// ───────────────────────────────────────────────────────────────
function renderOverview() {
  const root = $("#overview-table");
  if (!root || !state.manifest) return;

  const allRuns = state.manifest.runs;
  const metricSel = $("#overview-metric");
  const metric = metricSel?.value || "f1";
  const lowerBetter = metric.includes("leakage");

  // Original vs Custodian-transformed tracks. A transformed run's benchmark
  // is "<base>_transformed"; pair it with the original run of <base>.
  const trackSel = $("#overview-track");
  const track = trackSel?.value || "original";
  const isTransformed = (r) => r.benchmark.endsWith("_transformed");
  const baseOf = (r) => r.benchmark.replace(/_transformed$/, "");
  const origRuns = allRuns.filter((r) => !isTransformed(r));
  const transByBase = Object.fromEntries(
    allRuns.filter(isTransformed).map((r) => [baseOf(r), r]),
  );
  // Column set is always the original benchmarks (stable layout); in
  // transformed/delta modes cells read from the paired transformed run.
  const runs = track === "transformed"
    ? origRuns.filter((r) => transByBase[r.benchmark])
    : origRuns;

  // Build matrix: system -> {benchId: value}
  const rawVal = (sys, runId) => {
    const row = (state.allSummaries[runId] || []).find((s) => s.system === sys);
    return row ? row[metric] : null;
  };
  const systems = new Set();
  const runsForSystems = track === "original" ? origRuns : allRuns.filter(isTransformed);
  runsForSystems.forEach((r) => (state.allSummaries[r.id] || []).forEach((s) => systems.add(s.system)));
  const sysList = [...systems];

  const valFor = (sys, origRunId) => {
    const origRun = runs.find((r) => r.id === origRunId);
    if (!origRun) return null;
    if (track === "original") return rawVal(sys, origRunId);
    const tRun = transByBase[origRun.benchmark];
    if (!tRun) return null;
    const tv = rawVal(sys, tRun.id);
    if (track === "transformed") return tv;
    // delta
    const ov = rawVal(sys, origRunId);
    return tv == null || ov == null ? null : tv - ov;
  };

  const coverageFor = (sys) =>
    runs.filter((r) => valFor(sys, r.id) != null).length;

  // Mean is only meaningful when a system ran on every benchmark. Systems
  // with partial coverage (e.g. qwen-thinking, kept only for ASQ-PHI) get a
  // null mean so they don't rank against full-coverage systems on a single
  // cherry-picked score.
  const meanFor = (sys) => {
    if (coverageFor(sys) < runs.length) return null;
    const vals = runs.map((r) => valFor(sys, r.id)).filter((v) => v != null);
    return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
  };

  // Sort: full-coverage systems first (by mean), partial-coverage last.
  sysList.sort((a, b) => {
    const ma = meanFor(a), mb = meanFor(b);
    if (ma == null && mb == null) return coverageFor(b) - coverageFor(a);
    if (ma == null) return 1;
    if (mb == null) return -1;
    return lowerBetter ? ma - mb : mb - ma;
  });

  // Per-benchmark max/min for cell colour scaling
  const benchExtrema = {};
  runs.forEach((r) => {
    const vs = sysList.map((s) => valFor(s, r.id)).filter((v) => v != null);
    benchExtrema[r.id] = { min: Math.min(...vs), max: Math.max(...vs) };
  });

  // In delta mode the scale is centred on 0: no change = neutral green-ish,
  // degradation = red, improvement = deeper green. Otherwise scale min..max.
  const tFor = (val, min, max) => {
    if (track === "delta") {
      const eff = lowerBetter ? -val : val;             // positive = better
      const scale = Math.max(Math.abs(min), Math.abs(max), 0.02);
      return Math.max(0, Math.min(1, 0.5 + eff / (2 * scale)));
    }
    const span = (max - min) || 1;
    let t = (val - min) / span;
    return lowerBetter ? 1 - t : t;
  };
  const heatCss = (val, min, max) => {
    if (val == null) return "background:transparent;color:#94a3b8;";
    const t = tFor(val, min, max);
    // green (good) -> yellow -> red (bad)
    const h = Math.round(t * 130);     // 0=red, 130=green
    const s = 70, l = 86;
    return `background:hsl(${h},${s}%,${l}%);color:#0f172a;`;
  };
  const heatCssDark = (val, min, max) => {
    if (val == null) return "background:transparent;color:#64748b;";
    const t = tFor(val, min, max);
    const h = Math.round(t * 130);
    return `background:hsl(${h},45%,28%);color:#f1f5f9;`;
  };

  const isDarkMode = isDark();
  const cellStyle = (val, runId) => {
    const ext = benchExtrema[runId];
    return isDarkMode ? heatCssDark(val, ext.min, ext.max) : heatCss(val, ext.min, ext.max);
  };

  // Build mean column extrema across systems
  const meanVals = sysList.map(meanFor).filter((v) => v != null);
  const meanExt = { min: Math.min(...meanVals), max: Math.max(...meanVals) };

  const fmt = (x) => {
    if (x == null) return "—";
    if (track === "delta") {
      const v = (x * 100).toFixed(1);
      return x > 0 ? `+${v}` : v;       // signed percentage points
    }
    return (x * 100).toFixed(1);
  };
  const benchLabel = (r) => {
    const meta = state.datasets_meta?.[r.benchmark];
    return meta?.title || r.benchmark;
  };
  // Short header used in the overview heatmap so the table fits without
  // horizontal scroll. Tooltip on the <th> shows the full name.
  const benchShort = (r) => {
    const b = r.benchmark;
    if (b === "asq_phi") return "ASQ-PHI";
    if (b === "meddocan") return "MEDDOCAN";
    if (b === "multiconer_v2") return "MultiCoNER";
    if (b === "pii_masking_300k") return "PII (en)";
    if (b === "pii_masking_300k_dutch") return "PII (nl)";
    if (b === "pii_masking_300k_french") return "PII (fr)";
    if (b === "pii_masking_300k_german") return "PII (de)";
    return b;
  };

  const head = `
    <thead class="bg-slate-50 dark:bg-slate-800 text-slate-600 dark:text-slate-300 sticky top-0">
      <tr>
        <th class="px-3 py-2 text-left font-medium whitespace-nowrap sticky left-0 bg-slate-50 dark:bg-slate-800 z-20">System</th>
        ${runs.map((r) => `<th class="px-2 py-2 text-center font-medium whitespace-nowrap" title="${escapeHtml(benchLabel(r))} · ${escapeHtml(r.id)}">${escapeHtml(benchShort(r))}<br><span class="text-[10px] text-slate-400">n=${r.n_docs}</span></th>`).join("")}
        <th class="px-3 py-2 text-center font-semibold whitespace-nowrap border-l border-slate-300 dark:border-slate-700 sticky right-0 bg-slate-50 dark:bg-slate-800 z-20">Mean</th>
      </tr>
    </thead>
  `;

  const body = sysList.map((sys) => {
    const meta = sysMeta(sys);
    const cells = runs.map((r) => {
      const v = valFor(sys, r.id);
      return `<td class="px-2 py-2 text-center tabular-nums" style="${cellStyle(v, r.id)}" title="${escapeHtml(sys)} on ${escapeHtml(r.id)}">${fmt(v)}</td>`;
    }).join("");
    const m = meanFor(sys);
    const meanStyle = isDarkMode ? heatCssDark(m, meanExt.min, meanExt.max) : heatCss(m, meanExt.min, meanExt.max);
    const noteBadge = meta.note ? `<span class="metric-pill bad text-[9px]" title="${escapeHtml(meta.note)}">out-of-domain</span>` : "";
    return `
      <tr class="border-t border-slate-100 dark:border-slate-800">
        <td class="px-3 py-2 whitespace-nowrap sticky left-0 bg-white dark:bg-slate-900 z-10">
          <div class="font-medium text-xs sm:text-sm flex items-center gap-1.5 flex-wrap">${escapeHtml(meta.display)}${noteBadge}</div>
          <div class="text-[10px] text-slate-500"><code>${escapeHtml(meta.model_id)}</code></div>
        </td>
        ${cells}
        <td class="px-3 py-2 text-center tabular-nums font-semibold border-l border-slate-200 dark:border-slate-700 sticky right-0 z-10" style="${meanStyle}">${fmt(m)}</td>
      </tr>
    `;
  }).join("");

  root.innerHTML = head + `<tbody>${body}</tbody>`;

  // Footnote: explain track + any partial-coverage row (e.g. qwen-thinking).
  const fn = $("#overview-footnote");
  const trackNote =
    track === "delta"
      ? `<strong>Δ view:</strong> each cell is (score on Custodian-transformed
         docs) − (score on original docs), in percentage points, for the same
         250 documents per benchmark. PHI was replaced by Guardian Layer
         transform with plausible surrogates and gold spans were remapped to
         the surrogate locations. Cells near 0 mean the transformation
         preserves detector accuracy. `
      : track === "transformed"
        ? `<strong>Custodian-transformed:</strong> same 250 documents per
           benchmark, with PHI substituted by Guardian Layer
           <code>transform</code> (top-1 surrogate) and gold spans remapped. `
        : "";
  if (fn) {
    const partial = sysList.filter((s) => coverageFor(s) < runs.length);
    if (trackNote && !partial.length) fn.innerHTML = trackNote;
    if (partial.length) {
      const names = partial.map((s) => sysMeta(s).display).join(", ");
      fn.innerHTML = `
        <strong>Note on ${escapeHtml(names)}:</strong>
        shown for ASQ-PHI only. Qwen 3.5-4B with <code>enable_thinking=True</code>
        is the same weights as Qwen 3.5-4B, differing only by a decoding flag,
        so it is not an independent system — it is reported as an ablation.
        Its chain-of-thought reliably terminates within the token budget only
        on ASQ-PHI's short, sparse-PHI queries (F1 0.73, beating the
        no-thinking variant); on the denser benchmarks the CoT exhausts the
        2,500-token generation budget before emitting JSON on ~2/3 of
        documents, so those cells are omitted (—) rather than reported as
        misleading near-zero scores. Mean is computed only over
        full-coverage systems.`;
      if (trackNote) fn.innerHTML = trackNote + "<br><br>" + fn.innerHTML;
    } else if (!trackNote) {
      fn.innerHTML = "";
    }
  }

  // Hook metric + track switchers (idempotent — bound once).
  if (metricSel && !metricSel.dataset.bound) {
    metricSel.addEventListener("change", renderOverview);
    metricSel.dataset.bound = "1";
  }
  if (trackSel && !trackSel.dataset.bound) {
    trackSel.addEventListener("change", renderOverview);
    trackSel.dataset.bound = "1";
  }
}

// ───────────────────────────────────────────────────────────────
// NEW: ranking-proof panel — same doc, 6 systems side by side
// ───────────────────────────────────────────────────────────────
function setupRankingProof() {
  const sel = $("#ranking-bench");
  const next = $("#ranking-next");
  if (!sel || !next) return;

  const runs = state.manifest.runs.filter((r) => state.allSamples[r.id]?.length);
  if (!runs.length) return;

  sel.innerHTML = runs.map((r) => {
    const meta = state.datasets_meta?.[r.benchmark];
    const label = meta?.title || r.benchmark;
    return `<option value="${r.id}">${escapeHtml(label)} (${r.n_docs} docs)</option>`;
  }).join("");

  state.rankingState = { runId: sel.value, docIdx: 0 };

  sel.addEventListener("change", () => {
    state.rankingState.runId = sel.value;
    state.rankingState.docIdx = 0;
    renderRankingProof();
  });
  next.addEventListener("click", () => {
    const samples = state.allSamples[state.rankingState.runId] || [];
    state.rankingState.docIdx = (state.rankingState.docIdx + 1) % samples.length;
    renderRankingProof();
  });

  renderRankingProof();
}

function renderRankingProof() {
  const root = $("#ranking-proof-body");
  const { runId, docIdx } = state.rankingState || {};
  const samples = (state.allSamples[runId] || []).filter((s) => s.text);
  if (!samples.length) {
    root.innerHTML = `<p class="text-sm text-slate-500 dark:text-slate-400">No per-doc samples available for this benchmark.</p>`;
    return;
  }
  const sample = samples[docIdx % samples.length];

  const entries = Object.entries(sample.predictions || {});
  // Sort by per-doc F1 (best to worst)
  entries.sort((a, b) => (b[1].score?.f1 ?? 0) - (a[1].score?.f1 ?? 0));

  const gold = sample.gold_spans || [];

  const panels = entries.map(([sysName, p], i) => {
    const meta = sysMeta(sysName);
    const f1 = p.score?.f1 ?? 0;
    const score = p.score || {};

    // Categorise spans: tp (correct), fp (extra), and missed gold spans (fn)
    // Using span-exact match (start+end) since dashboard mode is "type".
    const predKeys = new Set((p.spans || []).map((s) => `${s.start}-${s.end}`));
    const goldKeys = new Set(gold.map((s) => `${s.start}-${s.end}`));
    const tpSpans = (p.spans || []).filter((s) => goldKeys.has(`${s.start}-${s.end}`));
    const fpSpans = (p.spans || []).filter((s) => !goldKeys.has(`${s.start}-${s.end}`));
    const missedSpans = gold.filter((s) => !predKeys.has(`${s.start}-${s.end}`));

    const allMarks = [
      ...tpSpans.map((s) => ({ ...s, kind: "tp" })),
      ...fpSpans.map((s) => ({ ...s, kind: "fp" })),
      ...missedSpans.map((s) => ({ ...s, kind: "missed" })),
    ];

    const rankBadge = `<span class="inline-flex items-center justify-center w-6 h-6 rounded-full bg-indigo-100 text-indigo-700 dark:bg-indigo-900 dark:text-indigo-200 text-xs font-bold">${i + 1}</span>`;
    const f1Badge = `<span class="metric-pill ${f1 >= 0.7 ? "good" : f1 >= 0.3 ? "" : "bad"}">F1 ${pct(f1)}</span>`;
    const counts = `<span class="text-[11px] text-slate-500 dark:text-slate-400">tp ${tpSpans.length} · fp ${fpSpans.length} · missed ${missedSpans.length}</span>`;

    return `
      <div class="border border-slate-200 dark:border-slate-800 rounded-md p-3">
        <div class="flex items-center justify-between gap-2 mb-2 flex-wrap">
          <div class="flex items-center gap-2 min-w-0">
            ${rankBadge}
            <div class="min-w-0">
              <div class="text-sm font-semibold truncate">${escapeHtml(meta.display)}</div>
              <div class="text-[11px] text-slate-500 dark:text-slate-400 truncate"><code>${escapeHtml(meta.model_id)}</code></div>
            </div>
          </div>
          <div class="flex items-center gap-2 shrink-0">
            ${counts}
            ${f1Badge}
          </div>
        </div>
        <div class="doc-text bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded p-3 max-h-44 overflow-y-auto">${rankingHighlight(sample.text, allMarks)}</div>
      </div>
    `;
  }).join("");

  const goldHighlight = rankingHighlight(sample.text, gold.map((s) => ({ ...s, kind: "tp" })));

  root.innerHTML = `
    <div class="text-xs text-slate-500 dark:text-slate-400 mb-1">
      doc <code>${escapeHtml(sample.doc_id)}</code> · ${gold.length} gold PHI spans · sample ${docIdx + 1} / ${samples.length}
    </div>
    <div class="border border-emerald-200 bg-emerald-50/40 dark:bg-emerald-900/15 dark:border-emerald-800 rounded-md p-3">
      <div class="text-xs font-semibold uppercase tracking-wide text-emerald-700 dark:text-emerald-300 mb-2">Gold (reference)</div>
      <div class="doc-text">${goldHighlight}</div>
    </div>
    ${panels}
  `;
}

function rankingHighlight(text, marks) {
  if (!marks || !marks.length) return escapeHtml(text);
  const sorted = [...marks].sort((a, b) => a.start - b.start);
  let out = "", cursor = 0;
  for (const m of sorted) {
    if (m.start < cursor) continue;
    out += escapeHtml(text.slice(cursor, m.start));
    out += `<span class="phi-mark phi-${m.kind}" title="${escapeHtml(m.label || "")} (${m.kind})">${escapeHtml(text.slice(m.start, m.end))}</span>`;
    cursor = m.end;
  }
  out += escapeHtml(text.slice(cursor));
  return out;
}

init();

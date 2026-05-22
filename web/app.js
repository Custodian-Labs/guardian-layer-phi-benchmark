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

  try {
    state.manifest = await fetchJSON("./data/index.json");
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
          <div class="font-medium">${escapeHtml(meta.display || r.system)}</div>
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

init();

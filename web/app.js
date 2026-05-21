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

async function init() {
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
  renderCards();
  renderTable();
  renderCharts();
  renderDrilldown();
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
    <div class="bg-white rounded-lg border border-slate-200 p-4 shadow-sm">
      <div class="text-xs uppercase tracking-wide text-slate-500">${c.title}</div>
      <div class="mt-1 text-2xl font-semibold">${c.row ? c.fmt(c.row[c.key]) : "—"}</div>
      <div class="mt-1 text-xs text-slate-500">${c.row ? c.row.system : ""}</div>
    </div>
  `).join("");
}

function renderTable() {
  const tbody = $("#comparison-tbody");
  const metric = state.metric;
  const metricVals = state.rows.map((r) => r[metric] ?? 0);
  const max = Math.max(...metricVals);

  tbody.innerHTML = state.rows.map((r) => {
    const lowerBetter = metric.includes("leakage");
    const isBest = lowerBetter ? r[metric] === Math.min(...metricVals) : r[metric] === max;
    return `
      <tr class="${isBest ? "highlight-row" : ""}">
        <td class="px-4 py-2.5 font-medium">${r.system}</td>
        <td class="px-4 py-2.5 text-right tabular-nums">${pct(r.precision)}</td>
        <td class="px-4 py-2.5 text-right tabular-nums">${pct(r.recall)}</td>
        <td class="px-4 py-2.5 text-right tabular-nums">${pct(r.f1)}</td>
        <td class="px-4 py-2.5 text-right tabular-nums">${pct(r.leakage_rate)}</td>
        <td class="px-4 py-2.5 text-right tabular-nums">${pct(r.char_leakage_rate)}</td>
        <td class="px-4 py-2.5 text-right text-slate-500">${r.n_docs}</td>
      </tr>
    `;
  }).join("");

  $("#row-meta").textContent = `${state.rows.length} systems · highlighted by ${metric}`;
}

function renderCharts() {
  const labels = state.rows.map((r) => r.system);
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

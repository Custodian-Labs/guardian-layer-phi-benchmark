#!/usr/bin/env python3
"""Generate web/annotate.html: a self-contained, offline single-page tool for
reviewing the surrogate-quality annotation sheet fast.

Reads data/annotation/surrogate_quality_prefilled.csv and embeds the rows into
a static HTML page (no fetch/CORS, works from file:// or any host). The page:
  * shows one span at a time (original -> surrogate + highlighted context);
  * pre-selects the heuristic's valid / type_consistent / failure guesses;
  * keyboard shortcuts for fast labelling; autosaves to localStorage;
  * exports a CSV in the exact schema (import to resume or merge two raters).

Re-run this whenever the prefilled CSV changes. Deterministic, CPU only.
"""
from __future__ import annotations
import csv, json, os

ROOT = os.path.join(os.path.dirname(__file__), os.pardir)
_AUDIT = os.path.join(ROOT, "data", "annotation", "surrogate_quality_audit.csv")
_PREFILL = os.path.join(ROOT, "data", "annotation", "surrogate_quality_prefilled.csv")
# Prefer the fuller manual-audit pass as the review starting point (authors
# confirm 46 flags rather than re-scan 200); fall back to the heuristic sheet.
SRC = _AUDIT if os.path.exists(_AUDIT) else _PREFILL
OUT = os.path.join(ROOT, "web", "annotate.html")
COLS = ["id", "benchmark", "doc_id", "type", "original", "surrogate", "context",
        "failure", "valid", "type_consistent", "reviewed", "notes"]

HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Surrogate-quality annotation</title>
<style>
:root{--accent:#0F6E5C;--clay:#BD5B36;--khaki:#7A6A3A;--paper:#F3F5F3;--line:#C5CDC8;--muted:#5C6B64;--ink:#1c2420;--card:#fff}
@media (prefers-color-scheme:dark){:root{--paper:#121613;--line:#2c3631;--ink:#e6ece8;--muted:#9fb0a8;--card:#1a201d}}
*{box-sizing:border-box}
body{margin:0;font:15px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:var(--paper);color:var(--ink)}
header{position:sticky;top:0;background:var(--card);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;z-index:5}
h1{font-size:15px;margin:0;font-weight:700;color:var(--accent)}
.bar{flex:1;height:8px;background:var(--line);border-radius:4px;overflow:hidden;min-width:120px}
.bar>i{display:block;height:100%;background:var(--accent);width:0}
.btn{font:inherit;padding:5px 11px;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:7px;cursor:pointer}
.btn:hover{border-color:var(--accent)}
.btn.k{border-color:var(--accent);color:var(--accent)}
select,input[type=text]{font:inherit;padding:5px 8px;border:1px solid var(--line);border-radius:7px;background:var(--card);color:var(--ink)}
main{max-width:820px;margin:18px auto;padding:0 16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px 22px}
.meta{display:flex;gap:8px;align-items:center;color:var(--muted);font-size:12.5px;margin-bottom:14px}
.tag{padding:2px 9px;border-radius:999px;border:1px solid var(--line);font-weight:600}
.tag.b{color:var(--accent);border-color:var(--accent)}
.tag.t{color:var(--khaki);border-color:var(--khaki)}
.ov{font-size:14px;margin:2px 0 16px}
.ov .o{color:var(--clay)}
.ov .arw{color:var(--muted);margin:0 8px}
.ov .s{color:var(--accent);font-weight:700}
.ctx{background:var(--paper);border:1px solid var(--line);border-radius:8px;padding:11px 13px;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12.5px;white-space:pre-wrap;word-break:break-word;margin-bottom:18px}
.ctx mark{background:rgba(15,110,92,.18);color:inherit;border-radius:3px;padding:0 2px;font-weight:700}
.row{display:flex;gap:22px;flex-wrap:wrap;margin-bottom:14px}
.grp{display:flex;flex-direction:column;gap:6px}
.grp>span{font-size:11.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.opts{display:flex;gap:6px;flex-wrap:wrap}
.opt{padding:5px 11px;border:1px solid var(--line);border-radius:7px;cursor:pointer;user-select:none;font-size:13.5px}
.opt.sel{background:var(--accent);color:#fff;border-color:var(--accent)}
.opt.no.sel{background:var(--clay);border-color:var(--clay)}
.opt small{opacity:.6;margin-left:5px}
.notes{width:100%}
.nav{display:flex;justify-content:space-between;align-items:center;margin-top:18px}
.rev{display:flex;align-items:center;gap:7px;color:var(--muted);font-size:13px}
.rev.on{color:var(--accent);font-weight:600}
.hint{color:var(--muted);font-size:12px;margin-top:10px;text-align:center}
kbd{border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;padding:0 5px;font-family:ui-monospace,monospace;font-size:11px}
</style></head><body>
<header>
  <h1>Surrogate annotation</h1>
  <div class="bar"><i id="prog"></i></div>
  <span id="count" style="font-size:12.5px;color:var(--muted)"></span>
  <select id="filter">
    <option value="all">All</option>
    <option value="unrev">Unreviewed</option>
    <option value="flag">Auto-flagged (valid=N)</option>
  </select>
  <button class="btn" id="imp">Import CSV</button>
  <button class="btn k" id="exp">Export CSV</button>
  <input type="file" id="file" accept=".csv" hidden>
</header>
<main>
  <div class="card" id="card"></div>
  <div class="hint">
    <kbd>V</kbd> valid &nbsp; <kbd>T</kbd> type &nbsp; <kbd>0</kbd> none <kbd>X</kbd> x-mask <kbd>G</kbd> garbled <kbd>S</kbd> salience <kbd>O</kbd> other
    &nbsp;·&nbsp; <kbd>&larr;</kbd>/<kbd>&rarr;</kbd> prev/next &nbsp; autosaves locally
  </div>
</main>
<script>
const COLS=__COLS__, DATA=__DATA__;
const FAILS=[["none","0"],["x_masked","x"],["truncated_garbled","g"],["salience_loss","s"],["other","o"]];
const KEY="custodian_surrogate_annot_v1";
let rows=DATA.map(r=>Object.assign({},r));
try{const saved=JSON.parse(localStorage.getItem(KEY)||"null");
    if(saved&&saved.length===rows.length) rows=saved;}catch(e){}
let order=[], pos=0;
const $=s=>document.querySelector(s);

function applyFilter(){
  const f=$("#filter").value;
  order=rows.map((r,i)=>i).filter(i=>{
    if(f==="unrev") return (rows[i].reviewed||"").toUpperCase()!=="Y";
    if(f==="flag")  return (rows[i].valid||"").toUpperCase()==="N";
    return true;});
  if(order.length===0) order=rows.map((r,i)=>i);
  if(pos>=order.length) pos=0;
}
function save(){localStorage.setItem(KEY,JSON.stringify(rows));}
function esc(s){return (s||"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));}
function ctxHtml(s){return esc(s).replace(/⟦([\s\S]*?)⟧/g,"<mark>$1</mark>");}

function render(){
  applyFilter();
  const i=order[pos], r=rows[i];
  const revd=(r.reviewed||"").toUpperCase()==="Y";
  const yn=(field,val,isNo)=>`<span class="opt ${isNo?'no':''} ${(r[field]||'').toUpperCase()===val?'sel':''}" data-f="${field}" data-v="${val}">${val}</span>`;
  const failOpts=FAILS.map(([name,k])=>`<span class="opt ${r.failure===name?'sel':''}" data-fail="${name}">${name}<small>${k}</small></span>`).join("");
  $("#card").innerHTML=`
    <div class="meta"><span class="tag b">${esc(r.benchmark)}</span><span class="tag t">${esc(r.type)}</span>
      <span>#${esc(r.id)} · ${esc(r.doc_id)}</span></div>
    <div class="ov"><span class="o">${esc(r.original)}</span><span class="arw">&rarr;</span><span class="s">${esc(r.surrogate)}</span></div>
    <div class="ctx">${ctxHtml(r.context)}</div>
    <div class="row">
      <div class="grp"><span>valid</span><div class="opts">${yn("valid","Y")}${yn("valid","N",1)}</div></div>
      <div class="grp"><span>type&nbsp;consistent</span><div class="opts">${yn("type_consistent","Y")}${yn("type_consistent","N",1)}</div></div>
    </div>
    <div class="grp" style="margin-bottom:14px"><span>failure mode</span><div class="opts">${failOpts}</div></div>
    <div class="grp"><span>notes</span><input class="notes" type="text" id="notes" value="${esc(r.notes).replace(/"/g,'&quot;')}"></div>
    <div class="nav">
      <button class="btn" id="prev">&larr; Prev</button>
      <span class="rev ${revd?'on':''}" id="revlbl">${revd?'✓ reviewed':'not reviewed'}</span>
      <button class="btn k" id="next">Next &rarr;</button>
    </div>`;
  $("#prog").style.width=(100*rows.filter(r=>(r.reviewed||"").toUpperCase()==="Y").length/rows.length)+"%";
  $("#count").textContent=`${rows.filter(r=>(r.reviewed||'').toUpperCase()==='Y').length}/${rows.length} reviewed · showing ${pos+1}/${order.length}`;
  $("#card").querySelectorAll(".opt").forEach(el=>el.onclick=()=>{
    if(el.dataset.fail!==undefined){r.failure=el.dataset.fail;}
    else{r[el.dataset.f]=el.dataset.v;}
    r.reviewed="Y"; save(); render();});
  $("#notes").oninput=e=>{r.notes=e.target.value; r.reviewed="Y"; save();};
  $("#prev").onclick=()=>{pos=(pos-1+order.length)%order.length;render();};
  $("#next").onclick=()=>{pos=(pos+1)%order.length;render();};
  $("#revlbl").onclick=()=>{r.reviewed=revd?"":"Y";save();render();};
}
document.addEventListener("keydown",e=>{
  if(e.target.tagName==="INPUT"||e.target.tagName==="SELECT") return;
  const i=order[pos], r=rows[i]; let hit=true;
  if(e.key==="ArrowLeft") pos=(pos-1+order.length)%order.length;
  else if(e.key==="ArrowRight"||e.key==="Enter") pos=(pos+1)%order.length;
  else if(e.key.toLowerCase()==="v"){r.valid=(r.valid||"").toUpperCase()==="Y"?"N":"Y";r.reviewed="Y";}
  else if(e.key.toLowerCase()==="t"){r.type_consistent=(r.type_consistent||"").toUpperCase()==="Y"?"N":"Y";r.reviewed="Y";}
  else {const m=FAILS.find(f=>f[1]===e.key.toLowerCase()); if(m){r.failure=m[0];r.reviewed="Y";} else hit=false;}
  if(hit){e.preventDefault();save();render();}
});
function toCSV(){
  const q=s=>{s=(s==null?"":""+s); return /[",\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s;};
  return COLS.join(",")+"\n"+rows.map(r=>COLS.map(c=>q(r[c])).join(",")).join("\n")+"\n";
}
$("#exp").onclick=()=>{
  const b=new Blob([toCSV()],{type:"text/csv"}), u=URL.createObjectURL(b);
  const a=document.createElement("a");a.href=u;a.download="surrogate_quality_reviewed.csv";a.click();URL.revokeObjectURL(u);
};
$("#imp").onclick=()=>$("#file").click();
$("#file").onchange=e=>{const f=e.target.files[0];if(!f)return;const rd=new FileReader();
  rd.onload=()=>{try{rows=parseCSV(rd.result);save();pos=0;render();}catch(err){alert("Parse error: "+err);}};rd.readAsText(f);};
function parseCSV(t){
  const out=[],lines=[];let cur="",q=false,row=[];
  for(let i=0;i<t.length;i++){const c=t[i];
    if(q){if(c==='"'){if(t[i+1]==='"'){cur+='"';i++;}else q=false;}else cur+=c;}
    else{if(c==='"')q=true;else if(c===','){row.push(cur);cur="";}
      else if(c==='\n'||c==='\r'){if(c==='\r'&&t[i+1]==='\n')i++;row.push(cur);lines.push(row);row=[];cur="";}
      else cur+=c;}}
  if(cur.length||row.length){row.push(cur);lines.push(row);}
  const hdr=lines.shift();
  return lines.filter(l=>l.length>1).map(l=>{const o={};hdr.forEach((h,j)=>o[h]=l[j]||"");return o;});
}
$("#filter").onchange=()=>{pos=0;render();};
render();
</script></body></html>
"""


def main():
    rows = list(csv.DictReader(open(SRC)))
    data = [{c: r.get(c, "") for c in COLS} for r in rows]
    html = (HTML.replace("__COLS__", json.dumps(COLS))
                .replace("__DATA__", json.dumps(data, ensure_ascii=False)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write(html)
    print(f"wrote {OUT} ({len(rows)} spans, {len(html)} bytes)")


if __name__ == "__main__":
    main()

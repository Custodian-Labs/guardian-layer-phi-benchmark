#!/usr/bin/env python3
"""Decision-utility study: do *typed downstream decisions* survive de-identification?

The paper asks whether PHI stays *detectable* after surrogate substitution. This
asks a complementary downstream question: if a de-identified document is handed to
a decision model (classify / route / score), does the model decide the same thing
it would on the original?

Every question is typed: a fixed option set, answered with a probability per
option. Arms: original, faker (C2 surrogate), redacted (C1
`*****`), and custodian (Guardian Layer transform, if data/transformed/<bench>.jsonl
exists). Per question we record the full option distribution; scripts/analyze_decisions
-- here `--analyze` -- reports gold accuracy (where the dataset carries a label),
argmax agreement with the original, and mean total-variation distance.

Backend: an open LLM scored by option log-likelihood (no generation, no parsing).

Usage
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/decision_eval.py --backend local --model qwen3_5_4b
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/decision_eval.py --backend local --model qwen3_5_4b --limit 5   # smoke
  PYTHONPATH=. python scripts/decision_eval.py --analyze
  # clinical benchmarks (ASQ-PHI, MEDDOCAN); written to <backend>__medical.jsonl
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/decision_eval.py --model qwen3_5_9b --benchmarks asq_phi meddocan
Resumable: (bench, arm, doc_id) rows already in the output file are skipped.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)
DD = os.path.join(ROOT, "data", "domains")
OUT = os.path.join(ROOT, "results", "decisions")
BENCHES = ["finance_gretel", "legal_tab", "enterprise_gretel"]
# Clinical benchmarks from the paper: originals are the released 250-doc subsets,
# faker/redacted are the C1/C2 corpora (data/{faker,redacted}/), custodian is data/transformed/.
MED_BENCHES = ["asq_phi", "meddocan"]
MED_ORIG = {"asq_phi": "asq_phi_250.jsonl", "meddocan": "meddocan_250.jsonl"}
# ICD-10 diagnosis codes for the same SPACCC case reports, from CodiEsp (CC BY 4.0;
# bigbio/codiesp, codiesp_D_source); all 250 MEDDOCAN docs match by document_id.
# data/external/codiesp/meddocan_icd10.json maps doc_id -> {codes, chapters}.
ICD_CHAPTERS = {
    "neoplasms": "a neoplasm (benign or malignant tumour)",
    "genitourinary": "a disease of the genitourinary system (kidney, urinary tract or genital organs)",
    "digestive": "a disease of the digestive system",
    "circulatory": "a disease of the circulatory system (heart or blood vessels)",
    "infectious": "an infectious or parasitic disease",
    "musculoskeletal": "a disease of the musculoskeletal system or connective tissue",
    "endocrine_metabolic": "an endocrine, nutritional or metabolic disease",
    "skin": "a disease of the skin or subcutaneous tissue",
}
ARMS = ["original", "faker", "redacted", "custodian"]
MAX_CHARS = 6000


def _labels(bench: str, key: str) -> list[str]:
    rows = [json.loads(l) for l in open(os.path.join(DD, "original", f"{bench}.jsonl"))]
    return sorted({r["meta"][key] for r in rows})


def _med_meta(bench: str) -> dict:
    """Gold labels derived from the original document's gold PHI spans (MEDDOCAN
    annotates the patient's sex and age as PHI). First SEXO / EDAD span wins."""
    import re
    male = {"h", "varón", "varon", "hombre", "masculino", "niño", "nino"}
    female = {"m", "mujer", "femenino", "femenina", "niña", "nina", "f"}
    out = {}
    for l in open(os.path.join(ROOT, "web", "data", "downloads", MED_ORIG[bench])):
        r = json.loads(l); m = {}
        for g in sorted(r["gold_spans"], key=lambda g: g["start"]):
            v = r["text"][g["start"]:g["end"]].strip().lower()
            if "SEXO" in g["label"] and "sex" not in m:
                m["sex"] = "male" if v in male else "female" if v in female else None
            if "EDAD" in g["label"] and "age_group" not in m:
                n = re.search(r"\d+", v)
                if n:
                    yrs = int(n.group()) if "año" in v or "ano" in v else 0  # months/days -> infant
                    m["age_group"] = "child (under 18)" if yrs < 18 else "adult (18-64)" if yrs < 65 else "older adult (65+)"
        out[r["doc_id"]] = m
    if bench == "meddocan":
        icd = json.load(open(os.path.join(ROOT, "data", "external", "codiesp", "meddocan_icd10.json")))
        for did, m in out.items():
            if did in icd:
                for ch in ICD_CHAPTERS:
                    m[f"icd_{ch}"] = "yes" if ch in icd[did]["chapters"] else "no"
    return out


YES_NO = ["yes", "no"]
LEVELS = ["low", "medium", "high"]


def tasks(bench: str) -> list[dict]:
    """Typed questions per benchmark. `gold` names a meta field when a label exists."""
    if bench == "finance_gretel":
        return [
            {"id": "doc_type", "q": "What type of financial document is this?",
             "options": _labels(bench, "document_type"), "gold": "document_type"},
            {"id": "money_movement", "q": "Does this document instruct, record or confirm a transfer or payment of money?",
             "options": YES_NO},
            {"id": "compliance_risk", "q": "How much regulatory or compliance review would this document need before being acted on?",
             "options": LEVELS},
        ]
    if bench == "enterprise_gretel":
        return [
            {"id": "industry", "q": "Which industry domain does this document belong to?",
             "options": _labels(bench, "domain"), "gold": "domain"},
            {"id": "has_personal_data", "q": "Does this document contain personal data about an identifiable individual (e.g. a customer, patient or employee)?",
             "options": YES_NO},
            {"id": "sensitivity", "q": "How sensitive is this document for the organisation?",
             "options": LEVELS},
        ]
    if bench == "legal_tab":
        return [
            {"id": "section", "q": "Which part of a European Court of Human Rights judgment is this passage from?",
             "options": ["procedure", "facts of the case", "relevant domestic or international law",
                         "the law (the Court's assessment)", "operative provisions / judgment"]},
            {"id": "alleged_interference", "q": "Does this passage describe an alleged interference with the applicant's rights by state authorities (e.g. detention, ill-treatment, censorship, search)?",
             "options": YES_NO},
            {"id": "severity", "q": "How severe is the interference with the applicant's rights described in this passage?",
             "options": ["none", "low", "medium", "high"]},
        ]
    specialties = ["cardiology", "oncology", "neurology", "endocrinology", "infectious disease",
                   "psychiatry", "surgery or orthopaedics", "obstetrics or gynaecology", "paediatrics",
                   "urology or nephrology", "gastroenterology", "other"]
    if bench == "asq_phi":  # short clinical questions that mention a patient
        return [
            {"id": "specialty", "q": "Which clinical specialty is this query mainly about?", "options": specialties},
            {"id": "intent", "q": "What is the query mainly asking for?",
             "options": ["treatment", "diagnosis", "prognosis or outcomes", "medication or dosing",
                         "administrative or record lookup", "other"]},
            {"id": "age_group", "q": "Which age group does the patient in this query belong to?",
             "options": ["child (under 18)", "adult (18-64)", "older adult (65+)", "not stated"]},
            {"id": "urgency", "q": "How urgent is the clinical situation described?", "options": LEVELS},
        ]
    if bench == "meddocan":  # Spanish clinical case reports; sex and age carry gold labels
        return [
            {"id": "sex", "q": "What is the sex of the patient?", "options": ["male", "female"], "gold": "sex"},
            {"id": "age_group", "q": "Which age group does the patient belong to?",
             "options": ["child (under 18)", "adult (18-64)", "older adult (65+)"], "gold": "age_group"},
            {"id": "specialty", "q": "Which clinical specialty is this case report mainly about?", "options": specialties},
            {"id": "surgery", "q": "Did the patient undergo a surgical procedure?", "options": YES_NO},
            {"id": "outcome", "q": "What was the patient's outcome?",
             "options": ["recovered or improved", "stable or chronic", "worsened or died", "not stated"]},
        ] + [{"id": f"icd_{ch}", "q": f"Is the patient diagnosed with {desc}?", "options": YES_NO, "gold": f"icd_{ch}"}
             for ch, desc in ICD_CHAPTERS.items()]
    raise KeyError(bench)


# --------------------------------------------------------------------------- backends

SYSTEM = ("You are a decision engine. Read the document and answer the typed question "
          "with exactly one of the listed options, verbatim, and nothing else.")


def _user(text: str, q: str, options: list[str]) -> str:
    opts = "\n".join(f"- {o}" for o in options)
    return f"Document:\n{text[:MAX_CHARS]}\n\nQuestion: {q}\nOptions:\n{opts}\n\nAnswer:"


class LocalBackend:
    """P(option | prompt) from summed option-token log-likelihood, softmaxed over
    the option set. Left-padded batches + `logits_to_keep` so only the option
    positions are projected onto the vocabulary."""

    def __init__(self, model_key: str, chunk: int = 16):
        import torch
        from systems import llm_local_hf as L
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg = getattr(L, model_key.upper())
        L._patch_dynamic_cache_compat()
        self.name = f"local:{cfg.name}"
        self.tok = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=cfg.trust_remote_code)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        # Shared GPUs: same per-device cap as LocalHFLLM (CUSTODIAN_AUTO_MAX_MEM=1).
        max_memory = L._auto_max_memory() if os.environ.get("CUSTODIAN_AUTO_MAX_MEM") == "1" else None
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id, dtype=torch.bfloat16, device_map=cfg.device_map, max_memory=max_memory,
            trust_remote_code=cfg.trust_remote_code).eval()
        self.kw = cfg.chat_template_kwargs or {}
        self.chunk = chunk
        self.torch = torch

    def choose(self, text: str, q: str, options: list[str]) -> list[float]:
        """Shared GPUs: on OOM halve the option chunk and retry (chunking does not
        change the scores -- verified to <1e-4 against chunk=1)."""
        while True:
            try:
                return self._choose(text, q, options)
            except self.torch.OutOfMemoryError:
                if self.chunk == 1:
                    raise
                self.torch.cuda.empty_cache()
                self.chunk = max(1, self.chunk // 2)
                print(f"  [oom] option chunk -> {self.chunk}", flush=True)

    def _choose(self, text: str, q: str, options: list[str]) -> list[float]:
        torch = self.torch
        prompt = self.tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": _user(text, q, options)}],
            tokenize=False, add_generation_prompt=True, **self.kw)
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        scores = []
        for i in range(0, len(options), self.chunk):
            opts = options[i:i + self.chunk]
            o_ids = [self.tok(o, add_special_tokens=False)["input_ids"] for o in opts]
            seqs = [p_ids + o for o in o_ids]
            L = max(map(len, seqs)); K = max(map(len, o_ids))
            pad = self.tok.pad_token_id
            ids = torch.tensor([[pad] * (L - len(s)) + s for s in seqs], device=self.model.device)
            mask = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=self.model.device)
            pos = (mask.cumsum(-1) - 1).clamp(min=0)
            with torch.no_grad():
                logits = self.model(input_ids=ids, attention_mask=mask, position_ids=pos,
                                    logits_to_keep=K + 1).logits.float()
            logp = torch.log_softmax(logits, -1)  # [B, K+1, V]; position j predicts token at L-(K+1)+j+1
            for b, o in enumerate(o_ids):
                n = len(o)
                # option tokens occupy the last n positions; their predictors are the n before them
                preds = logp[b, K - n:K, :]
                scores.append(sum(preds[j, t].item() for j, t in enumerate(o)))
        m = max(scores)
        e = [math.exp(s - m) for s in scores]
        z = sum(e)
        return [x / z for x in e]


# --------------------------------------------------------------------------- run

def arm_path(bench: str, arm: str) -> str:
    if bench in MED_BENCHES:
        if arm == "original":
            return os.path.join(ROOT, "web", "data", "downloads", MED_ORIG[bench])
        return os.path.join(ROOT, "data", "transformed" if arm == "custodian" else arm, f"{bench}.jsonl")
    if arm == "custodian":
        return os.path.join(ROOT, "data", "transformed", f"{bench}.jsonl")
    return os.path.join(DD, arm, f"{bench}.jsonl")


def run(args):
    be = LocalBackend(args.model)
    os.makedirs(OUT, exist_ok=True)
    med = any(b in MED_BENCHES for b in args.benchmarks)
    if med and not all(b in MED_BENCHES for b in args.benchmarks):
        raise SystemExit("run clinical and non-medical benchmarks separately")
    out = os.path.join(OUT, f"{be.name.replace(':', '__')}{'__medical' if med else ''}.jsonl")
    done = set()
    if os.path.exists(out):
        for l in open(out):
            r = json.loads(l); done.add((r["bench"], r["arm"], r["doc_id"]))
    meta = {b: _med_meta(b) if b in MED_BENCHES else
            {json.loads(l)["doc_id"]: json.loads(l)["meta"] for l in open(os.path.join(DD, "original", f"{b}.jsonl"))}
            for b in args.benchmarks}
    with open(out, "a") as f:
        for bench in args.benchmarks:
            T = tasks(bench)
            for arm in args.arms:
                p = arm_path(bench, arm)
                if not os.path.exists(p):
                    print(f"[skip] {arm}/{bench}: {p} missing", flush=True); continue
                docs = [json.loads(l) for l in open(p)][: args.limit]
                todo = [d for d in docs if (bench, arm, d["doc_id"]) not in done]
                print(f"[run] {be.name} {arm}/{bench}: {len(todo)}/{len(docs)}", flush=True)
                t0 = time.time()
                for i, d in enumerate(todo):
                    ans = {t["id"]: be.choose(d["text"], t["q"], t["options"]) for t in T}
                    f.write(json.dumps({"bench": bench, "arm": arm, "doc_id": d["doc_id"],
                                        "probs": ans,
                                        "gold": {t["id"]: meta[bench][d["doc_id"]].get(t["gold"])
                                                 for t in T if t.get("gold")}}) + "\n")
                    f.flush()
                    if (i + 1) % 25 == 0:
                        print(f"  {arm}/{bench}: {i+1}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
    print("[done]", flush=True)


# --------------------------------------------------------------------------- analysis

def _tost(diffs: list[float], margin: float) -> tuple[float, float, float]:
    """Paired TOST on per-doc differences (in points). Returns (mean, se, p)."""
    from scipy.stats import norm
    n = len(diffs); m = sum(diffs) / n
    var = sum((x - m) ** 2 for x in diffs) / max(n - 1, 1)
    se = math.sqrt(var / n) or 1e-9
    p = max(1 - norm.cdf((m + margin) / se), norm.cdf((m - margin) / se))
    return m, se, p


def _boot(diffs: list[float], B: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed); n = len(diffs)
    ms = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return ms[int(0.025 * B)], ms[int(0.975 * B)]


def analyze(margin: float = 2.0):
    rows = []
    for fn in sorted(os.listdir(OUT)):
        if fn.endswith(".jsonl"):
            rows += [dict(json.loads(l), backend=fn[:-6]) for l in open(os.path.join(OUT, fn))]
    idx = {(r["backend"], r["bench"], r["arm"], r["doc_id"]): r for r in rows}
    lines = ["# Decision utility after de-identification", "",
             f"Paired over documents. Agreement = argmax equals the original's; TVD = mean total-variation "
             f"distance between option distributions (0 = identical). Accuracy Δ in points with doc-bootstrap "
             f"95% CI and paired TOST (±{margin:g} pt).", ""]
    summary = {}
    for backend in sorted({r["backend"] for r in rows}):
        lines += [f"## {backend}", "",
                  "| Domain | Question | Arm | N | Agreement (%) | TVD | Acc orig (%) | Acc arm (%) | Δ acc (pt) | 95% CI | TOST p |",
                  "|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for bench in BENCHES + MED_BENCHES:
            if not any(r["backend"] == backend and r["bench"] == bench for r in rows):
                continue
            T = {t["id"]: t for t in tasks(bench)}
            for tid, t in T.items():
                for arm in ARMS[1:]:
                    pairs = [(idx[(backend, bench, "original", d)], r) for (bk, b, a, d), r in idx.items()
                             if bk == backend and b == bench and a == arm and (backend, bench, "original", d) in idx]
                    if not pairs:
                        continue
                    agree, tvd, acc_o, acc_a = [], [], [], []
                    for o, r in pairs:
                        if t.get("gold") and o["gold"].get(tid) is None:
                            continue  # no gold label for this doc (e.g. MEDDOCAN doc without a SEXO span)
                        po, pa = o["probs"][tid], r["probs"][tid]
                        io, ia = po.index(max(po)), pa.index(max(pa))
                        agree.append(io == ia)
                        tvd.append(0.5 * sum(abs(x - y) for x, y in zip(po, pa)))
                        if t.get("gold"):
                            g = t["options"].index(o["gold"][tid])
                            acc_o.append(100.0 * (io == g)); acc_a.append(100.0 * (ia == g))
                    n = len(agree)
                    cell = {"n": n, "agreement": 100 * sum(agree) / n, "tvd": sum(tvd) / n}
                    acc_cols = "| | | | | |"
                    if acc_o:
                        diffs = [a - b for a, b in zip(acc_a, acc_o)]
                        m, se, p = _tost(diffs, margin); lo, hi = _boot(diffs)
                        cell.update(acc_orig=sum(acc_o) / n, acc_arm=sum(acc_a) / n, delta=m, ci=[lo, hi], tost_p=p)
                        acc_cols = (f"| {cell['acc_orig']:.1f} | {cell['acc_arm']:.1f} | {m:+.1f} | "
                                    f"[{lo:+.1f}, {hi:+.1f}] | {p:.1e} |")
                    summary[f"{backend}|{bench}|{tid}|{arm}"] = cell
                    lines.append(f"| {bench} | {tid} | {arm} | {n} | {cell['agreement']:.1f} | {cell['tvd']:.3f} {acc_cols}")
        lines.append("")
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["local"], default="local")
    ap.add_argument("--model", default="qwen3_5_4b", help="LocalLLMConfig name in systems/llm_local_hf.py")
    ap.add_argument("--benchmarks", nargs="+", default=BENCHES, choices=BENCHES + MED_BENCHES)
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    if args.analyze:
        analyze()
    else:
        run(args)


if __name__ == "__main__":
    main()

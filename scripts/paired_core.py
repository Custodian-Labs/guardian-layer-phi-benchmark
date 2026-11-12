#!/usr/bin/env python3
"""Paired masked-span outcomes: re-projection, matching criteria, cluster statistics.

One module for the camera-ready analyses, so every table uses the same
definitions:

Re-projection of a gold span onto the transformed document d'
  difflib    the stored span (character-level difflib alignment). Masked iff the
             stored gold `text` fields differ: exactly the paper's convention
             (5,279 unique spans, 57,112 outcomes).
  snap_sym   difflib span, each boundary moved outward to the nearest word boundary
             only where the original gold boundary is itself a word boundary on d.
             Masked iff the original slice differs from the snapped slice. This
             recovers replacements that difflib cut back to the unchanged prefix
             (`Colombia` -> `Colombian`, `123456789` -> `1234567890`), which the
             paper counted as unmasked. Gold spans whose stored text does not
             match the document (template placeholders in pii-masking-300k) are
             excluded and counted in SKIPPED.
  Neither is the transform's own replacement offsets (not stored; the transform
  API is needed to recover them).

Matching criteria (found = any predicted span satisfies it)
  overlap     >= 1 shared character, label-agnostic (paper's primary criterion)
  iou50       character IoU >= 0.5, label-agnostic
  exact       identical start and end, label-agnostic
  exact_type  identical start and end and the same coarse type (TYPES below);
              predicted labels outside the taxonomy never match

Statistics: document-level cluster bootstrap stratified by benchmark (ratio
estimator, as in analyze_v2.boot_stats), TOST at margin m <=> 90% CI inside
(-m, m), smallest passing margin, Bayesian-bootstrap P(|diff| < m).
diff = recall(original) - recall(transformed), in points; positive = loss.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from analyze_equivalence import BENCH, load  # noqa: E402

CRITERIA = ("overlap", "iou50", "exact", "exact_type")
SKIPPED: "defaultdict[tuple, int]" = defaultdict(int)
REPROJ = ("difflib", "snap_sym")

# ---------------------------------------------------------------- coarse types
TYPES = ("NAME", "DATE", "AGE", "SEX", "LOC", "ID", "CONTACT", "OTHER")
_GOLD = {
    "NAME": {"NAME", "NOMBRE_SUJETO_ASISTENCIA", "NOMBRE_PERSONAL_SANITARIO", "FAMILIARES_SUJETO_ASISTENCIA",
             "GIVENNAME1", "GIVENNAME2", "LASTNAME1", "LASTNAME2", "LASTNAME3", "OtherPER", "Artist", "Politician",
             "Athlete", "Scientist", "Cleric"},
    "DATE": {"FECHAS", "DATE", "TIME", "BOD"},
    "AGE": {"EDAD_SUJETO_ASISTENCIA"},
    "SEX": {"SEXO_SUJETO_ASISTENCIA", "SEX"},
    "LOC": {"TERRITORIO", "CALLE", "PAIS", "STATE", "BUILDING", "STREET", "CITY", "POSTCODE", "SECADDRESS", "COUNTRY",
            "GEOGRAPHIC_LOCATION", "HOSPITAL", "INSTITUCION", "CENTRO_SALUD", "HumanSettlement", "Facility",
            "GEOCOORD", "ORG"},
    "ID": {"PASSPORT", "SOCIALNUMBER", "IDCARD", "DRIVERLICENSE", "ID_SUJETO_ASISTENCIA",
           "ID_TITULACION_PERSONAL_SANITARIO", "ID_ASEGURAMIENTO", "ID_CONTACTO_ASISTENCIAL", "MEDICAL_RECORD_NUMBER",
           "HEALTH_PLAN_BENEFICIARY_NUMBER", "SOCIAL_SECURITY_NUMBER", "UNIQUE_IDENTIFIER", "ACCOUNT_NUMBER"},
    "CONTACT": {"USERNAME", "EMAIL", "CORREO_ELECTRONICO", "EMAIL_ADDRESS", "IP", "TEL", "NUMERO_TELEFONO",
                "PHONE_NUMBER", "NUMERO_FAX"},
}
_PRED = {
    "NAME": {"PERSON", "NAME", "PATIENT", "STAFF", "DOCTOR", "SURNAME", "LAST_NAME"},
    "DATE": {"DATE_TIME", "DATE", "TIME"},
    "AGE": {"AGE"},
    "SEX": {"SEX", "GENDER"},
    "LOC": {"LOCATION", "LOC", "HOSP", "PATORG", "ADDRESS", "COUNTRY", "ZIP", "CP", "ORGANIZATION"},
    "ID": {"ID", "US_DRIVER_LICENSE", "US_BANK_NUMBER", "US_PASSPORT", "US_SSN", "US_ITIN", "UK_NHS",
           "MEDICAL_LICENSE", "MRN", "NHC", "NASS", "SSN", "LICENSE", "EPISODE"},
    "CONTACT": {"PHONE", "PHONE_NUMBER", "EMAIL", "EMAIL_ADDRESS", "URL", "IP", "IP_ADDRESS", "USERNAME", "FAX"},
    "OTHER": {"OTHER", "OTHERPHI", "NRP"},
}
GOLD_TYPE = {lab: t for t, labs in _GOLD.items() for lab in labs}
PRED_TYPE = {lab: t for t, labs in _PRED.items() for lab in labs}


def gold_type(label: str) -> str:
    """Unlisted gold labels (TITLE, PASS, PROFESION, MultiCoNER works/groups, ...) are OTHER."""
    return GOLD_TYPE.get(label, "OTHER")


def pred_type(label) -> str | None:
    """None for labels outside the taxonomy; they never satisfy exact_type."""
    return PRED_TYPE.get(str(label).upper()) if label is not None else None


# ---------------------------------------------------------------- re-projection
def _inside_word(text: str, i: int) -> bool:
    """True if position i splits two alphanumeric characters."""
    return 0 < i < len(text) and text[i - 1].isalnum() and text[i].isalnum()


def snap_sym(o_text: str, o_start: int, o_end: int, t_text: str, t_start: int, t_end: int) -> tuple[int, int]:
    """Snap the re-projected span outward to word boundaries on d', side by side,
    only where the original gold boundary on d is a word boundary."""
    s, e = t_start, t_end
    if not _inside_word(o_text, o_start):
        while _inside_word(t_text, s):
            s -= 1
    if not _inside_word(o_text, o_end):
        while _inside_word(t_text, e):
            e += 1
    return s, e


# ---------------------------------------------------------------- matching
def matches(gs: int, ge: int, glabel: str, p: dict, criterion: str) -> bool:
    ps, pe = p["start"], p["end"]
    if criterion == "overlap":
        return not (pe <= gs or ps >= ge)
    if criterion == "iou50":
        inter = max(0, min(ge, pe) - max(gs, ps))
        union = max(ge, pe) - min(gs, ps)
        return union > 0 and inter / union >= 0.5
    if criterion == "exact":
        return ps == gs and pe == ge
    if criterion == "exact_type":
        return ps == gs and pe == ge and pred_type(p.get("label")) == gold_type(glabel)
    raise ValueError(criterion)


def found(gs: int, ge: int, glabel: str, preds: list[dict], criterion: str) -> bool:
    return any(matches(gs, ge, glabel, p, criterion) for p in preds)


# ---------------------------------------------------------------- paired rows
def paired_rows(reproj: str = "difflib", benches=None, cache: dict | None = None):
    """Yield dicts, one per (masked span, detector) present in both conditions:
    bench, doc, det, idx, gtype, orig_text, surr_text, o (criterion->bool), t (criterion->bool).
    Spans are masked under the given re-projection. Detectors missing on either
    side of a document are skipped (the paper's convention)."""
    assert reproj in REPROJ
    cache = {} if cache is None else cache
    for bench, (og, tg) in BENCH.items():
        if benches and bench not in benches:
            continue
        if bench not in cache:
            cache[bench] = (load(og), load(tg), _texts(tg))
        do, dt, tt = cache[bench]
        for did, o in do.items():
            t = dt.get(did)
            if not t or len(o["gold"]) != len(t["gold"]):
                continue
            o_text = _orig_text(og, did, cache)
            t_text = tt.get(did, "")
            dets = sorted(set(o["preds"]) & set(t["preds"]))
            for i, (go, gt) in enumerate(zip(o["gold"], t["gold"])):
                ts, te = gt["start"], gt["end"]
                if reproj == "difflib":
                    surr = gt["text"]
                    if go["text"] == surr:
                        continue
                else:
                    if o_text[go["start"]:go["end"]] != go["text"]:
                        SKIPPED[(bench, "gold text != document")] += 1
                        continue
                    ts, te = snap_sym(o_text, go["start"], go["end"], t_text, ts, te)
                    surr = t_text[ts:te]
                    if go["text"] == surr:
                        continue
                for det in dets:
                    yield {"bench": bench, "doc": did, "det": det, "idx": i, "gtype": gold_type(go["label"]),
                           "orig_text": go["text"], "surr_text": surr,
                           "o": {c: found(go["start"], go["end"], go["label"], o["preds"][det], c) for c in CRITERIA},
                           "t": {c: found(ts, te, gt["label"], t["preds"][det], c) for c in CRITERIA}}


def _texts(globs):
    import glob as G
    import json
    out = {}
    for g in globs:
        for fn in sorted(G.glob(os.path.join(os.path.dirname(__file__), os.pardir, "results", g))):
            with open(fn) as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        out.setdefault(r["doc_id"], r.get("text", ""))
    return out


def _orig_text(og, did, cache):
    key = ("__orig_text__", tuple(og))
    if key not in cache:
        cache[key] = _texts(og)
    return cache[key].get(did, "")


# ---------------------------------------------------------------- statistics
def per_doc_arrays(rows, criterion: str, keep=lambda r: True):
    """stratum (bench) -> (n[], found_o[], found_t[]) summed per document."""
    acc = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        if keep(r):
            a = acc[(r["bench"], r["doc"])]
            a[0] += 1; a[1] += r["o"][criterion]; a[2] += r["t"][criterion]
    strata = defaultdict(list)
    for (b, _), v in acc.items():
        strata[b].append(v)
    return {b: tuple(np.array(x) for x in zip(*v)) for b, v in strata.items()}


def cluster_stats(per_doc: dict, B: int = 5000, seed: int = 0, margins=(1, 2, 3)) -> dict:
    """Stratified document-level cluster bootstrap of the pooled ratio estimator
    diff = 100 * (sum fo - sum ft) / sum n, plus Bayesian-bootstrap P(|diff| < m)."""
    rng = np.random.default_rng(seed)
    tot_n = sum(v[0].sum() for v in per_doc.values())
    if tot_n == 0:
        return {"n": 0}
    rec_o = 100 * sum(v[1].sum() for v in per_doc.values()) / tot_n
    rec_t = 100 * sum(v[2].sum() for v in per_doc.values()) / tot_n
    cb_n = np.zeros(B); cb_d = np.zeros(B); bb_n = np.zeros(B); bb_d = np.zeros(B)
    for n, fo, ft in per_doc.values():
        k = len(n)
        idx = rng.integers(0, k, size=(B, k))
        cb_n += n[idx].sum(1); cb_d += (fo - ft)[idx].sum(1)
        w = rng.dirichlet(np.ones(k), size=B)
        bb_n += (w @ n) * k; bb_d += (w @ (fo - ft)) * k
    cb = 100 * cb_d / cb_n; bb = 100 * bb_d / bb_n
    lo95, hi95 = np.percentile(cb, [2.5, 97.5]); lo90, hi90 = np.percentile(cb, [5, 95])
    return {"n": int(tot_n), "docs": int(sum(len(v[0]) for v in per_doc.values())),
            "rec_o": float(rec_o), "rec_t": float(rec_t), "diff": float(rec_o - rec_t),
            "ci95": [float(lo95), float(hi95)], "ci90": [float(lo90), float(hi90)],
            "min_margin": float(max(abs(lo90), abs(hi90))),
            "tost": {m: bool(-m < lo90 and hi90 < m) for m in margins},
            "p_rope": {m: float(np.mean(np.abs(bb) < m)) for m in margins}}

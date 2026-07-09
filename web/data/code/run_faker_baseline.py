#!/usr/bin/env python3
"""C2 -- open-source surrogate baseline (generality / reproducibility).

Replaces the commercial generator with an open, reproducible one (Faker),
substituting the *same* masked spans Custodian changed, then re-detects with
Presidio. If masked-span recall on Faker surrogates tracks recall on Custodian
surrogates, the equivalence result is a property of *well-formed substitution*,
not of one vendor -- and the pipeline is reproducible without proprietary access.

Deterministic (seeded). CPU only. Usage:
    PYTHONPATH=. python scripts/run_faker_baseline.py [bench ...]
"""
from __future__ import annotations

import json
import os
import random
import sys

from faker import Faker

RESULTS = os.path.join(os.path.dirname(__file__), os.pardir, "results")

BENCH = {
    "ASQ-PHI": ("asq_250_baselines.jsonl", "t_asq_phi_250_baselines.jsonl", "en"),
    "MEDDOCAN": ("meddocan_250_baselines.jsonl", "t_meddocan_250_baselines.jsonl", "es"),
    "PII en": ("pii_250_baselines.jsonl", "t_pii_masking_300k_250_baselines.jsonl", "en"),
    "PII nl": ("pii_dutch_250_baselines.jsonl", "t_pii_dutch_250_baselines.jsonl", "nl"),
    "PII fr": ("pii_french_250_baselines.jsonl", "t_pii_french_250_baselines.jsonl", "fr"),
    "PII de": ("pii_german_250_baselines.jsonl", "t_pii_german_250_baselines.jsonl", "de"),
}
LOCALE = {"en": "en_US", "es": "es_ES", "nl": "nl_NL", "fr": "fr_FR", "de": "de_DE"}
PRESIDIO_KEY = "presidio"


def overlap(a_s, a_e, b_s, b_e):
    return not (b_e <= a_s or b_s >= a_e)


def load_docs(fn):
    docs = {}
    for line in open(os.path.join(RESULTS, fn)):
        line = line.strip()
        if line:
            r = json.loads(line)
            docs[r["doc_id"]] = r
    return docs


def category(label: str) -> str:
    L = label.upper()
    def has(*ks): return any(k in L for k in ks)
    if has("MAIL", "CORREO"):
        return "email"
    if has("IP"):
        return "ip"
    if has("USER"):
        return "username"
    if has("PHONE", "TEL"):
        return "phone"
    if has("EDAD", "AGE"):
        return "age"
    if has("SEXO", "SEX", "GENDER"):
        return "sex"
    if has("FECHA", "DATE", "BOD", "BIRTH", "TIME", "DOB"):
        return "date"
    if has("PASSPORT", "SOCIAL", "IDCARD", "DRIVER", "LICEN", "SSN", "NUMBER", "ID_", "CARD", "TAX"):
        return "idnum"
    if has("TERRITORIO", "CALLE", "PAIS", "LOC", "GEO", "CITY", "ADDR", "STREET", "COUNTRY", "STATE", "ZIP", "HOSP"):
        return "loc"
    if has("NOMBRE", "NAME", "GIVEN", "LAST", "FIRST", "PER", "TITLE", "PATIENT", "DOCTOR"):
        return "name"
    return "name"  # default: treat unknown PII as a name-like token


def fake_value(fk: Faker, cat: str) -> str:
    return {
        "email": fk.email,
        "ip": fk.ipv4,
        "username": fk.user_name,
        "phone": fk.phone_number,
        "age": lambda: str(random.randint(1, 98)),
        "sex": lambda: random.choice(["male", "female"]),
        "date": lambda: fk.date(pattern="%d/%m/%Y"),
        "idnum": lambda: fk.bothify("########"),
        "loc": fk.city,
        "name": fk.name,
    }[cat]()


def rebuild(text, gold_o, gold_t, fk):
    """Return (new_text, new_masked_spans) replacing masked spans with fakes."""
    order = sorted(range(len(gold_o)), key=lambda i: gold_o[i]["start"])
    out = []
    cursor = 0
    shift = 0
    new_masked = []
    for i in order:
        g, t = gold_o[i], gold_t[i]
        masked = g.get("text") != t.get("text")
        out.append(text[cursor:g["start"]])
        if masked:
            surro = fake_value(fk, category(g["label"]))
            ns = g["start"] + shift
            out.append(surro)
            new_masked.append({"start": ns, "end": ns + len(surro), "label": g["label"]})
            shift += len(surro) - (g["end"] - g["start"])
        else:
            out.append(text[g["start"]:g["end"]])
        cursor = g["end"]
    out.append(text[cursor:])
    return "".join(out), new_masked


def recall(spans, preds):
    if not spans:
        return 0, 0
    hit = sum(1 for g in spans if any(overlap(g["start"], g["end"], p.start, p.end) for p in preds))
    return hit, len(spans)


def main():
    benches = [b for b in sys.argv[1:] if b in BENCH] or list(BENCH)
    print(f"{'benchmark':12s} {'masked':>7s} {'orig':>7s} {'custodian':>9s} {'faker':>7s}   (Presidio recall on masked spans)")
    for name in benches:
        ofn, tfn, lang = BENCH[name]
        odocs, tdocs = load_docs(ofn), load_docs(tfn)
        fk = Faker(LOCALE.get(lang, "en_US"))
        fk.seed_instance(20260709)
        random.seed(20260709)
        from systems.presidio import Presidio
        det = Presidio(language=lang)

        tot = ho = hc = hf = 0
        for did, od in odocs.items():
            td = tdocs.get(did)
            if not td:
                continue
            go, gt = od.get("gold_spans", []), td.get("gold_spans", [])
            if len(go) != len(gt) or not go:
                continue
            masked_o = [g for g, t in zip(go, gt) if g.get("text") != t.get("text")]
            masked_t = [t for g, t in zip(go, gt) if g.get("text") != t.get("text")]
            if not masked_o:
                continue

            def preds_of(doc, key):
                p = doc.get("predictions", {}).get(key)
                if p is None:
                    return det.predict(doc["text"]).spans
                return [type("P", (), {"start": s["start"], "end": s["end"]})() for s in p["spans"]]

            o_preds = preds_of(od, PRESIDIO_KEY)
            c_preds = preds_of(td, PRESIDIO_KEY)
            f_text, f_masked = rebuild(od["text"], go, gt, fk)
            f_preds = det.predict(f_text).spans

            for spans, preds, acc in ((masked_o, o_preds, "o"), (masked_t, c_preds, "c"), (f_masked, f_preds, "f")):
                hit, _ = recall(spans, preds)
                if acc == "o":
                    ho += hit
                elif acc == "c":
                    hc += hit
                else:
                    hf += hit
            tot += len(masked_o)

        def pct(x):
            return f"{100*x/tot:6.1f}%" if tot else "   n/a"
        print(f"{name:12s} {tot:7d} {pct(ho)} {pct(hc):>9s} {pct(hf)}")
        det.close()


if __name__ == "__main__":
    main()

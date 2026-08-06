#!/usr/bin/env python3
"""Build the C1 (redaction) and C2 (open-surrogate / Faker) corpora for ALL
benchmarks, so the full 11-detector suite can score them (not just Presidio).

For each benchmark we pair the original and Custodian-transformed subsets by
doc_id + gold-span index; a span is *masked* when its text changed. Then:

  data/redacted/<bench>.jsonl : original text, each masked span replaced by
                                '*' x len (same length -> gold positions kept).
  data/faker/<bench>.jsonl    : original text, each masked span replaced by a
                                seeded Faker same-type value (gold remapped).

Both are written as {doc_id, text, gold_spans, meta} exactly like
data/transformed/, so scripts/run_benchmark.py can score them with any system.
Deterministic. CPU only.
"""
from __future__ import annotations
import json, os, random
from faker import Faker

DL = os.path.join(os.path.dirname(__file__), os.pardir, "web", "data", "downloads")
ROOT = os.path.join(os.path.dirname(__file__), os.pardir)
PAIRS = {
    "asq_phi": ("asq_phi_250.jsonl", "asq_phi_250_transformed.jsonl", "en"),
    "meddocan": ("meddocan_250.jsonl", "meddocan_250_transformed.jsonl", "es"),
    "multiconer_v2": ("multiconer_v2_250.jsonl", "multiconer_v2_250_transformed.jsonl", "en"),
    "pii_masking_300k": ("pii_masking_300k_250.jsonl", "pii_masking_300k_250_transformed.jsonl", "en"),
    "pii_dutch": ("pii_masking_300k_dutch_250.jsonl", "pii_masking_300k_dutch_250_transformed.jsonl", "nl"),
    "pii_french": ("pii_masking_300k_french_250.jsonl", "pii_masking_300k_french_250_transformed.jsonl", "fr"),
    "pii_german": ("pii_masking_300k_german_250.jsonl", "pii_masking_300k_german_250_transformed.jsonl", "de"),
}
LOCALE = {"en": "en_US", "es": "es_ES", "nl": "nl_NL", "fr": "fr_FR", "de": "de_DE"}


def load(fn):
    d = {}
    p = os.path.join(DL, fn)
    for line in open(p):
        line = line.strip()
        if line:
            r = json.loads(line)
            d[r["doc_id"]] = r
    return d


def category(label):
    L = label.upper()
    def has(*k): return any(x in L for x in k)
    if has("MAIL", "CORREO"): return "email"
    if has("IP"): return "ip"
    if has("USER"): return "username"
    if has("PHONE", "TEL"): return "phone"
    if has("EDAD", "AGE", "BOD"): return "age"
    if has("SEXO", "SEX", "GENDER"): return "sex"
    if has("FECHA", "DATE", "TIME", "DOB"): return "date"
    if has("PASSPORT", "SOCIAL", "IDCARD", "DRIVER", "LICEN", "SSN", "NUMBER", "ID_", "CARD", "TAX"): return "idnum"
    if has("TERRITORIO", "CALLE", "PAIS", "LOC", "GEO", "CITY", "ADDR", "STREET", "COUNTRY", "STATE", "ZIP", "HOSP"): return "loc"
    return "name"


def fake_value(fk, cat):
    return {
        "email": fk.email, "ip": fk.ipv4, "username": fk.user_name, "phone": fk.phone_number,
        "age": lambda: str(random.randint(1, 98)), "sex": lambda: random.choice(["male", "female"]),
        "date": lambda: fk.date(pattern="%d/%m/%Y"), "idnum": lambda: fk.bothify("########"),
        "loc": fk.city, "name": fk.name,
    }[cat]()


def masked_flags(og, tg):
    """Return list of bools: is gold span i masked (text changed)?"""
    if len(og) != len(tg):
        return None
    return [og[i].get("text") != tg[i].get("text") for i in range(len(og))]


def build_redact(orig, flags):
    chars = list(orig["text"])
    for g, m in zip(orig["gold_spans"], flags):
        if m:
            for i in range(g["start"], min(g["end"], len(chars))):
                chars[i] = "*"
    gold=[{**g, "masked": bool(m)} for g, m in zip(orig["gold_spans"], flags)]
    return {"doc_id": orig["doc_id"], "text": "".join(chars),
            "gold_spans": gold,  # same positions (same length)
            "meta": {"mode": "redact", "n_masked": sum(flags)}}


def build_faker(orig, flags, fk):
    gs = sorted(range(len(orig["gold_spans"])), key=lambda i: orig["gold_spans"][i]["start"])
    text = orig["text"]; out = []; cursor = 0; shift = 0; new_gold = []
    for i in gs:
        g = orig["gold_spans"][i]; m = flags[i]
        out.append(text[cursor:g["start"]])
        if m:
            surro = fake_value(fk, category(g.get("label", "")))
            ns = g["start"] + shift
            out.append(surro)
            new_gold.append({"start": ns, "end": ns + len(surro), "label": g.get("label", ""), "text": surro, "masked": True})
            shift += len(surro) - (g["end"] - g["start"])
        else:
            seg = text[g["start"]:g["end"]]; out.append(seg)
            ns = g["start"] + shift
            new_gold.append({"start": ns, "end": ns + len(seg), "label": g.get("label", ""), "text": seg, "masked": False})
        cursor = g["end"]
    out.append(text[cursor:])
    new_gold.sort(key=lambda x: x["start"])
    return {"doc_id": orig["doc_id"], "text": "".join(out), "gold_spans": new_gold,
            "meta": {"mode": "faker", "n_masked": sum(flags)}}


def main():
    for sub in ("redacted", "faker"):
        os.makedirs(os.path.join(ROOT, "data", sub), exist_ok=True)
    for bench, (ofn, tfn, lang) in PAIRS.items():
        O, T = load(ofn), load(tfn)
        fk = Faker(LOCALE.get(lang, "en_US")); fk.seed_instance(20260806); random.seed(20260806)
        red, fak = [], []
        nred = nfak = 0
        for did, od in O.items():
            td = T.get(did)
            if not td:
                continue
            flags = masked_flags(od.get("gold_spans", []), td.get("gold_spans", []))
            if flags is None:
                continue
            red.append(build_redact(od, flags)); nred += sum(flags)
            fak.append(build_faker(od, flags, fk)); nfak += sum(flags)
        with open(os.path.join(ROOT, "data", "redacted", f"{bench}.jsonl"), "w") as f:
            for r in red: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(os.path.join(ROOT, "data", "faker", f"{bench}.jsonl"), "w") as f:
            for r in fak: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{bench:18s} docs={len(red):4d}  masked_spans~{nred}")
    print("\nwrote data/redacted/*.jsonl and data/faker/*.jsonl (format = data/transformed/)")


if __name__ == "__main__":
    main()

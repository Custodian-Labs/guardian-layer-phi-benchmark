# Benchmark datasets — what they are, why they matter for our claim

Our claim is **two-headed**:
1. Guardian Layer removes PHI effectively (recall high, leakage low).
2. After Guardian Layer transforms the text, downstream tasks still work
   nearly as well as on the raw text (utility preserved).

A benchmark is useful for us if it lets us measure **at least one** of those.
A great benchmark lets us measure **both**. Below, each dataset is described
plus the specific experiment we should run on it.

---

## 1. MEDDOCAN (Spanish synthetic) — ✅ in-hand today

| Field | Value |
| --- | --- |
| Language | Spanish |
| Size | 1,000 synthetic case reports (250 train / 250 dev / 250 test / 250 background) |
| PHI density | ~21 PHI spans / document |
| Entity types | 29 fine-grained (NOMBRE_SUJETO_ASISTENCIA, EDAD_SUJETO_ASISTENCIA, …) |
| License | Open (Zenodo, CC) |
| DUA needed | No |
| Status | Downloaded and running |

**Why we include it.** MEDDOCAN is the largest *public* clinical de-identification
corpus that anyone can use without a Data Use Agreement. Every paper in the
Spanish clinical NLP space cites it. Reviewer-credible.

**What we can measure for our claim**

- **PHI detection** (head 1): straightforward — span P/R/F1 vs gold.
  *Already measured*: Presidio F1=0.40, OBI-RoBERTa F1=0.07 on Spanish.
- **Utility preservation** (head 2): MEDDOCAN does not ship a downstream
  classifier label. To show utility, we plan a **proxy task** — *masked NER
  recovery*: take the Guardian-Layer-transformed text and ask a separate
  Spanish clinical NER model (e.g. `PlanTL-GOB-ES/roberta-base-bne-clinical`)
  to extract diseases, procedures, medications. Compare entity counts and
  type distribution to those extracted from the raw text. If counts match
  ±5%, utility is preserved.

---

## 2. ASQ-PHI (English synthetic queries) — 🟡 needs manual download (browser only)

| Field | Value |
| --- | --- |
| Language | English |
| Size | 1,051 clinician-style search queries (832 PHI-positive, 219 hard negatives) |
| PHI density | ~2,973 PHI elements across the corpus (~3 per positive query) |
| Entity types | 13 HIPAA Safe Harbor types representable as alphanumeric strings |
| License | MIT (Mendeley Data DOI 10.17632/csz5dzp7nx.1) |
| DUA needed | No, but Mendeley's download flow requires a browser session |
| Status | Awaiting your browser download |

**Why this one is the highlight of our story.** ASQ-PHI was *built for the
exact problem Guardian Layer solves*: clinicians asking external LLMs / search
engines questions that contain PHI. The benchmark provides:

- **Hard negatives** — 219 queries with NO PHI, used to measure over-redaction.
  If Guardian Layer flags entities in a hard negative, it's a false positive
  and damages query utility.
- **Per-PHI-type leakage breakdowns** — you can argue *which* identifier
  classes Guardian Layer handles best.

**What we can measure for our claim**

- **PHI detection** (head 1): paper reports baseline numbers for a "commercial
  PHI detection service" — Guardian Layer can be benchmarked directly against
  that printed baseline.
- **Utility preservation** (head 2): the dataset's "search utility" angle is
  perfect. Transform each query with Guardian Layer, then check whether the
  transformed query is still semantically equivalent to the original
  (cosine similarity in a sentence-embedding space ≥ 0.9, for example), and
  whether retrieval results from a reference corpus stay in the top-k.

**Action needed**: open https://data.mendeley.com/datasets/csz5dzp7nx/1
in your browser, click `synthetic_clinical_queries.txt`, download it to
`data/asq_phi/raw/synthetic_clinical_queries.txt`, then run
`python scripts/import_asq_phi.py`.

---

## 3. n2c2 2006 De-ID + Smoking (English EHR) — 🔴 DUA pending

| Field | Value |
| --- | --- |
| Language | English |
| Size | 889 discharge summaries (de-id task) / 502 smoking-labeled (downstream task) |
| Smoking labels | 5 classes (never / past / current / smoker temporality unknown / unknown) |
| Train/test split | Provided |
| License | Restricted (n2c2 DUA) |
| DUA needed | **Yes** — apply at https://portal.dbmi.hms.harvard.edu/projects/n2c2-2006/ |
| Status | Apply ASAP, 1–4 week turnaround |

**Why this is the gold-standard benchmark for our claim's *second* head.**
This is the only benchmark in our list that comes with a **canonical
downstream classification task** (smoking-status detection from clinical
notes) that is independent of PHI detection. The experiment is:

```
raw notes        → smoking classifier → accuracy = X
Guardian-Layer transformed notes → smoking classifier → accuracy = Y
                  Claim: Y ≈ X
```

The smoking task is hard enough that a published baseline exists (~92% F1)
but not so trivial that any transformation looks fine. **This is the
experiment your paper introduction should lead with.**

---

## 4. n2c2 2014 Heart Disease (English EHR, longitudinal) — 🔴 DUA pending

| Field | Value |
| --- | --- |
| Language | English |
| Size | 1,304 longitudinal notes / 296 diabetic patients |
| Downstream task | Risk-factor extraction (CAD, hypertension, hyperlipidemia, smoking, obesity, family history, diabetes, plus medication indicators) over time |
| License | Restricted (n2c2 DUA) |
| DUA needed | Yes — same DUA as #3 |
| Status | Apply alongside #3 |

**Why we need it as a second downstream test.** Risk-factor extraction is a
structured information extraction task that touches many entity types that
Guardian Layer might transform (medications, lab values, named conditions
with patient initials, dates). If transformation distorts disease names or
date ordering, accuracy drops — a strong test of utility preservation under
clinical-content-bearing transformations.

---

## 5. MIMIC-IV-Note (English real EHR) — 🔴 DUA pending

| Field | Value |
| --- | --- |
| Language | English |
| Size | 331,794 discharge summaries + 2,321,355 radiology reports |
| Source | Beth Israel Deaconess Medical Center |
| Downstream tasks | Open — readmission, mortality, diagnostic coding, etc. (your choice) |
| License | PhysioNet Credentialed (CITI + DUA) |
| DUA needed | **Yes** — credentialed access, CITI training required, takes longest |
| Status | Apply ASAP if you want real-EHR results |

**Why include it despite the long ramp.** MIMIC-IV-Note is what a *reviewer*
will say is the only "real" benchmark — everything else (MEDDOCAN, ASQ-PHI)
is synthetic. To deflect that critique, you need at least one number from a
real-EHR corpus. The scale is enormous (>2M reports) — sample a subset, say
2,000 docs, for the downstream-task experiment.

---

## 6. CARMEN-I (Spanish + Catalan real EHR) — 🔴 DUA pending

| Field | Value |
| --- | --- |
| Languages | Spanish (primary), Catalan (sections) |
| Size | 2,000 documents / 6,811 COVID-19 patients |
| Source | Hospital Clínic de Barcelona |
| Annotations | Sensitive-data spans + clinical concept annotations |
| License | PhysioNet + Spanish DUA |
| DUA needed | Yes |
| Status | Apply if you want multilingual real-EHR results |

**Why include.** This is the only benchmark on our list that gives us *both*
PHI spans and clinical concept annotations on the *same* documents — i.e.
ground truth for both heads of our claim simultaneously. Transform with
Guardian Layer, then check (a) what % of sensitive spans were caught and
(b) whether the clinical concept spans extracted from the transformed text
match those extracted from the raw text.

---

## Recommended experimental order

1. **Phase 1 — synthetic data, ship today**
   - MEDDOCAN: PHI detection (done) + masked-NER utility proxy (next)
   - ASQ-PHI: PHI detection + retrieval-utility (after browser download)

2. **Phase 2 — real EHR, ship when DUAs arrive**
   - n2c2 2006: smoking classifier accuracy before vs after
   - n2c2 2014: risk-factor F1 before vs after
   - CARMEN-I: dual-head experiment (both PHI and concept fidelity)
   - MIMIC-IV-Note: scale check on real notes

The submit-DUAs-now / experiment-with-synthetic-meanwhile pattern is
standard and reviewer-acceptable: reviewers know real-EHR access takes
weeks, and synthetic results de-risk the experimental design before you
spend DUA-protected effort.

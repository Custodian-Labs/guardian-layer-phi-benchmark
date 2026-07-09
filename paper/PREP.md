# Workshop submission — preparation notes

## 1. Target venues (ACL / EMNLP, clinical / privacy NLP)

Ranked by fit. All are co-located with *ACL, EMNLP, or NAACL and take 4–8 page short/long papers on the ACL template.

| Venue | Fit | Why |
|---|---|---|
| **ClinicalNLP** (Clinical Natural Language Processing Workshop) | ★★★ best | Clinical text + de-identification is squarely in scope; MEDDOCAN/i2b2-style PHI is a recurring topic. |
| **BioNLP** (ACL) | ★★★ | Biomedical/clinical NLP; methods + resources tracks; de-id and privacy welcome. |
| **PrivateNLP** (privacy-preserving NLP, at ACL/EMNLP/NAACL in recent years) | ★★★ | Exactly our thesis — does privacy transformation preserve utility. Statistical/utility framing lands here. |
| **LOUHI** (Health Text Mining and Information Analysis, EMNLP) | ★★ | Health text mining; multilingual clinical is a plus. |
| **Findings + workshop** | ★★ | If we strengthen it (see §3) a main-conf *Findings* short paper is plausible. |

**Action:** pick one primary (recommend **ClinicalNLP** or **PrivateNLP**) and one backup; check the current year's CFP for exact page limit, anonymity, and deadline. Workshops usually allow **dual submission** and are **non-archival or archival by choice** — good for a first paper.

## 2. Framing — lead with the *method*, not the product

For a venue reviewer, "a proprietary tool is good" reads as promotion. Reframe the contribution as a **reusable evaluation methodology** and report the proprietary transform as *one system under test*:

> **How do you show a de-identification transform preserves downstream utility — rigorously?**
> We contribute (a) a paired, multi-detector protocol that scores utility **only on the spans a system actually masks**, decoupling *coverage* from *utility*; (b) an **equivalence-testing** (TOST) analysis that avoids the large-N significance trap; (c) an **error typology** of surrogate-generation failures; across 11 detectors, 7 benchmarks, 7 languages.

This makes the paper about a *problem and a measurement*, with Guardian Layer as the concrete case study — publishable and honest.

## 3. Gaps to close before submission (ranked)

1. **Reproducible baseline surrogate generator (highest impact).** The Guardian transform is closed. Add ≥1 open substitution baseline (e.g. **Faker / Presidio anonymizer / a simple same-type LLM swap**) so results are reproducible and we can say whether the effect is generic to *any* structure-preserving substitution or specific to this tool. This also directly answers "is −1.2 pts the floor of substitution?".
2. **Human/quality check of surrogates.** A small annotated sample (e.g. 200 spans) rating surrogate *validity* + *type-consistency*, to quantify the "truncation/garbling" error class rather than relying on eyeballed examples.
3. **Redaction upper-bound.** Run `masking_type=redact` (PHI → `*****`) as the "how bad it can get" anchor; contrast transform vs redact on the same detectors. Strong figure.
4. **Statistical detail.** Report per-benchmark TOST + a random-effects (per-document / per-system) model so significance isn't only pooled; add effect sizes.
5. **Reproducibility package.** Release the 250-doc subsets + scoring code (already on the dashboard) with a data statement; note licenses.
6. **Ethics / data statement.** PHI benchmarks are synthetic/public; state no real patient data; discuss dual-use of "utility-preserving de-id".

## 4. Rough timeline

- Week 1: open-surrogate baseline + redaction runs (harness already supports both).
- Week 2: per-benchmark stats, surrogate-quality annotation, figures.
- Week 3: write (draft.md → ACL LaTeX), internal review.
- Week 4: polish, ethics/repro, submit.

## 5. Author / logistics to confirm
- Author list & order (Bill / advisors / Custodian collaborators), affiliations.
- Whether Custodian is a co-author or acknowledged (affects "proprietary tool" framing & COI).
- Archival vs non-archival; anonymity requirements of the chosen workshop.

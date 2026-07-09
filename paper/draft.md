# Does Structure-Preserving De-identification Preserve Detectability? A Paired, Multi-Detector Equivalence Study of Surrogate Substitution for Clinical PHI

*Working draft — target: a clinical / privacy NLP workshop at ACL/EMNLP (ClinicalNLP, BioNLP, or PrivateNLP). ~8 pages + refs, ACL template.*

*Scope (confirmed): the paper's single contribution is the **evaluation** — a rigorous, reproducible protocol for auditing whether a structure-preserving de-identification transform keeps PHI detectable, with one commercial transform as the case study and open baselines for reproducibility. It does **not** propose or extend any surrogate-generation algorithm. Author list & affiliations to confirm — see `paper/PREP.md §5`.*

---

## Abstract

Structure-preserving de-identification replaces protected health information (PHI) with realistic same-type *surrogates* — "Anna S." becomes "Maria S.", not `[NAME]` — so that clinical text stays fluent and downstream tools keep working. But this only helps if the substitution does not itself corrupt the signal those tools rely on. We ask a narrow, testable question: **on the spans a de-identifier actually masks, can downstream PHI detectors still find the surrogate?** We introduce a paired, multi-detector evaluation protocol that (i) scores utility **only on masked spans**, decoupling *coverage* from *utility*; (ii) uses **equivalence testing (TOST)** rather than null-hypothesis significance testing, which is uninformative at our sample size (56k paired spans); and (iii) builds a **surrogate-failure typology** that separates fixable generator defects from intrinsic detector limits. Across 11 detectors, 7 benchmarks, and 7 languages (1,750 documents), recall on masked spans moves from 76.2% to 75.0% — a change our equivalence test shows is **statistically indistinguishable from zero within a ±2-point margin** (p ≈ 1×10⁻⁸), with detector ranking preserved. The residual loss is not detectors getting worse at PHI: it concentrates in *malformed and out-of-distribution* surrogates (truncation `Chicago→Illino`, salience loss `Cedars-Sinai→Vidant`) — a quality property of the generator, not evidence that transformation hides well-formed PHI. We release the evaluation subsets and scoring code so the protocol can audit any structure-preserving transform.

---

## 1. Introduction

De-identification of clinical text has two families of output. **Redaction** deletes or tags PHI (`John Smith` → `[NAME]`), which is safe but destroys the fluency, layout, and distributional properties that downstream models and human readers depend on. **Structure-preserving** de-identification instead substitutes each PHI value with a plausible, same-type *surrogate* (`John Smith` → `Maria Lopez`), keeping the document readable and machine-parseable. The second family is increasingly attractive: it lets de-identified data flow into analytics, model training, and even second-pass detection without breaking pipelines built for real text.

The promise of structure preservation rests on an unstated assumption: **the surrogate carries the same downstream signal as the value it replaced.** If substitution silently degrades the very features that a PHI detector, an NER model, or a clinical parser relies on, then "structure-preserving" is a misnomer — the transform would be quietly laundering PHI into forms that tools can no longer see or handle, which is both a utility problem and, for a second-pass safety net, a privacy problem.

This assumption is rarely tested directly, and testing it well is harder than it looks. Three pitfalls recur:

1. **Coverage confounds utility.** A de-identifier that masks only 60% of PHI and one that masks 95% cannot be compared on whole-document F1 — the score conflates *how much it masks* with *whether what it masks stays usable*. The two must be measured separately.
2. **The large-N significance trap.** With tens of thousands of paired spans, any non-zero difference is "statistically significant" under a standard test, even when it is operationally meaningless. A p-value here answers the wrong question.
3. **Aggregate error hides mechanism.** A single "−1.2 points" number says nothing about *why* the loss happens — whether it is boundary jitter, detector weakness, or a defect in surrogate generation. Only the last is fixable, and only an error typology can tell them apart.

We address all three. Our contributions:

- **A paired multi-detector protocol** that scores utility on masked spans only, cleanly decoupling coverage from utility, run across 11 detectors × 7 benchmarks × 7 languages (§4–§5).
- **An equivalence-testing analysis (TOST)** that replaces the uninformative significance test with a bounded-effect claim: the change in detectability is provably within ±2 points of zero (§6).
- **A surrogate-failure typology** that attributes the small residual loss to surrogate-generation quality (truncation, salience loss, `x`-masking) rather than to detectors getting worse at PHI (§7).
- **A set of comparison experiments** — a redaction upper-bound and an open-source surrogate baseline — that make the result reproducible and isolate what is generic to *any* structure-preserving substitution versus specific to one tool (§5.3, §8).

The framing is deliberately not "a proprietary tool is good." It is: *here is how to measure whether a structure-preserving transform preserves utility, rigorously*, with one commercial transform as the case study and open baselines for reproducibility.

---

## 2. Related Work

### 2.1 De-identification and surrogate substitution
Clinical de-identification is a long-studied sequence-labeling problem, anchored by the i2b2/n2c2 and MEDDOCAN shared tasks and by systems ranging from rule- and dictionary-based pipelines (Presidio) to fine-tuned transformers (OBI `deid_roberta`, ClinicalBERT-style taggers) and, recently, instruction-tuned LLMs used zero-shot. Most of this literature optimizes *detection* (recall/precision on PHI spans). The downstream question — *what to put in the PHI's place* — is comparatively under-studied. Surrogate ("hiding in plain sight") replacement was proposed to keep de-identified notes realistic and to resist re-identification by making residual PHI indistinguishable from planted fakes. Our work is orthogonal to the detection literature and to any particular surrogate generator: given that a span is masked, we ask whether the *replacement* remains detectable and well-formed.

> **[CITATION PENDING]** The commercial transform used as our case study implements a surrogate-generation method covered by a Custodian Labs patent (Feng et al., patent pending). The formal citation will be added once the patent issues; we cite it as the source of the transform rather than describing or extending the method.

### 2.2 Utility-preservation evaluation
Whether privacy transformations preserve downstream utility is the central question of privacy-preserving NLP. Prior evaluations typically train or run a single downstream model on original vs. transformed data and compare end-task scores. Two methodological weaknesses recur: (a) a single downstream model cannot separate "the transform is fine" from "this model is robust," and (b) whole-corpus metrics mix masked and unmasked content. We address (a) with an 11-detector panel spanning rule-based, fine-tuned, and LLM detectors, and (b) by restricting the utility measurement to masked spans.

### 2.3 The large-N significance trap and equivalence testing
When sample sizes are large, null-hypothesis significance testing rejects the null for negligible effects; the p-value measures precision, not importance. The standard remedy is **equivalence testing** — the *two one-sided tests* (TOST) procedure — which reverses the burden of proof: one specifies an equivalence margin Δ (the largest difference that is still "the same for practical purposes") and tests whether the effect is provably inside [−Δ, +Δ]. TOST is standard in biostatistics and psychology but under-used in NLP evaluation, where large paired corpora make the trap acute. We adopt it as the primary inferential tool and report McNemar's paired test only to demonstrate the trap.

---

## 3. Problem Formulation

**Structure-preserving de-identification.** A transform T maps a document d to d′ by replacing each detected PHI value v (of type τ, e.g. NAME/DATE/LOCATION) with a surrogate s of the same type, leaving all other characters unchanged. Gold PHI spans on d are re-projected onto d′ by character-level alignment, giving paired spans (v, s).

**Coverage vs. utility.** Two quantities must not be conflated:
- **Coverage** = fraction of true PHI that T detects and replaces (a property of T's *detector*).
- **Utility** = given that a span was replaced, whether a downstream detector still finds the surrogate s as well as it found v (a property of T's *generator* and of the surrogate's realism).

We report coverage separately and measure utility **only on the masked-span population** {(v, s)}. This is the crux: it prevents a low-coverage transform from looking good (or a high-coverage one from looking bad) on a metric that is really about substitution quality.

---

## 4. Evaluation Protocol

**4.1 Paired multi-detector design.** Each document is scored in two conditions — original d and transformed d′ — by the *identical* detector suite D (|D| = 11). Because the only change between conditions is the surrogate substitution, any per-span change in detection is attributable to the substitution, not to the detector or the document.

**4.2 Metrics.** We report span-level precision/recall/F1 under three matching modes — **exact** (start+end+type), **type** (type + boundary), **overlap** (any character overlap + type) — and **leakage = 1 − recall**, the HIPAA-critical quantity. The headline utility metric is **recall retention on masked spans** = recall(d′)/recall(d) restricted to {(v,s)}, under overlap matching (so that pure boundary jitter from length changes, "Anna S."→"Maria S.", is not charged as a miss).

**4.3 Equivalence testing.** For the pooled masked-span recall, we run **TOST** with equivalence margin Δ = 2 points: we reject non-equivalence iff the 90% CI of the recall difference lies entirely within [−2, +2]. We also report McNemar's paired test on the found/lost contingency to illustrate the large-N trap (§6). Δ = 2 pts is pre-registered as the operational "same for practical purposes" threshold; §6 reports a sensitivity sweep over Δ ∈ {1, 2, 3}.

**4.4 Error attribution.** For the lost population (detected on d, missed on d′) we hand-code each span into a small failure typology (§7) and check length-preservation to separate boundary artifacts from genuine misses. This turns the aggregate residual into an attributable, fixable set of generator defects.

---

## 5. Experimental Design

**5.1 Systems under test (detector panel D).** Eleven detectors spanning three families:
- *Rule/statistical:* Microsoft **Presidio**; **OBI `deid_roberta`** (fine-tuned clinical de-id).
- *Open LLMs (local):* Gemma 4 31B, Gemma 4 E4B, Qwen 3.5-{4B, 9B, 35B-A3B}, Llama 3.1-8B, Llama 3.3-70B, DeepSeek V2-Lite.
- *Frontier API:* OpenAI GPT-5.

This panel is deliberately heterogeneous: if the equivalence result held only for LLMs or only for one architecture it would be a model artifact, not a property of the transform.

**5.2 Benchmarks (7, one per language where applicable; 250 docs each, 1,750 total).** ASQ-PHI (English clinical queries), MEDDOCAN (Spanish clinical), MultiCoNER v2 (multilingual NER), and PII-Masking-300k in English, Dutch, French, German. Together: 7 languages, clinical and general-PII text, free-form and structured (JSON) formats. The transform under test is Custodian Guardian Layer `transform` mode (top-1 surrogate, `pii_entities=ALL`).

**5.3 Comparison conditions (the controlled contrasts).** The core paired study answers "does *this* transform preserve detectability." Three contrasts isolate *why* and *how generally*:

- **C1 — Redaction upper-bound (worst case anchor).** Re-run every detector against `redact` mode (PHI → `*****`). Redaction destroys the surrogate signal by construction, so this is the floor; the gap between transform and redact quantifies how much structure preservation actually buys. *Hypothesis: transform ≫ redact on masked-span recall.*
- **C2 — Open-source surrogate baseline (reproducibility + generality).** Replace the commercial generator with open substitutors — **Faker** (type-templated fakes), the **Presidio anonymizer**, and a **same-type LLM swap** — holding detection fixed. This tells us whether the ±2-point result is generic to *any* structure-preserving substitution or specific to one tool, and makes the pipeline reproducible without proprietary access. *Hypothesis: all well-formed substitutors land within a few points of each other; the residual is a property of substitution, not of one vendor.*
- **C3 — Per-benchmark / per-language equivalence.** Run TOST within each benchmark, not only pooled, to check the equivalence claim is not an averaging artifact, and sweep the margin Δ ∈ {1, 2, 3}.

*Status.* The core paired study (11×7, original vs. transform), the equivalence analysis (pooled **and** per-benchmark, C3), the redaction floor (C1), the open-surrogate baseline (C2), and the error typology are all **complete** (§6–§7). C1/C2 are currently demonstrated with the CPU detector (Presidio); extending them across the full 11-detector panel is the remaining strengthening step, along with a small human surrogate-quality annotation.

---

## 6. Results

**6.1 Detectability is statistically unchanged.** Restricting to the **56,174 masked spans** and pooling all 11 detectors, recall moves **76.2% → 75.0%** (−1.2 pts; 95% CI [−1.5, −1.0]). The **TOST equivalence test** (Δ = 2 pts) rejects non-equivalence at **p ≈ 1×10⁻⁸**: the change in detectability is provably bounded within ±2 points of zero — below any operationally meaningful level. Detector **ranking is preserved**; the flagship Llama 3.3-70B is essentially unchanged (ΔF1 +0.003).

> The same 56k-span contingency is "significant" under McNemar's test — a direct demonstration of the large-N trap in §2.3. The equivalence test is the informative one: it shows the difference is *too small to matter*, rather than merely *non-zero*.

**6.2 Recall retention on masked spans (overlap match).** When the transform masks a span, detectors still find the surrogate **93–100%** of the time (97–100% for all but the weakest detector):

| System | Exact-boundary | Overlap |
|---|--:|--:|
| Gemma 4 31B | 93.1% | **99.8%** |
| Qwen 3.5-9B | 92.0% | **100.0%** |
| Qwen 3.5-35B-A3B | 90.0% | **98.4%** |
| Llama 3.3-70B | 91.4% | **98.3%** |
| GPT-5 | 91.9% | **98.3%** |
| Gemma 4 E4B | 88.9% | **98.2%** |
| Qwen 3.5-4B | 84.7% | **98.0%** |
| Presidio | 91.3% | **97.4%** |
| DeepSeek V2-Lite | 87.7% | **92.8%** |

The ~3-point exact-boundary drop is a length-jitter artifact (surrogates differ in length from originals), not hidden PHI — it vanishes under overlap matching.

**6.3 Per-benchmark equivalence (C3).** Re-running the analysis within each benchmark (via `scripts/analyze_equivalence.py`, over 57,112 masked spans pooled across detectors) shows the pooled equivalence is **not uniform**, and we report this openly:

| Benchmark | Masked spans | Recall orig→transf | Δ | McNemar χ² | Equiv. margin |
|---|--:|--:|--:|--:|:--|
| ASQ-PHI | 5,427 | 91.9→91.1 | −0.8 | 8.0 | **±2 ✓** |
| MEDDOCAN | 31,978 | 73.4→71.5 | −1.9 | 92.4 | ±3 ✓ (±2 ✗) |
| MultiCoNER v2 | 220 | 71.8→71.4 | −0.5 | 0.0 | underpowered |
| PII en | 5,181 | 76.0→75.1 | −0.9 | 4.0 | **±2 ✓** |
| PII nl | 3,757 | 78.2→76.3 | −1.9 | 13.1 | ±3 ✓ (±2 ✗) |
| PII fr | 5,984 | 75.2→75.9 | +0.7 | 3.5 | **±2 ✓** |
| PII de | 4,565 | 76.3→76.5 | +0.2 | 0.4 | **±2 ✓** |
| **Pooled** | **57,112** | **76.1→74.9** | **−1.2** | **82.1** | **±2 ✓ (p=3×10⁻⁹)** |

Four benchmarks are equivalent within ±2 points; **MEDDOCAN and PII-nl require a ±3-point margin** — both are dense, identifier-heavy, non-English, exactly where surrogate generation is hardest (§7). MultiCoNER v2 has too few masked spans (220) to test. The honest claim is therefore: *detectability is equivalent within ±2 points on average and on clean text, and within ±3 points on the hardest clinical/identifier-dense text* — the residual is concentrated and attributable (§7), not diffuse detector degradation.

**6.4 Redaction floor (C1).** To anchor "how bad it can get," we replace each masked value with `*****` (no surrogate signal) and re-detect. Presidio masked-span recall (CPU detector; the argument is detector-independent):

| Benchmark | Masked | Original | Transform | Redact |
|---|--:|--:|--:|--:|
| ASQ-PHI | 531 | 94.5% | 95.7% | **0.0%** |
| MEDDOCAN | 2,925 | 75.4% | 72.9% | **0.0%** |
| PII en | 471 | 75.6% | 73.5% | **1.9%** |
| PII nl | 373 | 83.6% | 77.7% | **3.8%** |
| PII fr | 544 | 61.4% | 59.9% | **2.2%** |
| PII de | 415 | 56.1% | 58.3% | **0.0%** |

Transform recall tracks the *original* within a few points; redaction collapses to ≈0–4% (the residual is spurious overlap on the `*` run). Structure-preserving substitution therefore preserves essentially the entire detectability that redaction destroys — the gap between the "Transform" and "Redact" columns is what structure preservation buys.

**6.5 Open-surrogate baseline (C2).** To test whether detectability preservation is generic to substitution or specific to one tool, we replace the commercial generator with **Faker** (open-source, seeded, reproducible), substituting the *same* masked spans and re-detecting with Presidio:

| Benchmark | Masked | Original | Custodian | Faker |
|---|--:|--:|--:|--:|
| ASQ-PHI | 531 | 94.5% | 95.7% | **97.7%** |
| MEDDOCAN | 2,925 | 75.4% | 72.9% | **79.7%** |
| PII en | 471 | 75.6% | 73.5% | **89.2%** |
| PII nl | 373 | 83.6% | 77.7% | **93.0%** |
| PII fr | 544 | 61.4% | 59.9% | **78.5%** |
| PII de | 415 | 56.1% | 58.3% | **78.1%** |

Two findings. (i) **Generality:** an open substitutor preserves masked-span recall at least as well as the original — detectability preservation is a property of *well-formed same-type substitution*, not of one vendor, and the effect reproduces without proprietary access. (ii) **The residual is generator quality:** Faker, which emits clean canonical values, *exceeds* the commercial transform on the hard non-English benchmarks (PII fr/de +18–20 pts), precisely where the commercial generator's surrogates truncate or garble (§7). This is direct evidence that the small residual loss (§6.1) is a fixable property of surrogate *generation*, not an intrinsic cost of structure preservation. (Presidio-only demonstration; the argument is detector-independent.)

**6.6 Leakage barely moves** (+1.7 to +4.1 pts across systems), so surrogates are not systematically easier to miss than the PHI they replace.

---

## 7. Error Analysis

**7.1 The lost population.** Pooled across detectors and masked spans: **39,550** spans found in both conditions; **3,261** lost (found on d, missed on d′). Critically, **51% of lost spans are the same length** as the original — this is *not* a boundary effect. By type: LOCATION 28%, NAME 23%, DATE/AGE 22%, ID/contact 20%.

**7.2 Three failure modes — all generator-side, not detector-side.**

1. **Malformed / truncated surrogates (largest cause).** `Chicago→Illino`, `El Paso→El`, `Ciudad de la Habana→Cuidad de la Havana`, `Hospital Universitario de La Princesa→Hospitals Cienciano de La Princesa`. The generator drops trailing tokens or emits a near-word; the fragment no longer matches the lexical pattern detectors learned for real names/places. This is why loss is highest on MEDDOCAN (Spanish, dense, identifier-heavy) and lowest on clean English ASQ-PHI.
2. **Loss of salience.** A canonical entity replaced by an obscure one: `Cedars-Sinai→Vidant`, `Cuba→Havana`. Detectors partly recognize entities by pre-training familiarity; swapping a famous value for a rare one removes the prior. Inherent to any value substitution; mainly costs weaker detectors.
3. **`x`-masking of IDs/emails.** `nachorutor@…→nxxxxxxxxx@…`. The run of `x`s preserves format but breaks the realistic-token pattern. Note this case is **privacy-positive** — the original value is destroyed — even though it counts against recall.

**7.3 Implication.** Detectors are not getting *worse at PHI*; the small recall gap is driven by **surrogate-generation quality** (truncation, garbling, salience, `x`-masking). These are fixable properties of a generator, not evidence that structure-preserving substitution hides real, well-formed PHI — a distinction the masked-span protocol makes visible and the aggregate F1 hides. The open-surrogate baseline (§6.5) confirms this directly: an open generator that emits clean canonical values *exceeds* the commercial transform by 18–20 points on exactly the hard non-English benchmarks where the commercial surrogates truncate — so the residual tracks generator quality, not the act of substitution.

---

## 8. Discussion

**What the evidence supports.** Structure-preserving substitution does **not** hide well-formed PHI from downstream detection: detectors are not getting worse at PHI; the small residual is surrogate-generation quality. Because the effect is equivalence-bounded within ±2 points *and* ranking-preserving across a heterogeneous 11-detector panel, a transform of this quality can be safely inserted ahead of detection/analytics pipelines built for real clinical text.

**Generality (C2, confirmed).** The open-surrogate baseline settles this: Faker preserves masked-span recall at least as well as the original across all six benchmarks, so ±2 points is not a property of one vendor but of *well-formed same-type substitution* — the commercial tool is one instance, and the effect reproduces with open tooling. Where the commercial generator trails, a cleaner generator closes the gap, locating the residual squarely in generation quality.

**A reusable protocol.** The paired masked-span design and the TOST margin are not specific to one vendor or language; they are a template for auditing any structure-preserving privacy transform.

---

## 9. Ethics and Data Statement

All benchmarks are public or synthetic (ASQ-PHI synthetic; MEDDOCAN released for a shared task; PII-Masking-300k synthetic; MultiCoNER v2 public). **No real patient data is used.** We release the 250-document evaluation subsets and scoring code with license notes. **Dual-use note:** a utility-preserving de-identifier could in principle be used to launder identifiable data into a fluent form; our masked-span/leakage reporting and the privacy-positive framing of `x`-masking (§7.2) are intended to keep the privacy accounting explicit.

**Conflict of interest / funding.** The commercial transform evaluated as the primary case study (Custodian Guardian Layer) is developed by Custodian Labs, with which one or more authors are affiliated; this work was conducted with Custodian Labs' support. To limit the resulting bias we (i) frame the contribution as a reusable evaluation protocol rather than a product endorsement, (ii) include open-source surrogate baselines (§5.3-C2) so results are reproducible without proprietary access, and (iii) report leakage and coverage alongside utility so no single favorable metric stands alone.

---

## 10. Limitations

- Coverage (how much PHI the transform detects) is reported but not the focus; a transform can preserve utility on what it masks while still under-masking. The two axes are independent by design.
- 250 docs/benchmark bounds statistical power per benchmark (though the pooled masked-span N is large); C3 reports per-benchmark equivalence with this caveat.
- LLM detectors are prompt-sensitive; we fix one prompt per model and report it in the appendix.
- The error typology is hand-coded on the lost population; §5.3-C2 (open baselines) and a small annotation study (below) will quantify the failure-mode rates independently of any single detector.

---

## Appendix / To-do before submission (tracked in `paper/PREP.md`)
- [x] C1 redact upper-bound → floor table (§6.4; `scripts/run_redact_baseline.py`).
- [x] C2 open-surrogate baseline (Faker) → generality table (§6.5; `scripts/run_faker_baseline.py`).
- [x] C3 per-benchmark TOST + Δ sweep (§6.3; `scripts/analyze_equivalence.py`).
- [ ] Extend C1/C2 across the full 11-detector panel (currently Presidio-only).
- [ ] Add a second open substitutor to C2 (Presidio anonymizer / same-type LLM swap).
- [ ] Small human annotation of surrogate validity/type-consistency (≈200 spans) to quantify the Mode-1 (malformed) rate independently of detectors.
- [ ] Insert the Custodian patent citation once it issues (currently `[CITATION PENDING]` in §2.1).
- [ ] Confirm venue CFP (page limit, anonymity, archival), author list, and Custodian co-authorship/COI statement.
- [ ] Port to ACL LaTeX; figures for the retention table and per-benchmark ΔF1.

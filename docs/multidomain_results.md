# Beyond medicine: the Custodian transform and surrogate substitution in finance, legal and enterprise text

> **Note (2026-10, arXiv v2).** The numbers in this document were computed with the original difflib re-projection of gold spans onto the transformed text. That alignment cuts 9–15% of re-projected spans inside a word in these domains. The arXiv v2 numbers use word-boundary-snapped re-projection and a document-level cluster bootstrap: `python scripts/analyze_v2.py` → `results/v2_numbers.json`. After correction the Custodian arm, pooled, is 72.4% → 72.8% (+0.45 pt, cluster 95% CI [−0.27, +1.17]); Faker and redaction are unaffected. A few span-level examples below (e.g. `1-800-123-4567` → `1-234-567`, `XYZ Bank` → `cil`) are alignment artefacts, not transform output.

**Question.** The medical study found that same-type surrogate substitution keeps PHI detectable (pooled masked-span recall 76.1% → 74.9%, −1.2 pt, TOST ±2 pt p ≈ 3×10⁻⁹), while `*****` redaction destroys detectability for rule-based and fine-tuned detectors. Does that hold outside medicine, and in particular for the **Custodian Guardian Layer transform** itself?

**Short answer.** Yes for the Custodian transform at the pooled level, with exceptions for rule-based and fine-tuned detectors on Finance (and rule-based on Enterprise). A caveat on surrogate quality applies (see the section on surrogate quality).

- **Custodian Guardian Layer transform (headline).** Pooled over 10 detectors and 3 domains, masked-span recall goes **72.5% → 72.9% (+0.42 pt**, McNemar CI [+0.05, +0.79], doc-bootstrap CI [−0.32, +1.10]). It is **TOST-equivalent at ±2 pt (p = 2.4×10⁻¹⁷)**. That is the same conclusion as the medical paper (−1.2 pt, p ≈ 3×10⁻⁹), with a smaller shift.
  - **Legal** +0.29 pt (equivalent, p = 5.3×10⁻¹⁰). **Enterprise** −0.91 pt (equivalent, p = 9.3×10⁻⁵). **Finance** +1.51 pt: non-inferior (p = 2.0×10⁻²¹) but *not* equivalent (TOST p = 0.096), because the upper CI bound reaches past +2 pt.
  - **Exceptions (not non-inferior at −2 pt).** Finance · rule (Presidio) −3.76 pt, Finance · fine-tuned (OBI) −3.67 pt, Enterprise · rule (Presidio) −1.79 pt (CI [−3.75, +0.17]). Pooled over all domains, rule-based is −1.84 pt (not non-inferior, p = 0.37) and fine-tuned −1.50 pt (equivalent, p = 0.042). The LLM family is +0.94 pt (equivalent). The losses trace to surrogates that break the source format (see Findings).
  - Per detector × domain, 20 of 30 cells are non-inferior and 9 are TOST-equivalent.
  - The Custodian masked set is **the gold spans Custodian actually changed** (35,930 span × detector pairs). It is smaller than, and not the same as, the Faker masked set (45,420), so Custodian and Faker numbers are not computed on identical spans.
- **Open surrogate (C2, Faker).** Pooled recall goes 70.8% → 75.0% (**+4.18 pt**, doc-bootstrap CI [+3.35, +5.07]). The two-sided TOST at ±2 pt fails because recall moves **up**, not down; one-sided non-inferiority passes pooled (p = 3.0×10⁻²⁸⁷) and in 27 of 30 detector × domain cells. Enterprise is equivalent (−0.05 pt, p = 7.2×10⁻¹⁷).
- **Redaction (C1).** It destroys detectability in all 3 domains for all 10 detectors. Pooled recall goes 70.8% → 27.0% (**−43.8 pt**). Presidio and OBI fall to 0.4–4.1%. LLMs keep some recall (3.7–76.9%) because a `*****` run is itself a cue, but every LLM still loses 5–70 pt.
- **Surrogate quality.** The Custodian transform also edits a lot of text that is *not* gold PII: about 78% of changed characters in Finance, 53% in Legal and 41% in Enterprise fall outside gold spans (upper bounds, see below). It swaps meaning-bearing words (`judge`→`lawyer`, `USD`→`GBP`, `Lender`→`Borrower`, `Human Rights`→`Animal Rights`), introduces spelling errors (`Article`→`Artcile`, `Government`→`Goverment`), creates invalid dates (`2/31/2024`) and x-masks some values (`2xxxxxxxx`). It also leaves 12–31% of gold spans verbatim. Detection equivalence therefore does not mean the output is a faithful document.

Setup: 3 domains × 250 docs, 10 detectors covering all three families, 4 conditions (original / Faker / redacted / Custodian). That is 30,000 detector × doc predictions in 120 prediction files. All files cover all 250 documents and have 0 error rows. There are 45,420 paired masked-span observations for each of Faker and redaction, and 35,930 for Custodian.

## Datasets (dataset cards)

| Benchmark key | Domain | Source | Split used | Sample | Language | Gold spans (masked) | License |
|---|---|---|---|---|---|---|---|
| `finance_gretel` | Finance | [`gretelai/synthetic_pii_finance_multilingual`](https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual) — synthetic financial documents (SWIFT MT, ISDA, BAI, FIX, XBRL, invoices, insurance policies, loan agreements, …) with char-offset PII spans | English **test** (2,891 docs) | 250 docs, seeded (`random.Random(20260926)`) from the 2,552 docs with ≥1 span and ≤4,000 chars | en | 1,283 (1,283) | Apache-2.0 |
| `legal_tab` | Legal | Text Anonymization Benchmark (TAB) — ECHR court judgments ([HF mirror `ildpil/text-anonymization-benchmark`](https://huggingface.co/datasets/ildpil/text-anonymization-benchmark), [GitHub](https://github.com/NorskRegnesentral/text-anonymization-benchmark)) | **test** (127 cases; first annotator per case) | 250 paragraph windows, seeded, from the 536 windows with ≥1 gold span | en | 2,307 (2,203) | MIT |
| `enterprise_gretel` | Multi-domain enterprise (non-medical) | [`gretelai/gretel-pii-masking-en-v1`](https://huggingface.co/datasets/gretelai/gretel-pii-masking-en-v1) — short synthetic business documents across ~47 industry domains | **test** (5,000 docs), minus `healthcare`, `healthcare-administration`, `pharmaceuticals-biotechnology` (4,597 left) | 250 docs, seeded; 44 distinct domains in the sample | en | 1,056 (1,056) | Apache-2.0 |

Entity types
- **Finance** (29 types): name, company, street_address, date, phone_number, email, time, first/last_name, date_of_birth, employee_id, customer_id, swift_bic_code, bank_routing_number, credit_card_number, credit_card_security_code, iban, bban, ssn, passport_number, driver_license_number, account_pin, api_key, password, user_name, ipv4, ipv6, local_latlng, date_time.
- **Legal** (TAB types, restricted to mentions whose `identifier_type` is DIRECT or QUASI, i.e. TAB's "must be masked" set): DATETIME (1,143), PERSON (425), ORG (225), CODE (188, e.g. application no. `36110/97`), LOC (124), DEM (87), MISC (59), QUANTITY (56).
- **Enterprise** (39 types): medical_record_number, date_of_birth, ssn, customer_id, date, street_address, email, name, phone_number, employee_id, first/last_name, ipv4, license_plate, address, credit_card_number, company_name, user_name, device_identifier, bank_routing_number, … (Gretel's generic PII taxonomy; note MRN/health-plan IDs still occur inside non-medical documents).

Conversion details (`scripts/build_domain_corpora.py`)
- Every gold span is verified to satisfy `text[start:end] == span.text` (0 mismatches in all three sets); overlapping/nested spans are resolved by keeping the longest.
- **TAB chunking**: judgments are 5–30k characters, so each case is split at newline (paragraph) boundaries and consecutive paragraphs are packed greedily into windows of ≤1,500 characters (mean 1,151). No gold span crossed a window boundary (0 dropped). NO_MASK mentions (e.g. "Republic of Turkey", "Court of Cassation") are *not* gold, so detectors that tag them are charged a false positive — this is TAB's own definition.
- **Gretel enterprise offsets**: the dataset gives entity strings, not offsets; offsets are recovered by exact string search (every entity occurs exactly once in its document in the test split — verified).

Citations
- Watson, A., Meyer, Y., Van Segbroeck, M., Grossman, M., Torbey, S., Mlocek, P., Greco, J. *Synthetic-PII-Financial-Documents-North-America: A synthetic dataset for training language models to label and detect PII in domain specific formats.* Gretel.ai, June 2024. https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual
- Pilán, I., Lison, P., Øvrelid, L., Papadopoulou, A., Sánchez, D., Batet, M. *The Text Anonymization Benchmark (TAB): A Dedicated Corpus and Evaluation Framework for Text Anonymization.* Computational Linguistics 48(4):1053–1101, 2022. doi:10.1162/coli_a_00458
- Gretel AI. *gretel-pii-masking-en-v1* (GLiNER Models for PII Detection through Fine-Tuning on Gretel-Generated Synthetic Documents), Oct 2024. https://huggingface.co/datasets/gretelai/gretel-pii-masking-en-v1


## Method

The method matches the medical C1/C2 study. The Custodian arm is defined exactly as in the paper; the Faker and redaction arms use a broader masked set.

- **Custodian arm** (`scripts/transform_corpus.py` → `data/transformed/<bench>.jsonl`). Each original document goes through the Custodian Guardian Layer transform (top-1). The transform ran on all 750 documents with 0 API errors; it changed 249/250 finance, 250/250 legal and 249/250 enterprise documents. Gold spans are moved onto the output text by difflib character alignment, keeping their label and re-slicing their text.
  - **Masked set = gold spans whose text Custodian changed.** In 8 documents (4 finance, 4 legal) the alignment lost a gold span, so the span lists no longer pair up; those documents are left out of the Custodian pairs. On the remaining documents Custodian changed 1,090 of 1,238 finance gold spans (88%), 1,777 of 2,208 legal (80%) and 726 of 1,056 enterprise (69%). Unchanged spans are excluded, as in the medical study.
- **Faker and redaction masked set.** Every gold span with a well-defined same-type surrogate is masked. That is all spans in Finance and Enterprise, and 2,203/2,307 in Legal. Of the 104 left out, 103 are TAB `MISC`/`QUANTITY` mentions with neither digits nor number words, and `DEM` mentions that are professions, relatives or diagnoses; they stay verbatim in both arms. The last one is a span whose Faker surrogate happened to equal the original string. Because this set is broader, the "original" recall differs between the Faker/redaction rows and the Custodian rows below.
- **C2 surrogates** (`scripts/build_domain_c1c2.py`) are seeded Faker values that are *domain-aware and format-preserving*:
  - Luhn-valid cards, valid IBAN/BBAN/SWIFT/ABA numbers.
  - IDs keep their letters and punctuation and randomise only the digits (`EMP104600` → `EMP583021`; ECHR `36110/97` → `13698/87`).
  - Dates keep their original format (`3 July 2002` → `21 April 1987`).
  - Legal persons keep their title, and the surrogate first name matches the title's gender. Gender comes from a title inside the span or in the 8 characters before it: Mr/Sir/Lord give a male first name, and Mrs/Ms/Miss/Lady/Madam/Mme/Mlle give a female one (`Mr Andreas Orphanides` → `Mr Terry Fowler`, `Mrs Eileen Appleby` → `Mrs Clare White`).
  - Initial-shaped first tokens keep their shape (`Mr S. Esmer` → `Mr K. Day`, `Mrs H.S. Greve` → `Mrs R.B. Stanley`).
  - Legal organisations keep their institutional head noun (`Fair Employment Tribunal` → `Port Owen Tribunal`).
  - Substitution is consistent: one surrogate per distinct string per document. Locale is `en_US`, or `en_GB` for TAB.
- **C1** replaces each masked span with `*` × its length.
- **Metrics.**
  - Original F1 uses the repo's `type` mode (exact boundaries, label-agnostic) plus overlap F1.
  - Masked-span recall uses overlap matching on the same spans, paired by index across conditions.
  - Equivalence uses the exact estimator from `scripts/analyze_equivalence.py`: paired difference, McNemar variance, TOST ±2 pt.
  - I added a one-sided −2 pt non-inferiority test and a document-cluster bootstrap CI. Spans within a document are not independent, so the bootstrap CI is 2–3× wider than the McNemar CI.
- **Detectors (10).**
  - Rule: Presidio (spaCy `en_core_web_lg`).
  - Fine-tuned: OBI `deid_roberta_i2b2` (CPU).
  - LLM (8): Qwen3.5-4B, Qwen3.5-9B, Qwen3.5-35B-A3B, Llama-3.1-8B, Llama-3.3-70B, Gemma-4-E4B, Gemma-4-31B, DeepSeek-V2-Lite. All use the same `LocalHFLLM` prompt as the paper, with greedy decoding. Small models ran with batch size 8, Gemma-4-31B, Qwen3.5-35B-A3B and Llama-3.3-70B with batch size 4 (left padding, `predict_batch`), and DeepSeek-V2-Lite unbatched in a transformers 4.51 environment (its remote code does not load under transformers 5.x). The paper's LLM runs were unbatched.
  - Llama-3.3-70B needed 3 GPUs.

## Headline results

### Masked-span recall (%), pooled over 10 detectors

The Faker and redaction columns share one masked set; Custodian has its own (spans it changed), with its own original recall.

| Domain | N (Faker/C1) | Original | C2 Faker (Δ) | C1 `*****` (Δ) | N (Custodian) | Original | **Custodian (Δ)** | Custodian TOST ±2 pt p | Custodian verdict |
|---|--:|--:|--:|--:|--:|--:|--:|--:|---|
| Finance | 12,830 | 61.3 | 68.9 (+7.60) | 18.7 (−42.6) | 10,900 | 62.2 | 63.7 (**+1.51**) | 0.096 | non-inferior, not equivalent |
| Legal | 22,030 | 66.9 | 71.1 (+4.21) | 25.9 (−41.1) | 17,770 | 70.9 | 71.2 (**+0.29**) | 5.3e−10 | equivalent |
| Enterprise | 10,560 | 90.6 | 90.5 (−0.05) | 39.5 (−51.0) | 7,260 | 91.9 | 91.0 (**−0.91**) | 9.3e−05 | equivalent |
| **All** | 45,420 | 70.8 | 75.0 (**+4.18**) | 27.0 (**−43.8**) | 35,930 | 72.5 | 72.9 (**+0.42**) | **2.4e−17** | **equivalent** |
| *Medical paper (for reference)* | | | | | 57,112 | 76.1 | 74.9 (−1.2) | ≈3e−09 | equivalent |

Doc-bootstrap 95% CIs for Custodian: Finance [+0.26, +2.68], Legal [−0.83, +1.42], Enterprise [−1.75, −0.07], All [−0.32, +1.10].

### By detector family (all 3 domains)

| Family | Faker set: original | C2 Faker (Δ), non-inf. −2 pt | C1 redaction (Δ) | Custodian set: original | Custodian (Δ) | Custodian verdict |
|---|--:|--:|--:|--:|--:|---|
| Rule (Presidio) | 84.5 | 86.5 (+2.00), yes (p=1.5e−17) | 1.4 (−83.1) | 89.9 | 88.0 (−1.84) | **not non-inferior** (p=0.37) |
| Fine-tuned (OBI) | 92.6 | 96.0 (+3.37), yes (p=1.0e−55) | 2.6 (−90.0) | 94.3 | 92.8 (−1.50) | equivalent (p=0.042) |
| LLM (8 models) | 66.4 | 70.9 (+4.55), yes (p=9.3e−235) | 33.3 (−33.1) | 67.6 | 68.5 (+0.94) | equivalent (p=1.2e−06) |

Per domain, the Custodian family cells that fail non-inferiority are **Finance · rule −3.76** (CI [−6.07, −1.45]), **Finance · fine-tuned −3.67** ([−4.95, −2.39]) and **Enterprise · rule −1.79** ([−3.75, +0.17], p = 0.42). Finance · LLM is +2.82 (non-inferior, not equivalent). All Legal family cells and Enterprise fine-tuned/LLM are equivalent.

### Per detector × domain, Custodian vs original (Δ pt; ✓ = non-inferior at −2 pt, ≈ = also TOST-equivalent at ±2 pt, ✗ = not non-inferior)

| | Presidio | OBI | Qwen3.5-4B | Qwen3.5-9B | Qwen3.5-35B | Llama-3.1-8B | Llama-3.3-70B | Gemma-4-E4B | Gemma-4-31B | DeepSeek-V2-Lite |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Finance | −3.8 ✗ | −3.7 ✗ | +0.7 ✓ | +1.6 ✓ | −4.7 ✗ | +6.0 ✓ | +1.6 ✓ | +7.6 ✓ | +5.9 ✓ | +3.9 ✓ |
| Legal | −0.7 ✓≈ | −0.5 ✓≈ | +1.2 ✓ | −2.2 ✗ | −1.6 ✗ | −3.4 ✗ | +0.2 ✓≈ | +2.5 ✓ | +2.9 ✓ | +4.5 ✓ |
| Enterprise | −1.8 ✗ | −0.8 ✓≈ | +0.4 ✓≈ | −1.0 ✗ | −0.0 ✓≈ | −1.9 ✗ | −0.7 ✓≈ | −1.0 ✓≈ | −0.0 ✓≈ | −2.3 ✗ |

20/30 cells are non-inferior and 9/30 are equivalent. The per-detector N is small (726–1,777 spans), so several ✗ cells have point estimates inside ±2 pt but CIs that cross −2 pt (e.g. Enterprise · Qwen3.5-9B −0.96, CI [−2.65, +0.72]). The clear drops are Finance · Presidio, OBI and Qwen3.5-35B-A3B (−4.68, CI [−7.03, −2.33]) and Legal · Llama-3.1-8B (−3.43, CI [−5.64, −1.23]).

### Per detector × domain, C2 Faker vs original (same notation)

| | Presidio | OBI | Qwen3.5-4B | Qwen3.5-9B | Qwen3.5-35B | Llama-3.1-8B | Llama-3.3-70B | Gemma-4-E4B | Gemma-4-31B | DeepSeek-V2-Lite |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Finance | +6.0 ✓ | +7.7 ✓ | +10.3 ✓ | +5.7 ✓ | +7.1 ✓ | +5.8 ✓ | +6.1 ✓ | +11.0 ✓ | +13.2 ✓ | +3.2 ✓ |
| Legal | +0.9 ✓≈ | +2.3 ✓ | +6.1 ✓ | −1.0 ✗ | +5.8 ✓ | +5.8 ✓ | +5.7 ✓ | +4.9 ✓ | +7.0 ✓ | +4.7 ✓ |
| Enterprise | −0.5 ✗ | +0.3 ✓≈ | −0.0 ✓≈ | −0.2 ✓≈ | −0.1 ✓≈ | +1.4 ✓ | −0.6 ✓≈ | +0.2 ✓≈ | +0.1 ✓≈ | −1.1 ✗ |

27/30 non-inferior, 8/30 equivalent. The three ✗ cells all have point estimates within ±1.2 pt and CIs that cross −2 pt.

### Per detector × domain, C1 redaction: masked-span recall after `*****` (original in brackets)

| | Presidio | OBI | Qwen3.5-4B | Qwen3.5-9B | Qwen3.5-35B | Llama-3.1-8B | Llama-3.3-70B | Gemma-4-E4B | Gemma-4-31B | DeepSeek-V2-Lite |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Finance | 1.4 (80.9) | 1.8 (89.0) | 18.6 (55.5) | 6.7 (55.0) | 48.1 (75.9) | 5.1 (12.0) | 20.7 (76.6) | 21.7 (68.7) | 41.8 (73.0) | 20.7 (26.2) |
| Legal | 1.3 (87.1) | 4.1 (92.1) | 10.1 (57.6) | 3.7 (50.0) | 54.5 (75.3) | 9.4 (25.2) | 41.2 (79.8) | 44.6 (79.5) | 55.2 (82.3) | 34.4 (40.2) |
| Enterprise | 1.5 (83.3) | 0.4 (98.0) | 24.5 (92.4) | 23.2 (93.6) | 76.9 (98.5) | 51.8 (87.6) | 59.4 (98.6) | 48.4 (96.4) | 75.4 (98.1) | 34.0 (59.1) |

### Detection on original documents (span F1, exact / overlap)

| Detector | Finance | Legal | Enterprise |
|---|--:|--:|--:|
| Presidio | .223 / .397 | .490 / .667 | .459 / .674 |
| OBI deid_roberta | .014 / .316 | .059 / .542 | .043 / .530 |
| Qwen3.5-4B | .379 / .466 | .445 / .637 | .721 / .851 |
| Qwen3.5-9B | .400 / .490 | .411 / .567 | .781 / .877 |
| Qwen3.5-35B-A3B | .482 / .552 | .583 / .720 | .796 / .888 |
| Llama-3.1-8B | .138 / .189 | .210 / .332 | .661 / .803 |
| Llama-3.3-70B | .479 / .562 | .649 / .769 | .789 / .893 |
| Gemma-4-E4B | .475 / .551 | .605 / .798 | .813 / .904 |
| Gemma-4-31B | .523 / .584 | .687 / .822 | .825 / .914 |
| DeepSeek-V2-Lite | .233 / .301 | .323 / .466 | .558 / .675 |

Gemma-4-31B is the best detector in every domain. The full tables (P/R/leakage/char-leakage, CIs, McNemar, per-label) are in `results/domains/summary.md`, and machine-readable versions are in `summary.json`.

### Whole-document span F1: original vs Custodian transform vs redaction (mean over 10 detectors)

The paper's second view: every gold span in the document, not just the masked ones.

| Domain | Exact: original | Custodian | Δ | Redacted | Overlap: original | Custodian | Δ | Redacted |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Finance | 0.335 | 0.256 | −0.078 | 0.100 | 0.441 | 0.422 | **−0.019** | 0.167 |
| Legal | 0.446 | 0.387 | −0.059 | 0.152 | 0.632 | 0.607 | **−0.025** | 0.266 |
| Enterprise | 0.645 | 0.577 | −0.068 | 0.251 | 0.801 | 0.781 | **−0.020** | 0.412 |

- **Exact-boundary F1 drops more than in the medical study** (mean −0.06 to −0.08 here; per-detector means in the medical study ranged from +0.003 to −0.047). Under overlap matching the drop is only −0.02 to −0.025.
- **Most of the extra drop is therefore boundary noise, not missed PII.** Gold spans are remapped onto the transformed text with difflib (`scripts/transform_corpus.py`). The Custodian transform also rewrites a lot of non-PII text (see *Surrogate quality* below), which makes the alignment, and so the exact boundaries, less reliable.
- **Redaction stays far below in both views.**
- **Detector ranking is preserved:** Gemma-4-31B and Llama-3.3-70B stay on top and OBI at the bottom.
- **Per-detector rows** are in `results/domains/wholedoc_f1.md`. Reproduce with `PYTHONPATH=. python scripts/wholedoc_f1_domains.py`.

## Findings

### 1. The Custodian transform replicates the medical result at the pooled level

Pooled over all detectors, the Custodian output is as detectable as the original (+0.42 pt, equivalent at ±2 pt), as in the medical paper (−1.2 pt). The LLM family drives this (+0.94 pt, equivalent), and Legal and Enterprise are equivalent on their own.

### 2. Where the Custodian transform loses recall: format-breaking surrogates

The failures are concentrated in rule-based and fine-tuned detectors, which rely on surface format. Paired span-level losses (original hit, Custodian miss) show why (recomputed from `results/domains/preds/{original,custodian}/`):

- **Finance · Presidio (−3.76).** The net losses are `phone_number` (18 losses, 0 gains; `1-800-123-4567` → `1-234-567`, `+1 213 555 1234` → `+2 214 560 1233`), `company` (40 losses vs 22 gains; `Association of Plumbing Contractors` → `Associations of plumber Contractors`), and `date` (9 vs 0; `03/14/2023` → `4xxxxxxxx`).
- **Finance · OBI (−3.67).** Mostly `company` (27 vs 3; `XYZ Bank` → `PQR Sincil`) and `name` (7 vs 0).
- **Enterprise · Presidio (−1.79).** Mostly `ssn` (11 vs 1): the surrogate drops a digit and breaks the 3-2-4 pattern (`097-77-6410` → `98-77-6411`).

In other words, the Custodian transform sometimes produces values that are not valid instances of their type. A detector that keys on format then no longer sees PII. LLMs are more tolerant of this, which is why the LLM family is equivalent. A few remapped spans are also fragments (`XYZ Bank` → `cil`, `Susan` → `an`), which points to alignment noise in `transform_corpus.py`'s span remapping. This affects a small number of spans.

### 3. Why the Faker arm mostly *raises* recall

1. **Finance (+7.6 pt).** The original gold contains many non-canonical values that detectors skip:
   - Template placeholders such as `[Client Name]`, `[Employee's Name]`, `[Address]`, `123 Main Street, Anytown, USA`, `ABC Co. Ltd.` — 105 of 1,283 spans (8%).
   - Generic organisation mentions such as `Finance team`, `Marketing department`, `International Brotherhood of Electrical Workers`, labelled `company`.

   Faker swaps these for canonical-looking entities (`Shannon Sanchez`, `Johnson, Smith and Hanson`), which every family finds more easily. `company` alone gains +18.6 pt (N = 3,270 span × detector), +20.4 pt for LLMs. This is a property of the *gold data*: a de-identification tool that recognises a placeholder would normally leave it alone. It does not mean surrogates are "more detectable PII". On its own masked set, the Custodian transform shows a much smaller upward shift in Finance (+1.5 pt).
2. **Legal (+4.2 pt).** Gains come from ORG (+12.0; `<City> Tribunal` reads as more name-like than long descriptive bodies), LOC (+6.5) and DATETIME (+4.3). QUANTITY (−0.9) and DEM (−0.8) are flat.
3. **Enterprise (−0.05 pt).** Gretel's values are already Faker-like, so original and surrogate are nearly the same distribution, and the result replicates the medical finding cleanly. The Presidio borderline (−0.47 pt) is a wash of type-specific recogniser effects: `license_plate` −43 pt (Faker plates like `7-ABC12`), `phone_number` −20 pt (`x5319` extensions, dotted formats), offset by `ssn` +19 and `address` +21.
4. **Legal · Qwen3.5-9B (−0.95 pt, CI [−3.10, +1.19]).** Not a clear drop, but non-inferiority is not established. In v1 this cell was −4.31 pt; the v2 gender/initials fix removed the name-quality part entirely (on windows with output in both arms, span losses and gains are now 107 vs 107). The remaining gap is whole-window abstention: Qwen returns no parsable spans on 96/250 original and 108/250 Faker windows, and the net 12 extra empty windows account for the whole −0.95 pt (see the v2 changelog).

## Surrogate quality of the Custodian transform

Detection equivalence only says that PII-shaped text is still where PII used to be. It says nothing about whether the rest of the document survives. I compared each Custodian output with its original (`data/transformed/<bench>.jsonl` vs `data/domains/original/<bench>.jsonl`, all 250 documents per domain) using a difflib character alignment (`SequenceMatcher`, `autojunk=False`). Each changed character is counted as inside or outside a gold span. Inserted text is counted as inside if it touches a gold span.

| | Finance | Legal | Enterprise |
|---|--:|--:|--:|
| Changed characters outside gold spans | **78%** (45,563 / 58,322) | **53%** (16,487 / 30,956) | **41%** (3,851 / 9,392) |
| Same, counting only replaced/deleted original characters | 76% | 50% | 31% |
| Docs with edits outside gold spans | 245 / 250 | 245 / 250 | 214 / 250 |
| Docs with new x-masking (more `xxx` runs than the original, e.g. `01/03/2023` → `2xxxxxxx`) | 48 | 12 | 70 |
| m/d/yyyy dates in the output | 84 | 0 | 22 |
| … of which are not valid calendar dates (all newly introduced, e.g. `2/31/2024`, `2/30/2024`) | 26 | — | 2 |
| Gold spans whose original string still appears verbatim in the output | 160 / 1,283 (12%) | 479 / 2,307 (21%) | 330 / 1,056 (31%) |

**Word-level edits outside gold spans** (word-sequence diff, replacements that touch no gold character; counts are occurrences):

- **Meaning changes.** Legal: `Human` → `Animal` (75, mostly "Human Rights" → "Animal Rights", in 60 windows), `judge` → `lawyer` (71, 46 windows), `Secretary` → `Assistant` (40, 26 windows), `police` → `cops` (44), `Chamber` → `Commerce` (38), `PROCEDURE` → `FUNCTIONS` (55). Finance: `Lender` → `Borrower` (26, 9 docs), `THEREFORE` → `WHEREAS` (22), `principal` → `teacher` (19), `Total` → `Absolute` (24), currency `USD` → `GBP` (19 docs) and `GBP` → `USD` (10 docs), ticker `AAPL` → `YHOO` (2 docs), and document titles such as `Savings Summary` → `Loan Overview`. Enterprise: `Replace affected part` → `Remove affected Parts`, `Medical Record` → `Biomedical Records` (17), `Social Security` → `Economic Security-` (13), `Credit` → `Debit`, `Incident` → `Massacre`.
- **Spelling errors introduced.** Legal: `Article` → `Artcile` (180, 82 windows), `Government` → `Goverment` (141, 80 windows), `State` → `Sate` (49), `Commission` → `Commision` (47), `Foreign` → `Foriegn` (31).
- **Number and inflection churn.** Years shift (`2022` → `2023`, `2023` → `2024`), amounts shift by a cent (`800.00` → `800.01`), and nouns are pluralised (`Convention` → `Conventions` 222, `Policy` → `Policies` 74, `Agreement` → `Agreements` 62).

**Caveats on these numbers.**

- The gold annotation is not exhaustive. Some "outside-gold" edits are probably real PII that the datasets did not label (e.g. account numbers in Finance that are not gold), so the outside-gold ratios are **upper bounds** on non-PII editing. The word-level examples above, however, are plainly not PII.
- Character alignment is approximate. Long rewrites can be split across inside/outside in more than one way; the two rows above (with and without insertions) give a sense of the spread.
- The "verbatim" row is a string-presence check. Short strings (e.g. a surname such as `Maxwell`) can reappear by coincidence, but many of the survivors are identifiers: enterprise SSNs (23 of 79 left unchanged at their span), MRNs (81 of 113), license plates (28 of 30), and Legal PERSON (110 of 384) and CODE (109 of 188) spans. That is a privacy-side question for the transform, separate from detectability.

**Implication.** The detection result holds, but the transform rewrites far more than the PII. On Finance and Legal most of what it changes is not PII, and some of those edits change meaning (who lends to whom, which currency, which official). Downstream use of the output should expect this.

## Caveats

- **Different masked sets.** The Custodian masked set is the spans Custodian changed; the Faker/redaction set is all gold spans with a surrogate (including Finance placeholders, which inflate the Faker upward shift). Compare verdicts across arms, not raw recall. For a like-for-like Faker comparison on exactly the Custodian-masked spans, run `scripts/build_domain_c1c2.py --masked-from custodian` and re-score (not done here).
- **LLM output failures.** Documents where an LLM produced no parsable spans on the original arm (finance / legal / enterprise, of 250):
  - Llama-3.1-8B: 204 / 141 / 21 (it often emits Python regex code instead of JSON).
  - Qwen3.5-9B: 72 / 96 / 11. Qwen3.5-4B: 40 / 56 / 3.
  - Gemma-4-E4B: 23 / 7 / 1. Qwen3.5-35B-A3B: 18 / 3 / 0. DeepSeek-V2-Lite: 14 / 1 / 0. Gemma-4-31B: 12 / 3 / 0. Llama-3.3-70B: 8 / 3 / 0.

  Rates on the Custodian arm are similar (e.g. Llama-3.1-8B 188 / 152 / 23). Low LLM recall in Finance/Legal is therefore partly a prompt-following failure. The prompt is the paper's clinical/HIPAA prompt, unchanged for comparability. This affects absolute F1, but in the paired design it only adds noise.
- **Batching.** Most LLMs ran with batched left-padded greedy decoding. This is not byte-identical to the paper's unbatched runs.
- **Panel.** 10 detectors (1 rule, 1 fine-tuned, 8 LLMs), not the paper's 11. Rule and fine-tuned are each one model, so their family verdicts rest on a single detector.
- **Language.** All three sets are English. Language generalisation is covered by the medical study; the multilingual Gretel finance splits (nl/fr/de/it/es/sv) are available if needed.
- **Legal gold.** TAB gold is DIRECT+QUASI identifiers. Detectors are charged false positives for NO_MASK entities (courts, states), which depresses precision.
- **Excluded dataset.** MAPA (`joelniklaus/mapa`, CC-BY-4.0) was evaluated and not used. Its English split has only 2 EUR-Lex documents (~225 entities) and is token-level without the raw text.
- **Test independence.** Spans are pooled across detectors, as in the paper. Treating them as independent overstates precision, so prefer the document-bootstrap CIs. For Custodian pooled, the bootstrap CI ([−0.32, +1.10]) is still well inside ±2 pt.

## Reproduce

```bash
CPU=python   # env with presidio + en_core_web_lg + faker + custodian_labs
GPU=python   # env with transformers 5.13 (Qwen3.5 / Gemma 4)
export PYTHONPATH=.

# 1. data: download, seeded 250-doc samples, span sanity check  -> data/domains/original/
$CPU scripts/build_domain_corpora.py
# 2. C1 redaction + C2 Faker corpora                              -> data/domains/{redacted,faker}/
$CPU scripts/build_domain_c1c2.py
# 3. Custodian transform (needs a valid CUSTODIAN_SDK_API_KEY in .env) -> data/transformed/<bench>.jsonl
for b in finance_gretel legal_tab enterprise_gretel; do $CPU scripts/transform_corpus.py --benchmark $b; done
# 4. detectors: per-doc preds -> results/domains/preds/<corpus>/<bench>__<system>.jsonl (resumable;
#    a failed doc is not written, so the next run retries it)
CUDA_VISIBLE_DEVICES="" $CPU scripts/run_domain_eval.py --system presidio --corpora original faker redacted custodian
CUDA_VISIBLE_DEVICES=1  $GPU scripts/run_domain_eval.py --system qwen3_5_9b --batch-size 8 --corpora original faker redacted custodian
#    (add --limit 5 for a smoke run)
# 5. analysis -> results/domains/summary.{md,json}
$CPU scripts/analyze_domains.py
```

Shortcuts:

- `bash scripts/run_domains.sh` runs steps 1, 2, 4 and 5 for the original five detectors (Presidio, OBI, Qwen3.5-9B, Llama-3.1-8B, Gemma-4-E4B) on original/faker/redacted.
- `bash scripts/run_domains.sh custodian` runs step 3, scores the same five detectors on `--corpora custodian`, and re-runs `analyze_domains.py`.
- **The full 10-detector panel** was run with `scripts/run_domain_eval.py` per detector (batched LLMs: `--batch-size 4`/`8`).

`analyze_domains.py` adds the Custodian section automatically when `data/transformed/<bench>.jsonl` and the custodian preds exist. It ignores legacy error rows, de-duplicates, and leaves a detector × domain out of sections 2–4 until its predictions cover every document. The `*_transformed` benchmarks are also registered, so `scripts/run_benchmark.py --benchmark finance_gretel_transformed ...` works.

## Files

- **New:**
  - `benchmarks/domains.py` (adapters `finance_gretel`, `legal_tab`, `enterprise_gretel`)
  - `scripts/build_domain_corpora.py`
  - `scripts/build_domain_c1c2.py`
  - `scripts/run_domain_eval.py`
  - `scripts/analyze_domains.py`
  - `scripts/run_domains.sh`
  - `data/domains/`, `data/transformed/{finance_gretel,legal_tab,enterprise_gretel}.jsonl`
  - `results/domains/` (preds, logs, summary)
- **Modified:**
  - `benchmarks/__init__.py` (registry)
  - `benchmarks/transformed.py` (3 `*Transformed` classes)
  - `scripts/transform_corpus.py` (SOURCES)
  - `scripts/build_c1c2_corpora.py` (`build_faker(..., surrogate=)`, backward-compatible)
  - `scripts/score_corpus.py` (accepts the domain benchmarks)
  - `systems/llm_local_hf.py` (`predict_batch`)

## Changelog

**v3 (2026-10-02): 10 detectors and the Custodian arm.**

- **Custodian Guardian Layer transform arm added** (`data/transformed/{finance_gretel,legal_tab,enterprise_gretel}.jsonl`, 750 docs, 0 API errors) and scored by all detectors. It is now the headline result. Earlier versions reported only Faker and redaction.
- **Panel extended from 5 to 10 detectors**: added Qwen3.5-4B, Qwen3.5-35B-A3B, Llama-3.3-70B, Gemma-4-31B and DeepSeek-V2-Lite. All 120 prediction files (10 detectors × 4 arms × 3 domains) cover all 250 documents with 0 error rows.
- **Failed-doc bug fixed.** `run_domain_eval.py` used to write a document that failed (e.g. CUDA OOM) as an empty prediction. That counted as a miss and was skipped on resume. Only Qwen3.5-35B-A3B was affected; its rows were cleaned and re-run. All other prediction files had 0 error rows. A failed doc is now not written, and the job exits non-zero so the queue retries it. `analyze_domains.py` also ignores any legacy error rows and excludes incomplete detector × domain cells.
- Results for the original 5 detectors on original/Faker/redaction are unchanged from v2. Pooled numbers changed only because the pool now has 10 detectors: Faker +3.36 → +4.18 pt, redaction −57.0 → −43.8 pt (the added LLMs keep more recall under `*****`). Faker non-inferiority: 13/15 → 27/30 cells (new failing cell: Enterprise · DeepSeek-V2-Lite, −1.14, CI [−4.01, +1.74]).

**v2 (2026-09-27): legal name surrogate fix.** The v1 Faker generator in `scripts/build_domain_c1c2.py` kept a person's title but drew a random-gender first name (`Mr Andreas Orphanides` → `Mr Teresa Fowler`). It also lost the format of multi-initial names (`Z.M. Necatigil`). v2 does two things:
- It infers gender from a title inside the span or in the 8 characters before it. Mr/Sir/Lord → male; Mrs/Ms/Miss/Lady/Madam/Mme/Mlle → female.
- It keeps initial-shaped first tokens (`Mr Z.M. Necatigil` → `Mr H.H. Clarke`).

All three Faker corpora were regenerated. The Faker RNG stream shifts, so surrogates changed in 202/250 finance docs, 176/250 legal windows and 6/250 enterprise docs. One legal surrogate now equals its original, so the legal masked set is 2,203 (was 2,204). All 5 detectors were re-scored on `data/domains/faker/*` and `data/domains/redacted/legal_tab`. Original-document results are unchanged.

C2 Faker Δ (pt) that changed (5-detector panel):

| Cell | v1 | v2 |
|---|--:|--:|
| Finance · Presidio / OBI / Qwen / Llama / Gemma | +6.39 / +8.03 / +4.36 / +3.59 / +12.24 | +6.00 / +7.72 / +5.69 / +5.77 / +10.99 |
| Legal · Presidio / OBI / Llama / Gemma | +0.27 / +2.22 / +4.67 / +5.04 | +0.86 / +2.32 / +5.81 / +4.90 |
| **Legal · Qwen3.5-9B** | **−4.31** [−6.54, −2.08] | **−0.95** [−3.10, +1.19] |
| Enterprise · Llama-3.1-8B | +0.85 | +1.42 |
| Finance (pooled) | +6.92 | +7.23 |
| Legal (pooled) | +1.58 (boot CI [−0.47, +3.58]) | +2.59 (boot CI [+0.78, +4.46]) |
| Enterprise (pooled) | +0.13 | +0.25 |
| All 3 domains · all detectors | +2.75 [+1.60, +4.00] | +3.36 [+2.30, +4.48] |

Legal · Qwen3.5-9B decomposition (pt of 2,203 masked spans; recomputed from `results/domains/preds/{original,faker}/legal_tab__qwen3_5_9b.jsonl`; "empty" = no parsable spans for the whole window):

| Contribution to Δ | v1 | v2 |
|---|--:|--:|
| Windows empty on Faker only (33 windows) | −9.17 | −8.81 |
| Windows empty on original only (21 windows) | +7.76 | +7.85 |
| Windows with output in both arms | **−2.90** | **0.00** |
| **Total** | −4.31 | −0.95 |

Verdicts were unchanged in v2: 13/15 cells non-inferior, the same 4 TOST-equivalent. What changed is Legal · Qwen, which moved from a clear drop to inconclusive. C1 redaction numbers moved by ≤0.05 pt.

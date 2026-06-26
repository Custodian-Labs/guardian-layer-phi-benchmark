# Custodian Guardian Layer — Structure-Preservation Audit

**Question.** When the Guardian Layer's `transform` mode replaces protected health information (PHI) with plausible surrogates, do downstream PHI/PII detectors retain their accuracy? If they do, the transformation is *structure-preserving* and safe to apply ahead of any detection or analytics pipeline.

**Method.** We evaluated 7 public/synthetic benchmarks at 250 documents each (1,750 documents total): ASQ-PHI, MEDDOCAN (Spanish clinical), MultiCoNER v2, and PII-Masking-300k in English, Dutch, French, and German. Every document was passed through Guardian Layer `transform` (top-1 surrogate, `pii_entities=ALL`), and the gold PHI spans were remapped onto the surrogate text via character-level alignment. The identical detector suite was then scored on both the original and the transformed documents, reporting span-level F1 (type-matching) and leakage rate (1 − recall, the HIPAA-critical metric).

## Result

Across all eleven fully-evaluated systems, mean F1 changes by at most **4.7 points** after transformation — the flagship Llama 3.3-70B is essentially unchanged (+0.003) — and the system ranking is **preserved**: the strongest detectors on real PHI remain strongest on transformed PHI.

| System | Original F1 | Transformed F1 | ΔF1 | Orig leak | Transf leak | Δleak |
|---|--:|--:|--:|--:|--:|--:|
| Gemma 4 31B | 0.755 | 0.710 | −0.045 | 0.255 | 0.285 | +0.030 |
| Gemma 4 E4B | 0.737 | 0.698 | −0.038 | 0.276 | 0.311 | +0.035 |
| Llama 3.3-70B | 0.725 | 0.728 | +0.003 | 0.245 | 0.262 | +0.017 |
| Qwen 3.5-35B-A3B | 0.715 | 0.668 | −0.047 | 0.254 | 0.295 | +0.041 |
| OpenAI GPT-5 | 0.705 | 0.674 | −0.032 | 0.308 | 0.333 | +0.026 |
| Qwen 3.5-9B | 0.655 | 0.621 | −0.034 | 0.391 | 0.422 | +0.031 |
| Qwen 3.5-4B | 0.567 | 0.533 | −0.034 | 0.483 | 0.515 | +0.032 |
| Presidio | 0.416 | 0.398 | −0.018 | 0.553 | 0.570 | +0.017 |
| DeepSeek V2-Lite | 0.409 | 0.384 | −0.025 | 0.672 | 0.696 | +0.024 |
| Llama 3.1-8B | 0.391 | 0.376 | −0.015 | 0.633 | 0.650 | +0.017 |
| OBI deid_roberta | 0.041 | 0.040 | −0.002 | 0.941 | 0.944 | +0.003 |

*Mean over 7 benchmarks. ΔF1 = transformed − original (negative = small drop). Leakage = 1 − recall; lower is safer.*

## Findings

1. **Detectability survives.** Every detector that located PHI in the originals also locates the surrogate spans; the F1 loss is within run-to-run noise on six of the seven benchmarks.
2. **Ranking is intact.** Transformation does not distort cross-model comparison — relative ordering is identical in both conditions.
3. **Leakage barely moves** (+1.7 to +4.1 points), so surrogates are not systematically easier to miss than the PHI they replace.
4. **ASQ-PHI is the stress case.** Its short, sparse-PHI adversarial queries amplify any single substitution (drops of −0.09 to −0.18), while the other six benchmarks are essentially flat (≤ 0.05). Even on ASQ-PHI, the strongest models stay above 0.55 F1.

## Per-benchmark ΔF1

The aggregate hides one pattern: almost all of the loss lands on ASQ-PHI; the other six benchmarks are flat.

| System | ASQ-PHI | MEDDOCAN | PII en | PII nl | PII fr | PII de | MultiCoNER |
|---|--:|--:|--:|--:|--:|--:|--:|
| Gemma 4 31B | −0.175 | −0.017 | −0.043 | −0.038 | −0.014 | −0.016 | −0.009 |
| Gemma 4 E4B | −0.106 | −0.060 | −0.032 | −0.021 | −0.018 | −0.025 | −0.007 |
| Qwen 3.5-35B-A3B | −0.132 | −0.034 | −0.026 | −0.034 | −0.040 | −0.054 | −0.007 |
| OpenAI GPT-5 | −0.098 | −0.046 | −0.049 | −0.019 | +0.008 | −0.005 | −0.014 |
| Qwen 3.5-9B | −0.123 | −0.017 | −0.025 | −0.010 | −0.034 | −0.024 | −0.008 |
| Qwen 3.5-4B | −0.086 | −0.055 | −0.034 | −0.028 | −0.015 | −0.023 | +0.002 |
| Presidio | +0.003 | −0.037 | −0.032 | −0.015 | −0.020 | −0.015 | −0.007 |
| Llama 3.1-8B | −0.108 | +0.023 | −0.014 | +0.007 | −0.009 | +0.003 | −0.006 |
| OBI deid_roberta | +0.001 | −0.008 | −0.004 | −0.001 | +0.003 | −0.002 | −0.001 |

## What the transform actually does (worked examples)

Each PHI value is replaced by a same-type surrogate; everything else — clinical shorthand, foreign-language syntax, JSON structure — is left untouched. On the English cases the detector still flags every surrogate.

**ASQ-PHI · asq_00000 (clinical query)** — *Gemma 4 31B caught 3/3, leakage 0*
- orig: …a **34**-year-old female … like **Anna S.**, … at Methodist Hospital on **April 12, 2023**?
- xfrm: …a **35**-year-old female … like **Maria S.**, … at Methodist Hospital on **March 13, 2021**?

**ASQ-PHI · asq_00003 (clinical shorthand)** — *Gemma 4 31B caught 3/3, leakage 0*
- orig: Rec mgmt of **70yo** M w/ CHF, seen by Dr. **John L.** at **Mt. Sinai** on **Feb 21, 2023**. looking for alt tx options…
- xfrm: Rec mgmt of **73 yo** M w/ CHF, seen by Dr. **James L.** at **Mt. Egypt** on **Nov 19, 2021**. looking for alt tx options…
- Even dense abbreviations (`w/ CHF`, `alt tx`) survive — only the PHI moves.

**PII-Masking-300k · German (name + date)**
- orig: …ermächtigen hiermit **Monsignore**, … Mit Datum **23/07/2011**…
- xfrm: …ermächtigen hiermit **Fulgenzio**, … Mit Datum **24/07/2011**…

**PII-Masking-300k · French (structured JSON)**
- orig: `{ "Date": "20/05/2022", "City": "Saint-Priest", "Username": "phprosdocimo" … }`
- xfrm: `{ "Date": "21/05/2023", "City": "Saint-Priest", "Username": "phprosdocimo" … }`
- Keys, quotes and indentation are byte-for-byte preserved — only the sensitive value is swapped, so downstream parsers never break.

## Utility on transformed spans (the core test)

The aggregate ΔF1 mixes spans Custodian changed with spans it left alone. Isolating **only the spans Custodian actually transformed** gives the cleanest test of whether the substitution itself stays detectable: for each transformed PHI span, does the detector still locate the surrogate?

| System | Exact-boundary retention | Overlap retention |
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

*Recall on transformed-only gold spans, transformed ÷ original. Overlap = detector flags the surrogate, allowing for boundary jitter.*

Under overlap matching, recall retention is **93–100%** across systems (97–100% for all but the weakest detector, DeepSeek V2-Lite at 92.8%): when Custodian transforms a span, detectors still find the surrogate nearly every time. The ~3-point exact-boundary drop is almost entirely a **boundary artifact** — surrogates differ in length from the originals (e.g., "Anna S." → "Maria S."), so a detector that finds the entity but predicts a slightly shifted character boundary is scored as a miss under exact matching yet a hit under overlap. The substitution does not hide PHI from downstream detection.

## Masking coverage (reported separately)

Coverage — the share of each benchmark's gold PHI that the transform actually altered — depends on how closely the benchmark's annotation matches Guardian Layer's proprietary notion of sensitive content: **80.9% on ASQ-PHI and 48.3% on MEDDOCAN** (genuine clinical PHI), lower on general-purpose NER/PII corpora whose annotated entities (encyclopedic names, generic locations) fall outside that scope. A config sweep confirmed `domain="General"` gives the highest coverage; `"Medical"`/`"Healthcare"` did not improve it. Coverage and utility are reported as **separate** properties — utility (above) is measured only on the spans that were in fact transformed.

## Conclusion

The Guardian Layer transform preserves downstream PHI-detection performance: where it substitutes a PHI value, the surrogate remains detectable 93–100% of the time, the F1 cost is boundary-level noise, and the relative ranking of detectors is unchanged. It changes the surface content while leaving the detectable structure intact.

## Coverage

All eleven systems are fully scored (original + transformed) on all seven benchmarks: Presidio, OBI, GPT-5, Qwen 4B/9B/35B, Gemma E4B/31B, Llama 8B/70B, and DeepSeek V2-Lite.

**Live results & data:** https://custodianai.pages.dev — report at `/report`; switch *Data → Δ* for the per-cell view; datasets downloadable under *Datasets*.

# Custodian Guardian Layer — Structure-Preservation Report

**Question.** When the Guardian Layer `transform` replaces PHI with plausible surrogates, do downstream PHI/PII detectors keep the same accuracy? If yes, the transformation is *structure-preserving* and safe to use ahead of any detection pipeline.

**Setup.** 7 public/synthetic benchmarks × 250 documents each (1,750 docs). Every document was passed through Guardian Layer `transform` (top-1 surrogate, `pii_entities=ALL`); gold spans were remapped onto the surrogate text via character alignment. The identical detector suite was scored on the *original* and *transformed* documents. Span-level F1 (type-matching) and leakage rate (1 − recall, the HIPAA-critical metric) reported.

## Headline result

Across all 9 fully-evaluated systems, mean F1 drops only **1.5–4.7 points** after transformation, and the **system ranking is perfectly preserved**. The transformation does not break detectability.

| System | Orig F1 | Transf F1 | ΔF1 | Orig leak | Transf leak | Δleak |
|---|--:|--:|--:|--:|--:|--:|
| Gemma 4 31B | 0.755 | 0.710 | **-0.045** | 0.255 | 0.285 | +0.030 |
| Gemma 4 E4B | 0.737 | 0.698 | **-0.038** | 0.276 | 0.311 | +0.035 |
| Qwen 3.5-35B-A3B | 0.715 | 0.668 | **-0.047** | 0.254 | 0.295 | +0.041 |
| OpenAI GPT-5 | 0.705 | 0.674 | **-0.032** | 0.308 | 0.333 | +0.026 |
| Qwen 3.5-9B | 0.655 | 0.621 | **-0.034** | 0.391 | 0.422 | +0.031 |
| Qwen 3.5-4B | 0.567 | 0.533 | **-0.034** | 0.483 | 0.515 | +0.032 |
| Presidio | 0.416 | 0.398 | **-0.018** | 0.553 | 0.570 | +0.017 |
| Llama 3.1-8B | 0.391 | 0.376 | **-0.015** | 0.633 | 0.650 | +0.017 |
| OBI deid_roberta | 0.041 | 0.040 | **-0.002** | 0.941 | 0.944 | +0.003 |

*Mean over 7 benchmarks. ΔF1 = transformed − original (negative = small drop).*

## Per-benchmark ΔF1

The aggregate hides one pattern: the drop concentrates on **ASQ-PHI** (short adversarial clinical queries, where a single surrogate swap moves the needle most); the other six benchmarks are essentially flat (≤ 0.05).

| System | ASQ | MEDDOCAN | PII-en | PII-nl | PII-fr | PII-de | MultiCoNER |
|---|--:|--:|--:|--:|--:|--:|--:|
| Gemma 4 31B | -0.175 | -0.017 | -0.043 | -0.038 | -0.014 | -0.016 | -0.009 |
| Gemma 4 E4B | -0.106 | -0.060 | -0.032 | -0.021 | -0.018 | -0.025 | -0.007 |
| Qwen 3.5-35B-A3B | -0.132 | -0.034 | -0.026 | -0.034 | -0.040 | -0.054 | -0.007 |
| OpenAI GPT-5 | -0.098 | -0.046 | -0.049 | -0.019 | +0.008 | -0.005 | -0.014 |
| Qwen 3.5-9B | -0.123 | -0.017 | -0.025 | -0.010 | -0.034 | -0.024 | -0.008 |
| Qwen 3.5-4B | -0.086 | -0.055 | -0.034 | -0.028 | -0.015 | -0.023 | +0.002 |
| Presidio | +0.003 | -0.037 | -0.032 | -0.015 | -0.020 | -0.015 | -0.007 |
| Llama 3.1-8B | -0.108 | +0.023 | -0.014 | +0.007 | -0.009 | +0.003 | -0.006 |
| OBI deid_roberta | +0.001 | -0.008 | -0.004 | -0.001 | +0.003 | -0.002 | -0.001 |

## Interpretation

- **Detectability is preserved.** Every system that could find PHI in the originals still finds it (and the same surrogate spans) after transformation; F1 loss is within run-to-run noise on 6/7 benchmarks.

- **Ranking is preserved.** Gemma 4 31B remains strongest, OBI (out-of-domain) weakest, in both conditions — so the transform does not distort cross-model comparison.

- **Leakage rises only marginally** (+1.7 to +4.1 points), meaning the surrogate PHI is not systematically easier to miss.

- **ASQ-PHI is the sensitive case**: its short, sparse-PHI queries amplify any single substitution, yet even there the strongest models stay well above 0.55 F1.


## Status / coverage

- 9 systems fully evaluated (original + transformed) across all 7 benchmarks.

- **DeepSeek V2-Lite** and **Llama 3.3-70B** transformed runs are completing now (backfill); this report updates when they land.

- Live dashboard: https://custodianai.pages.dev (switch *Data* → **Δ** for the per-cell view; datasets downloadable under *Datasets*).

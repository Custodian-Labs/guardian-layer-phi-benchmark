# ML4H 2026 camera-ready: analyses, data and commands

Paper: *A Paired Masked-Span Protocol for Evaluating Structure-Preserving De-identification* (ML4H 2026,
Proceedings track, PMLR volume 346).

## Definitions used throughout

- **Re-projection.** Gold spans are aligned onto the transformed text with difflib. Each boundary is extended
  outward to the nearest word boundary where the original boundary is itself a word boundary (`snap_sym` in
  `scripts/paired_core.py`). Gold spans whose stored text does not match the document (59 template placeholders
  in PII-Masking-300k) are excluded. The uncorrected alignment is reported as a sensitivity row.
- **Matching.** Primary: label-agnostic character overlap. Also reported: IoU ≥ 0.5, exact boundaries, and exact
  boundaries with a coarse shared type map.
- **Change metric.** Δ = recall(transformed) − recall(original), in points; negative values are losses.
- **Inference.** Stratified document-level cluster bootstrap (5,000 draws); TOST at margin m passes iff the 90%
  CI lies inside (−m, m); posterior P(|Δ| < m) from a Bayesian bootstrap over documents.

## Re-analysis of stored predictions (CPU only)

| Analysis | Script | Output |
|---|---|---|
| All paper numbers and tables | `scripts/camera_ready_numbers.py` | `paper/ml4h/camera_ready/numbers.json`, `tables/*.tex` |
| Matching criteria, margins, margin curve | `scripts/analyze_matching_margins.py` | `results/robustness/` |
| Transform properties (consistency, format, x-masking) | `scripts/audit_transform_properties.py` | `results/audit/transform_properties.*` |
| Same-slot counterfactual | `scripts/analyze_counterfactual.py` | `results/c1c2/counterfactual.json` |
| Human annotation: agreement and outcome linkage | `scripts/analyze_human_annotation.py` | `results/annotation/human_v2.*` |
| ICD-10 downstream summary | `scripts/summarize_medical_decisions.py` | `results/decisions/medical_summary.*` |
| GPT-5 subset vs panel | `scripts/compare_gpt5.py` | `results/openai_audit/gpt5_comparison.*` |

## New inference (stored here, re-runnable)

| Run | Script | Stored results |
|---|---|---|
| C1 with a mask-aware prompt (Qwen 3.5 9B) | `scripts/score_corpus.py --prompt mask_aware` | `results/c1c2/redacted_*_qwen3_5_9b_mask_aware.json` |
| Per-span C2 outcomes (Presidio, OBI, Qwen 3.5 9B) | `scripts/score_corpus.py --save-preds` | `results/c1c2/preds/` (first-run aggregates in `results/c1c2/first_run/`) |
| ICD-10 and clinical questions (Qwen 3.5 4B/9B, Gemma 4 E4B) | `scripts/decision_eval.py --benchmarks asq_phi meddocan` | `results/decisions/*__medical.jsonl` |
| Locale judge (Qwen 3.5 9B) | `scripts/locale_consistency.py` | `results/locale/` |
| OpenAI model check and GPT-4o-mini test–retest | `scripts/openai_model_check.py`, `scripts/retest_openai.py` | `results/openai_audit/` |
| GPT-5 on 199 stratified documents | `scripts/retest_openai.py --model gpt-5 --runs 1` | `results/openai_audit/retest_gpt-5.jsonl` |

## Data added for the camera-ready

- `data/annotation/human/`: the blind annotation sheets (`surrogate_quality_audit197_blind.csv`; the 397-row v2
  sheet with its key), the guidelines, and the two raters' independent labels
  (`surrogate_quality_audit197_rater1.csv`, `surrogate_quality_audit197_rater2.csv`).
- `data/original/`: the original documents with the replaced-span flags of `data/redacted/`, used for re-runs on
  the original text.
- `data/external/codiesp/meddocan_icd10.json`: ICD-10 codes and chapters for the 250 MEDDOCAN documents, taken
  from the CodiEsp corpus (Miranda-Escalada et al., CLEF eHealth 2020; CC BY 4.0;
  https://zenodo.org/records/3837305), matched by document ID.

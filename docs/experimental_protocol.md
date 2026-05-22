# Experimental Protocol — CustodianAI PHI Benchmark

This document specifies how the benchmark was run, so that another team can
reproduce the numbers exactly. Written in the style expected by a typical
clinical NLP venue (JAMIA, AMIA, Bioinformatics).

## 1. Research questions

**RQ1.** Across one Spanish clinical PHI benchmark (MEDDOCAN), one English
adversarial clinical-query benchmark (ASQ-PHI), and one English general-PII
benchmark (PII-Masking-300k), how do specialized de-identification systems
(Microsoft Presidio, OBI `deid_roberta_i2b2`) compare against frontier LLMs
prompted as PHI detectors?

**RQ2.** Within the LLM-as-detector class, can a small *local* open-source
model (e.g. Gemma 4 E4B, 8 B params) match the strongest hosted model
(GPT-5) at this task, and at what cost / latency?

**RQ3.** Does Custodian Guardian Layer (the system under test) preserve
downstream-task utility after PHI transformation? (Deferred to phase 2 once
the Custodian SDK credentials are wired and the n2c2 / MIMIC-IV-Note DUAs
arrive; the current dashboard establishes the detection-side baselines.)

## 2. Datasets

All three datasets in this protocol are *synthetic and open-licensed*; no
DUA is required.

| Dataset | Lang | Used split | Sample size | License |
| --- | --- | --- | --- | --- |
| MEDDOCAN | es | `test` | 25 docs (sorted-order first-N) | CC, Zenodo |
| ASQ-PHI | en | full corpus | 25 queries (sorted-order first-N) | MIT, Mendeley |
| PII-Masking-300k | en + multi | `validation` filtered to `language=English` | 25 docs (deterministic shuffle, seed=0) | "other" (HF page) |

The 25-doc subset is a deliberate compromise: large enough to surface F1
gaps of ≥0.05 between systems (the smallest meaningful difference here),
small enough that one LLM run completes in under 25 minutes per system.

## 3. Systems under test

### Specialized de-identification

| System | Version | Setup |
| --- | --- | --- |
| Microsoft Presidio | analyzer + anonymizer 2.2 | spaCy `en_core_web_lg` 3.7.1 / `es_core_news_lg` 3.7.0 |
| OBI `deid_roberta_i2b2` | HuggingFace `obi/deid_roberta_i2b2` | inference on A100, bf16 |

### Hosted LLMs

| System | Model ID | Sampling |
| --- | --- | --- |
| OpenAI GPT-5 | `gpt-5` (default tier) | temperature 0, `response_format={"type": "json_object"}` |

### Local LLMs (HuggingFace transformers)

| System | Model ID | Params | Inference |
| --- | --- | --- | --- |
| Gemma 4 E4B-it | `google/gemma-4-E4B-it` | 8 B | bf16, transformers 5.5.4, A100 80 GB |
| Qwen 3.5-4B | `Qwen/Qwen3.5-4B` (dense, `enable_thinking=False`) | 4 B | bf16, transformers 5.5.4, A100 80 GB |
| DeepSeek V2-Lite-Chat | `deepseek-ai/DeepSeek-V2-Lite-Chat` | 16 B MoE / 2.4 B active | bf16, transformers 4.57.6 + custom DynamicCache shim, A100 80 GB |

All LLMs received the same English system prompt (see
`systems/llm_openai_compatible.py:SYSTEM_PROMPT`) and a user message
containing the document text. JSON output is parsed by a robust regex
extractor that recovers complete span objects even when generation is cut
off by `max_new_tokens=1000`.

### LLM-as-detector offset recovery

LLMs reliably identify the *correct PHI substrings* but unreliably emit
character offsets. We therefore use the LLM's offsets only as a *hint*
and recover the canonical offset by searching the predicted text in the
source document. When a value occurs multiple times, we pick the
occurrence closest to the hint and not already claimed. The full
implementation is in
[`systems/llm_openai_compatible.py::_parse_spans`](../systems/llm_openai_compatible.py).

### Systems attempted but blocked

- Moonshot Kimi-VL-A3B-Thinking-2506 — custom modeling imports
  `PytorchGELUTanh` (removed pre-4.58) and `is_torch_fx_available`
  (removed pre-5.0); cannot coexist in any single transformers version.
- Moonshot Moonlight-16B-A3B-Instruct — attention-mask shape off-by-one
  vs. modern transformers broadcasting; bug is in the upstream modeling
  code.
- John Snow Labs Spark NLP for Healthcare, Philter (UCSF) — stub
  wrappers in `systems/`; both require non-trivial setup (commercial
  license / Spark cluster, or UCSF repo clone). Not run for this round.
- Custodian Guardian Layer — wrapper is in
  [`systems/custodian.py`](../systems/custodian.py); requires
  `CUSTODIAN_SDK_API_KEY`. Not run for this round.

## 4. Evaluation metrics

For each (system, document) pair we compute span-level true positives,
false positives, and false negatives under three matching modes:

| Mode | Definition |
| --- | --- |
| `strict` | predicted (start, end) and label both match gold |
| `type` | predicted (start, end) match gold (label ignored) |
| `relaxed` | predicted span overlaps gold span by ≥1 character (label ignored) |

We report per-system Precision, Recall, F1, **leakage rate**
(`1 − recall`, the regulator-facing metric — every missed PHI span is a
HIPAA breach regardless of how clean the rest of the output is), and
**character-weighted leakage** (gold-PHI characters not masked / gold-PHI
characters total, which weighs long spans more heavily than short ones).

The default mode on the dashboard is `type`. We treat label disagreement
as a labelling-scheme difference rather than a substantive error, which
is the convention in cross-system de-identification comparisons.

## 5. Reproducibility

```bash
# 0. Pin a fresh environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg
python -m spacy download es_core_news_lg

# 1. Bring in the data
#    MEDDOCAN:
wget -O data/meddocan/raw/meddocan.zip \
     'https://zenodo.org/api/records/4279323/files/meddocan.zip/content' && \
unzip data/meddocan/raw/meddocan.zip -d data/meddocan/raw/

#    ASQ-PHI: requires a Mendeley OAuth Bearer (issued anonymously by
#    public-api) → see scripts/import_asq_phi.py for the format expected
#    once the source file is on disk.

#    PII-Masking-300k: streamed automatically by datasets library.

# 2. Run a system on a benchmark
python scripts/run_benchmark.py --benchmark meddocan --split test \
    --systems openai presidio obi --mode type --include-text --limit 25 \
    --out results/run.jsonl

# 3. Merge multi-run results
python scripts/merge_runs.py --out results/meddocan_25 --benchmark meddocan \
    results/run_openai.jsonl results/run_presidio.jsonl ...

# 4. Publish to dashboard
python scripts/publish_results.py
```

Random seeds: the runner is deterministic given the same sorted-order
document iteration; we use `temperature=0` for all LLMs. Where a dataset
sub-samples (PII-Masking-300k), the loader is parameterised by
`--seed` (default 0).

## 6. Threats to validity

- **Sample size.** 25 documents per (benchmark, system) is enough to rank
  systems whose F1 differs by ≥0.05 but not to discriminate between
  e.g. GPT-5 and Gemma 4 E4B on MEDDOCAN (their F1s are within 0.03).
  Phase-2 results will scale to 250 documents per benchmark once we have
  budget signoff for GPT-5 calls and longer GPU bookings.
- **Label-set mismatch.** OBI `deid_roberta_i2b2` predicts a *clinical*
  label set (PATIENT, DOCTOR, HOSP, ID, DATE, …). On PII-Masking-300k
  this set hardly overlaps with the gold labels (USERNAME, EMAIL,
  ADDRESS, …) even under our label-agnostic `type` mode, because the
  span boundaries also diverge. We report this result as evidence of
  domain transfer failure, not as a verdict on OBI's capability inside
  clinical text.
- **Cross-lingual transfer.** OBI is English-trained; running it on
  Spanish MEDDOCAN measures how badly cross-lingual transfer fails for
  a token-classification model with no Spanish pre-training. Including
  this number is intentional — it is a useful contrast for Spanish-NER
  baselines (Presidio with `es_core_news_lg`).
- **LLM prompt.** All LLMs received an identical prompt. We have not
  per-tuned the prompt per model; doing so would likely lift each by
  3–8 F1 but conflate model capability with prompt-engineering effort.
- **Single-shot LLM calls.** No self-consistency, ensembling, or
  retrieval. Numbers reported are direct one-shot outputs.

## 7. Files of record

| Artefact | Path |
| --- | --- |
| Aggregate JSON used by dashboard | `web/data/<run>.summary.json` |
| Per-document predictions | `results/<run>.jsonl` |
| Dataset descriptors | `web/data/datasets_meta.json` |
| Per-run dashboard manifest | `web/data/index.json` |
| Source for offsets recovery | `systems/llm_openai_compatible.py` |
| Source for local-HF inference | `systems/llm_local_hf.py` |
| GitHub Pages dashboard | `https://14h034160212.github.io/Custodianai/` |

## 8. Roadmap

- Phase 2: add n2c2 2014 De-ID + Heart Disease Risk Factors (DUA pending).
- Phase 2: add MIMIC-IV-Note for real-EHR scale check (PhysioNet DUA).
- Phase 2: hook up Custodian Guardian Layer (4 variants: MASKED × redact,
  MASKED × transform, PROPRIETARY × redact, PROPRIETARY × transform).
- Phase 3: utility-preservation experiment — downstream classifier (e.g.
  smoking status on n2c2 2006) trained on raw text and re-evaluated on
  Guardian-Layer transformed text; report Δaccuracy as the
  utility-preservation score.

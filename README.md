# CustodianAI · PHI/PII Benchmark Harness

End-to-end evaluation harness for measuring how well **Custodian Labs' Guardian
Layer** transforms clinical text, compared against:

- **Specialized de-id systems**: Microsoft Presidio, Philter (UCSF),
  John Snow Labs Spark NLP for Healthcare, OBI `deid_roberta_i2b2`.
- **Frontier LLMs** prompted as PHI detectors: OpenAI GPT-5, DeepSeek V4,
  Moonshot Kimi K2.6, Alibaba Qwen 3.7, Google Gemma 4.

## Live dashboard

**Public URL (auto-deployed from `web/` via GitHub Actions):**
https://14h034160212.github.io/Custodianai/

**Backup URL (works without Pages enabled, slower):**
https://raw.githack.com/14H034160212/Custodianai/main/web/index.html

## Current numbers (2026-05-22)

Three datasets, 25 documents each, type-mode (span-exact, label-agnostic).

### MEDDOCAN test, Spanish clinical synthetic

| System | P | R | **F1** | Leakage |
| --- | ---: | ---: | ---: | ---: |
| **OpenAI GPT-5** | 0.82 | 0.63 | **0.71** | 0.37 |
| **Gemma 4 E4B** (8B, local) | 0.76 | 0.62 | **0.68** | 0.38 |
| **Qwen 3.5-4B** (local, no thinking) | 0.62 | 0.43 | **0.51** | 0.57 |
| **Presidio** (es spaCy `lg`) | 0.47 | 0.44 | **0.45** | 0.57 |
| **DeepSeek V2-Lite** (16B MoE, local) | 0.54 | 0.30 | **0.39** | 0.70 |
| **OBI `deid_roberta_i2b2`** (English) | 0.07 | 0.07 | **0.07** | 0.93 |

### ASQ-PHI, English adversarial clinical queries

| System | P | R | **F1** | Leakage |
| --- | ---: | ---: | ---: | ---: |
| **OpenAI GPT-5** | 0.69 | 0.86 | **0.76** | 0.14 |
| **Presidio** (en spaCy `lg`) | 0.42 | 0.69 | **0.52** | 0.31 |
| **OBI `deid_roberta_i2b2`** | 0.03 | 0.08 | **0.05** | 0.92 |

Note: Presidio's recall (0.69) is much higher on ASQ-PHI than on MEDDOCAN
(0.41) or PII-Masking-300k (0.35), because ASQ-PHI's NAME / DATE /
GEOGRAPHIC_LOCATION labels match Presidio's English recognizers exactly.

### PII-Masking-300k validation (English), general PII

| System | P | R | **F1** | Leakage |
| --- | ---: | ---: | ---: | ---: |
| **Gemma 4 E4B** (8B, local) | 0.88 | 0.81 | **0.84** | 0.19 |
| **OpenAI GPT-5** | 0.85 | 0.74 | **0.79** | 0.26 |
| **Qwen 3.5-4B** (local) | 0.92 | 0.64 | **0.75** | 0.36 |
| **Presidio** (en) | 0.30 | 0.35 | **0.32** | 0.65 |
| **DeepSeek V2-Lite** (local) | 0.56 | 0.22 | **0.31** | 0.78 |
| **OBI `deid_roberta_i2b2`** | 0.02 | 0.04 | **0.02** | 0.96 |

### Headlines

- **A small *local* 8B model beats GPT-5 on general PII.** Gemma 4 E4B
  reaches F1=0.84 on PII-Masking-300k, ~5 F1 above GPT-5. First task
  where open-source ≥ hosted in our matrix.
- **GPT-5 still leads on clinical PHI** — F1=0.71 (Spanish, MEDDOCAN)
  and 0.76 (English, ASQ-PHI). Gemma trails by 3-5 F1 on clinical
  text; the gap is real but small.
- **Non-LLM baselines lag by 20+ F1 points** on every benchmark.
  Presidio is competitive only on tasks where its built-in
  recognizers (English NAME/DATE/EMAIL) match the gold labels —
  evident in the ASQ-PHI score jumping to F1=0.52.
- **Domain-specific transformers do not transfer.** OBI
  `deid_roberta_i2b2` predicts clinical PHI labels (PATIENT, DOCTOR,
  HOSP, …); on general PII its F1 collapses to 0.02 even though the
  input is English. Cross-domain failure is the rule, not the
  exception, for token-classification de-id models.

### Models attempted but blocked

- **Moonshot Kimi-VL-A3B-Thinking-2506** — custom modeling code imports
  symbols absent from both transformers 4.57 and 5.5; requires an
  intermediate transformers version to load.
- **Moonshot Moonlight-16B-A3B-Instruct** — attention-mask shape
  off-by-one with our transformers; modeling code expects an older mask
  broadcasting convention.
- For "Kimi family" coverage, we used **Qwen 3.5-4B** as an Alibaba
  substitute (similar geographic / vendor profile to Moonshot, fully
  compatible with our transformers).

## Project layout

```
custodianai/
├── benchmarks/        # Dataset loaders → Document(text, gold_spans)
│   ├── base.py
│   ├── asq_phi.py            # English clinical-queries synthetic, no DUA
│   ├── meddocan.py           # Spanish clinical synthetic, no DUA
│   └── pii_masking_300k.py   # Multilingual general PII (HF, open)
├── systems/           # Each model wraps a DeIDSystem.predict(text) → Prediction
│   ├── base.py
│   ├── custodian.py
│   ├── presidio.py
│   ├── obi_deid.py
│   ├── philter.py
│   ├── johnsnow.py
│   ├── llm_openai_compatible.py   # OpenAI / DeepSeek / Kimi / Qwen (hosted)
│   ├── llm_local_hf.py           # Local HF inference (Gemma, Qwen3.5-4B, ...)
│   └── llm_gemma.py
├── evaluation/        # Strict / type / relaxed span scoring + leakage rate
│   └── metrics.py
├── scripts/
│   └── run_benchmark.py
├── data/              # Local datasets (git-ignored, see data/README.md)
├── results/           # Per-doc JSONL + summary JSON
└── web/               # Static dashboard, deploys to Vercel (see web/README.md)
```

## Web dashboard

A self-contained static dashboard lives in `web/`. After running benchmarks,
publish the numbers to the dashboard:

```bash
python scripts/publish_results.py
cd web && python -m http.server 8080      # local preview
# or `vercel deploy --prod` for a public URL
```

See [`web/README.md`](web/README.md) for deployment notes and privacy
caveats around publishing per-document samples.

## Quick start

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg
python -m spacy download es_core_news_lg     # MEDDOCAN / CARMEN-I

# 2. Configure secrets (copy template, fill in YOUR keys, NEVER commit)
cp .env.example .env
$EDITOR .env

# 3. Drop ASQ-PHI under data/asq_phi/raw/  (see data/README.md)

# 4. Run a sanity check
python scripts/run_benchmark.py --benchmark asq_phi --systems presidio --limit 20

# 5. Run the real comparison
python scripts/run_benchmark.py --benchmark asq_phi \
    --systems custodian_all presidio obi openai deepseek kimi qwen gemma \
    --include-text

# 6. Publish to the static dashboard
python scripts/publish_results.py
```

`custodian_all` automatically expands into one row per Custodian
configuration (MASKED × redact, MASKED × transform, PROPRIETARY × redact,
PROPRIETARY × transform). Pick a specific variant with
`--systems custodian:PROPRIETARY:transform`.

## Metrics

| Metric | Definition |
| --- | --- |
| Precision / Recall / F1 | Span-level with `--mode` ∈ {strict, type, relaxed} |
| Leakage rate | `1 - recall` (fraction of gold PHI not masked) |
| Char leakage rate | Fraction of gold PHI **characters** not masked (weights long spans heavier) |
| Over-redaction | Predicted spans not overlapping any gold span (= FP). On ASQ-PHI hard negatives this becomes a pure false-positive rate. |

Recall is the regulatory metric for HIPAA — a missed PHI span is a compliance
breach regardless of how clean the rest of the output is. Track it
independently of F1.

## Adding a new system

1. Subclass `systems.base.DeIDSystem`.
2. Implement `predict(text) -> Prediction` returning a list of
   `PredictedSpan(start, end, label, text, score)` and (optionally) a
   transformed text.
3. Register it in `scripts/run_benchmark.py::_build_system`.

## Adding a new benchmark

1. Subclass `benchmarks.base.Benchmark`.
2. Implement `__iter__` yielding `Document(doc_id, text, gold_spans)`.
3. Register in `benchmarks/__init__.py::REGISTRY`.

## Security notes

- `.env` is git-ignored. Never paste API keys into PRs, docs, or chat.
- Never commit DUA-protected text (`data/n2c2_*`, `data/mimic_iv_note/`,
  `data/carmen_i/`) — `.gitignore` covers them by default.
- Results may quote PHI verbatim in the `predictions[].spans[].text` field.
  Keep `results/` local-only when working with non-synthetic corpora.

## Reproducibility checklist (for the eventual paper)

- [ ] Pin `requirements.txt` versions
- [ ] Log the `OPENAI_MODEL` / `KIMI_MODEL` / etc. used per run (the runner
      already records the system `name`, but extend to record `model` and
      `temperature` if you tweak)
- [ ] Re-run with seeded sampling (`--limit` + a fixed seed) for cost-bounded
      ablations
- [ ] Report Custodian SDK version (`pip show custodian-labs`)
- [ ] Document any prompt changes; if you change `SYSTEM_PROMPT`, re-run every
      LLM, not just the new one

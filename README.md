# CustodianAI · PHI/PII Benchmark Harness

End-to-end evaluation harness for measuring how well **Custodian Labs' Guardian
Layer** transforms clinical text, compared against:

- **Specialized de-id systems**: Microsoft Presidio, Philter (UCSF),
  John Snow Labs Spark NLP for Healthcare, OBI `deid_roberta_i2b2`.
- **Frontier LLMs** prompted as PHI detectors: OpenAI GPT-5, DeepSeek V4,
  Moonshot Kimi K2.6, Alibaba Qwen 3.7, Google Gemma 4.

## Project layout

```
custodianai/
├── benchmarks/        # Dataset loaders → Document(text, gold_spans)
│   ├── base.py
│   ├── asq_phi.py     # English synthetic, no DUA
│   └── meddocan.py    # Spanish synthetic, no DUA
├── systems/           # Each model wraps a DeIDSystem.predict(text) → Prediction
│   ├── base.py
│   ├── custodian.py
│   ├── presidio.py
│   ├── obi_deid.py
│   ├── philter.py
│   ├── johnsnow.py
│   ├── llm_openai_compatible.py   # OpenAI / DeepSeek / Kimi / Qwen
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

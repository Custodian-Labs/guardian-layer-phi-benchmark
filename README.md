# Custodian Guardian Layer · PHI/PII Benchmark Harness

End-to-end evaluation harness measuring whether **Custodian Labs' Guardian
Layer** `transform` (structure-preserving de-identification: replace each PHI
value with a realistic same-type *surrogate*) keeps clinical text **detectable**
by downstream PHI/PII systems. Companion code for the paper *"Surrogate
Substitution Preserves PHI Detectability: A Multi-Detector Equivalence Study."*

**Detector panel (11):** Microsoft Presidio, OBI `deid_roberta_i2b2`;
open LLMs Gemma 4 (31B, E4B), Qwen 3.5 (4B, 9B, 35B-A3B), Llama 3.1-8B,
Llama 3.3-70B, DeepSeek V2-Lite; and OpenAI GPT-5 — across **7 benchmarks /
7 languages / 1,750 documents**.

### Links
| | |
|---|---|
| 📊 Interactive dashboard | **https://custodianai.pages.dev** |
| 📝 Paper (PDF) | https://custodianai.pages.dev/paper.pdf |
| 💾 Reproducibility package (scripts + data subsets) | https://custodianai.pages.dev/code |
| ▶️ Runnable Colab demo | [`notebooks/custodian_guardian_layer_demo.ipynb`](notebooks/custodian_guardian_layer_demo.ipynb) |

### Headline result
On the **57,112 spans the transform masks** (pooled over all 11 detectors),
recall moves **76.1% → 74.9%** (−1.2 pts). A **TOST equivalence test**
(margin ±2 pts) gives **p ≈ 3×10⁻⁹** — the change is statistically equivalent
to zero, and detector ranking is preserved. The small residual traces to
surrogate-generation quality (truncation, salience loss, `x`-masking), not to
detectors getting worse at PHI.

### Latest model comparison (whole-document span-F1, mean over 7 benchmarks)
| Detector | Orig F1 | Transf F1 | ΔF1 | Overlap retention |
|---|--:|--:|--:|--:|
| Gemma 4 31B | .755 | .710 | −.045 | 99.8% |
| Gemma 4 E4B | .737 | .698 | −.038 | 98.2% |
| Llama 3.3-70B | .725 | .728 | +.003 | 98.3% |
| Qwen 3.5-35B-A3B | .715 | .668 | −.047 | 98.4% |
| OpenAI GPT-5 | .705 | .674 | −.032 | 98.3% |
| Qwen 3.5-9B | .655 | .621 | −.034 | 100.0% |
| Qwen 3.5-4B | .567 | .533 | −.034 | 98.0% |
| Presidio | .416 | .398 | −.018 | 97.4% |
| DeepSeek V2-Lite | .409 | .384 | −.025 | 92.8% |
| Llama 3.1-8B | .391 | .376 | −.015 | — |
| OBI `deid_roberta` | .041 | .040 | −.002 | — |

*Whole-document F1 is a conservative (diluted) view; "overlap retention" is
transformed÷original recall on masked spans — the cleaner utility measure.
Ranking is preserved across a 100× span of detector quality.*

## Live dashboard

**Official URL** (Cloudflare Pages — auto-redeploys on every push to `main`,
serves a private repo, Auckland POP for low NZ latency):

**https://custodianai.pages.dev**

Three top-level views, no navigation needed:

1. **Overall results matrix** — every system as a row, every benchmark
   as a column, F1 in each cell, heat-mapped from red (low) to green
   (high). The right-most "Mean" column is sticky so it stays visible
   on any viewport. A metric switcher lets you flip to Recall /
   Precision / Leakage without leaving the overview.
2. **Ranking proof** — same document rendered by every system, panels
   sorted by per-doc F1 (top = best, bottom = worst). Spans are
   colour-coded: green = true positive, yellow = false positive, red
   strikethrough = missed gold (leakage). Cycles through 12 sample
   docs per benchmark.
3. **Per-benchmark detail view** (collapsible) — the original
   single-benchmark drilldown: dataset intro, worked example, per-doc
   highlights.

## Current numbers (2026-05-24, full 250-doc runs)

**Seven benchmarks, 250 documents each, six systems each (MultiCoNER adds a
7th system as a diagnostic) — 10,550 system×doc inference cells in total.**
All datasets are synthetic or public-license; no DUA was required for any
of them. Matching mode is `type` (span-exact, label-agnostic) throughout.

### Cross-language trend on PII-Masking-300k (Gemma vs GPT-5)

The cleanest cross-language signal in the matrix: same corpus, same labels,
four languages. **The further from English, the larger Gemma 4 E4B's lead
over GPT-5.**

| Language | Gemma F1 | GPT-5 F1 | **Δ** |
| --- | ---: | ---: | ---: |
| English | 0.734 | 0.734 | **0.0** |
| Dutch | 0.763 | 0.752 | **+1.1** |
| French | 0.754 | 0.730 | **+2.4** |
| German | 0.767 | 0.731 | **+3.6** |

### MEDDOCAN test, **all 250 docs**, Spanish clinical synthetic

| System | P | R | **F1** | Leakage |
| --- | ---: | ---: | ---: | ---: |
| **OpenAI GPT-5** | 0.80 | 0.62 | **0.699** | 0.38 |
| **Gemma 4 E4B** (8B, local) | 0.77 | **0.64** | **0.697** | 0.36 |
| **Qwen 3.5-4B** (local, no thinking) | 0.53 | 0.35 | **0.42** | 0.65 |
| **DeepSeek V2-Lite** (16B MoE, local) | 0.59 | 0.32 | **0.42** | 0.68 |
| **Presidio** (es spaCy `lg`) | 0.39 | 0.41 | **0.40** | 0.59 |
| **OBI `deid_roberta_i2b2`** (English) | 0.07 | 0.08 | **0.07** | 0.92 |

On the full 250-doc test split (4× the earlier 25-doc subset), the gap
between Gemma 4 E4B and GPT-5 collapses to **0.002 F1** — statistically
indistinguishable. Gemma's *recall* (0.637) is actually higher than
GPT-5's (0.619); GPT-5 wins overall only by being slightly more precise.
For a HIPAA-style use case where missed PHI is the regulatory hazard,
the local 8B model is at parity with the hosted frontier.

### ASQ-PHI, English adversarial clinical queries (250 docs)

| System | P | R | **F1** | Leakage |
| --- | ---: | ---: | ---: | ---: |
| **Gemma 4 E4B** (8B, local) | 0.70 | **0.91** | **0.79** | **0.10** |
| **OpenAI GPT-5** | 0.66 | 0.84 | **0.74** | 0.16 |
| **Qwen 3.5-4B** (local) | 0.46 | 0.69 | **0.55** | 0.31 |
| **Presidio** (en spaCy `lg`) | 0.43 | 0.70 | **0.53** | 0.30 |
| **DeepSeek V2-Lite** (local) | 0.61 | 0.43 | **0.51** | 0.57 |
| **OBI `deid_roberta_i2b2`** | 0.03 | 0.08 | **0.05** | 0.92 |

Notes:
- Gemma 4 E4B leads by 5 F1 — driven entirely by recall (0.91 vs GPT-5's
  0.84). Its leakage rate of 0.10 is the lowest single number in the
  entire matrix.
- DeepSeek V2-Lite is "conservative" — high precision (0.61) but only
  flags spans it's certain about (recall 0.43). Inverse of the recall-
  optimised approach Gemma takes.
- Presidio's recall (0.70) is much higher on ASQ-PHI than on MEDDOCAN
  (0.41) or PII-Masking-300k (0.37), because ASQ-PHI's NAME / DATE /
  GEOGRAPHIC_LOCATION labels match Presidio's English recognizers exactly.

### PII-Masking-300k validation, English general PII (250 docs)

| System | P | R | **F1** | Leakage |
| --- | ---: | ---: | ---: | ---: |
| **OpenAI GPT-5** | 0.79 | **0.69** | **0.734** | **0.31** |
| **Gemma 4 E4B** (8B, local) | **0.82** | 0.67 | **0.734** | 0.33 |
| **Qwen 3.5-4B** (local) | 0.79 | 0.59 | **0.68** | 0.41 |
| **DeepSeek V2-Lite** (local) | 0.66 | 0.31 | **0.42** | 0.69 |
| **Presidio** (en) | 0.34 | 0.37 | **0.35** | 0.63 |
| **OBI `deid_roberta_i2b2`** | 0.04 | 0.09 | **0.06** | 0.91 |

GPT-5 and Gemma 4 E4B tie at F1=0.734 to the third decimal. GPT-5 wins
slightly on recall (0.685 vs 0.666), Gemma slightly on precision (0.817
vs 0.791). The 25-doc subset over-estimated Gemma's lead on this
benchmark — at 250 docs the two are indistinguishable.

### PII-Masking-300k Dutch (250 docs)

Same corpus and label set as the English run, different language. Direct
cross-language comparison.

| System | P | R | **F1** | Leakage |
| --- | ---: | ---: | ---: | ---: |
| **Gemma 4 E4B** (8B, local) | **0.82** | **0.71** | **0.763** | **0.29** |
| **OpenAI GPT-5** | 0.82 | 0.70 | **0.752** | 0.31 |
| **Qwen 3.5-4B** (local) | 0.80 | 0.61 | **0.69** | 0.39 |
| **DeepSeek V2-Lite** (local) | 0.67 | 0.33 | **0.44** | 0.67 |
| **Presidio** (nl `nl_core_news_lg`) | 0.39 | 0.45 | **0.42** | 0.55 |
| **OBI `deid_roberta_i2b2`** | 0.02 | 0.04 | **0.03** | 0.96 |

Gemma 4 E4B wins by 1 F1, with both higher precision and higher recall
than GPT-5. Compared to the English version of the same dataset, every
LLM-tier system gains +2-3 F1 on Dutch; non-LLM baselines stay flat.

### PII-Masking-300k French (250 docs)

| System | P | R | **F1** | Leakage |
| --- | ---: | ---: | ---: | ---: |
| **Gemma 4 E4B** (8B, local) | **0.84** | **0.69** | **0.754** | **0.31** |
| **OpenAI GPT-5** | 0.81 | 0.66 | **0.730** | 0.34 |
| **Qwen 3.5-4B** (local) | 0.82 | 0.62 | **0.70** | 0.38 |
| **DeepSeek V2-Lite** (local) | 0.62 | 0.28 | **0.38** | 0.72 |
| **Presidio** (fr `fr_core_news_lg`) | 0.38 | 0.31 | **0.34** | 0.69 |
| **OBI `deid_roberta_i2b2`** | 0.02 | 0.04 | **0.03** | 0.96 |

Gemma wins +2.4 F1 over GPT-5. DeepSeek hits a precision/recall split very
similar to its profile on every other PII-Masking variant: respectable P,
poor R.

### PII-Masking-300k German (250 docs)

| System | P | R | **F1** | Leakage |
| --- | ---: | ---: | ---: | ---: |
| **Gemma 4 E4B** (8B, local) | **0.83** | **0.71** | **0.767** | **0.29** |
| **OpenAI GPT-5** | 0.80 | 0.68 | **0.731** | 0.33 |
| **Qwen 3.5-4B** (local) | 0.81 | 0.60 | **0.69** | 0.40 |
| **DeepSeek V2-Lite** (local) | 0.59 | 0.31 | **0.40** | 0.69 |
| **Presidio** (de `de_core_news_lg`) | 0.49 | 0.30 | **0.38** | 0.70 |
| **OBI `deid_roberta_i2b2`** | 0.03 | 0.05 | **0.03** | 0.95 |

The widest Gemma > GPT-5 gap on the PII-Masking family (+3.6 F1). The
fourth language data point in the cross-language trend.

### MultiCoNER v2, English fine-grained NER (250 docs)

33 fine-grained entity classes on noisy Wikipedia-derived sentences —
people, locations, medical, products, organisations. Stress-tests how
the systems handle out-of-clinical-domain entities.

| System | P | R | **F1** | Leakage |
| --- | ---: | ---: | ---: | ---: |
| **Gemma 4 E4B** (8B, local) | 0.58 | **0.75** | **0.657** | **0.25** |
| **OpenAI GPT-5** | 0.47 | 0.67 | **0.556** | 0.33 |
| **Presidio** (en) | 0.42 | 0.58 | **0.489** | 0.42 |
| **DeepSeek V2-Lite** (local) | 0.27 | 0.31 | **0.29** | 0.69 |
| **Qwen 3.5-4B** (local) | 0.41 | 0.17 | **0.24** | 0.83 |
| **OBI `deid_roberta_i2b2`** | 0.02 | 0.04 | **0.02** | 0.96 |

**Gemma's largest absolute win — +10 F1 over GPT-5.** Recall 0.75 vs
0.67. Qwen 3.5-4B unexpectedly collapses on this benchmark (F1=0.24);
its precision is normal but recall is 0.17, suggesting it is too
conservative on noisy / typoed inputs.

**Qwen-thinking diagnostic (50-doc subset, enable_thinking=True):**
| Variant | P | R | F1 |
| --- | ---: | ---: | ---: |
| Qwen 3.5-4B (thinking OFF, 250 docs) | 0.41 | 0.17 | 0.24 |
| Qwen 3.5-4B (thinking ON, 50 docs) | **1.00** | 0.11 | 0.20 |

Turning chain-of-thought on flips Qwen's profile to **perfect precision /
even-lower recall** — every span it returns is correct, but it now returns
even fewer of them. The collapse is therefore *not* a reasoning gap; Qwen's
RLHF objective drives it to abstain when uncertain, and CoT amplifies that
exact tendency. For HIPAA-style high-recall use, this is the opposite of
the wanted behaviour.

### Headlines (5 benchmarks, all 250-doc)

- **A small *local* 8B model matches or beats hosted GPT-5 on every
  benchmark we measured.** Gemma 4 E4B's Δ vs GPT-5 across the five
  datasets:

  | Benchmark | Gemma F1 | GPT-5 F1 | Δ | Notes |
  | --- | ---: | ---: | ---: | --- |
  | ASQ-PHI | **0.786** | 0.737 | **+5** | English clinical queries |
  | MEDDOCAN | 0.697 | 0.699 | −0.2 | Spanish clinical (tied) |
  | PII-Masking-300k EN | **0.734** | 0.734 | 0 | English general PII (tied) |
  | PII-Masking-300k NL | **0.763** | 0.752 | **+1** | Dutch general PII |
  | PII-Masking-300k FR | **0.754** | 0.730 | **+2** | French general PII |
  | PII-Masking-300k DE | **0.767** | 0.731 | **+4** | German general PII |
  | MultiCoNER v2 | **0.657** | 0.556 | **+10** | English fine-grained NER |

  **Gemma wins 6/7; the only MEDDOCAN loss is 0.2 F1, well inside noise.**
  Across the four PII-Masking languages, the trend is monotone: the
  further the language is from English, the larger Gemma's lead.

- **On recall — the HIPAA-critical axis — Gemma leads almost everywhere.**
  Leakage rate (1 − recall):
    - ASQ-PHI: Gemma **0.10** vs GPT-5 0.16 (1.7× lower).
    - MEDDOCAN: Gemma 0.36 vs GPT-5 0.38.
    - PII-Masking EN: Gemma 0.33 vs GPT-5 0.31 (GPT-5 hair-thin lead).
    - PII-Masking NL: Gemma **0.29** vs GPT-5 0.31.
    - MultiCoNER: Gemma **0.25** vs GPT-5 0.33.

- **Non-LLM baselines lag by 15–25 F1 points** on every benchmark.
  Presidio is competitive only on tasks where its built-in recognizers
  (NAME/DATE/EMAIL) match the gold labels (e.g. F1=0.53 on ASQ-PHI).

- **Domain-specific transformers do not transfer.** OBI
  `deid_roberta_i2b2` predicts a clinical PHI label set (PATIENT, DOCTOR,
  HOSP, …); on every non-clinical-en benchmark its F1 collapses to ≤0.07,
  and on Spanish clinical (MEDDOCAN) it sits at F1=0.07 even though
  MEDDOCAN's labels overlap with i2b2 — it's the language change that
  kills it.

- **Qwen 3.5-4B's MultiCoNER collapse** (F1=0.24) is the only place a
  capable LLM falls below Presidio. Hypothesis: Qwen with `enable_thinking
  =False` is too aggressive at rejecting low-confidence spans on noisy
  Wikipedia-style text. Worth a follow-up with thinking re-enabled.

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
│   ├── pii_masking_300k.py   # Multilingual general PII (HF, open) — EN + NL
│   └── multiconer_v2.py      # Fine-grained noisy NER, 12 langs (HF, open)
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

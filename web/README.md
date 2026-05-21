# Web demo (static dashboard)

Pure HTML + Tailwind + Chart.js. No build step, no backend, no secrets ever
shipped to the client. Reads pre-aggregated JSON out of `web/data/`.

## Local preview

```bash
cd web
python -m http.server 8080
# open http://localhost:8080
```

The shipped `web/data/*` files are placeholders (all metrics = 0). They get
overwritten by `scripts/publish_results.py` once you have real runs.

## Deploy to Vercel (recommended — gives you a fixed URL)

Two ways.

### A) One-shot CLI deploy

```bash
npm i -g vercel
cd web
vercel deploy --prod
```

Vercel will ask for a project name; pick something like `custodianai-bench`.
The production URL becomes `https://<project>.vercel.app`.

### B) Git-backed (auto-redeploy on push)

1. `git init` at repo root, push to GitHub.
2. https://vercel.com/new → "Import" your repo.
3. **Root directory** = `web/`, **Framework Preset** = "Other", build command
   blank, output directory blank.
4. Deploy. The URL updates whenever you push.

## Updating results

After a new benchmark run:

```bash
python scripts/run_benchmark.py --benchmark asq_phi \
    --systems custodian presidio openai deepseek kimi qwen gemma \
    --include-text                       # samples need this
python scripts/publish_results.py        # writes web/data/<run>.summary.json
                                         # and web/data/index.json
git add web/data/ && git commit -m "publish bench results $(date +%F)"
git push                                 # Vercel auto-redeploys
```

## What gets published

`scripts/publish_results.py` only bundles per-document samples for
benchmarks in its `SAMPLE_SAFE` set (currently ASQ-PHI and MEDDOCAN — both
synthetic / open-license). For DUA-protected corpora (n2c2, MIMIC-IV-Note,
CARMEN-I) only the aggregate numbers ship to `web/data/`; per-doc text never
leaves your machine unless you explicitly opt in with `--publish-samples`.

## Privacy / hosting decisions to make before going public

- **Public vs gated**: Vercel projects are public by default. If your numbers
  are sensitive (e.g. Custodian under-performing some commercial competitor),
  add password protection via Vercel's "Deployment Protection" (paid tier) or
  switch to a static host with HTTP auth.
- **Don't publish raw EHR text** even if Vercel's CDN is fast — assume any
  URL with `data/` will be indexed by archive.org within hours.
- **API keys never leave server-side**: the dashboard is static, so the
  client never sees `CUSTODIAN_SDK_API_KEY` or any other secret. Run
  benchmarks locally / in CI, publish only the resulting numbers.

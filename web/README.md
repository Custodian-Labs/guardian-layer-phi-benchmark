# Web demo (static dashboard)

Pure HTML + Tailwind + Chart.js. No build step, no backend, no secrets ever
shipped to the client. Reads pre-aggregated JSON out of `web/data/`.

**Live URL:** https://custodianai.pages.dev

## Local preview

```bash
cd web
python -m http.server 8080
# open http://localhost:8080
```

`web/data/*` ships with real benchmark results. They get overwritten by
`scripts/publish_results.py` each time you re-run.

## Deploy

The dashboard is deployed on **Cloudflare Pages**, which supports private
GitHub repos on its free tier (unlike GitHub Pages, which needs Pro for
private repos).

### Cloudflare Pages setup (one-time, ~5 min)

1. Sign in at https://dash.cloudflare.com/
2. **Workers & Pages → Create application → Pages → Connect to Git**
3. Authorise GitHub to expose just this repo (`Only select repositories`)
4. Select `Custodianai`
5. Build configuration:
   ```
   Project name:              custodianai
   Production branch:         main
   Framework preset:          None
   Build command:             (leave blank)
   Build output directory:    web
   Root directory:            (leave blank)
   ```
6. Save and Deploy → the URL goes live in ~1 min

After this, every `git push` to `main` triggers an auto-redeploy on
Cloudflare. ~30 s end-to-end from push to live.

### Alternative deploys (not currently used)

- **Vercel** — same flow as Cloudflare, public by default, password
  protection is paid-tier.
- **GitHub Pages** — only free on public repos or the
  `<username>.github.io` user site.
- **Local Cloudflare tunnel** — `cloudflared tunnel --url
  http://localhost:8090` gives a temporary `*.trycloudflare.com` URL,
  good for live debugging against the local server.

## Updating results

After a new benchmark run:

```bash
python scripts/run_benchmark.py --benchmark asq_phi \
    --systems presidio obi openai gemma_e4b qwen3_5_4b deepseek_v2_lite \
    --include-text                       # samples need this
python scripts/publish_results.py        # writes web/data/<run>.summary.json
                                         # and web/data/index.json
git add web/data/ && git commit -m "publish bench results $(date +%F)"
git push                                 # Cloudflare Pages auto-redeploys
```

## What gets published

`scripts/publish_results.py` only bundles per-document samples for
benchmarks listed in its `SAMPLE_SAFE` set (currently the seven open
synthetic benchmarks we evaluate). For DUA-protected corpora (n2c2,
MIMIC-IV-Note, CARMEN-I) only aggregate numbers ship to `web/data/`;
per-doc text never leaves your machine unless you explicitly opt in
with `--publish-samples`.

## Privacy / hosting notes

- The dashboard is fully static. The client never sees
  `CUSTODIAN_SDK_API_KEY` / `OPENAI_API_KEY` / any other secret. All
  inference happens off-page; the dashboard only displays the
  resulting numbers.
- Don't publish raw EHR text. Cloudflare's CDN is fast — any URL with
  `data/` will be indexed by archive.org within hours. Stay on
  `SAMPLE_SAFE` benchmarks for the per-doc drill-down.
- Cloudflare Pages projects can be **password-protected** via
  "Settings → Access policies" if you ever need to gate the URL
  (requires Cloudflare Zero Trust, free tier covers up to 50 users).

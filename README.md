# AI-VCP Taiwan

A cloud-run Taiwan equities VCP research scanner. It downloads daily end-of-day prices from public exchange endpoints, calculates a transparent VCP-style score, and publishes a dated CSV/Markdown report via GitHub Actions.

> **Research tool only. Not investment advice.** The first release is a screening prototype. It is not yet a validated trading strategy or audited backtest. Corporate actions/adjusted prices, historical universe membership, delistings, and survivorship bias require further work before performance claims are meaningful.

## What it does
- Scans TWSE and TPEx daily quote snapshots after the Taiwan market close.
- Applies a trend template, contraction structure, volume dry-up, pivot breakout, and relative-strength checks.
- Ranks candidates with an explainable 100-point score.
- Saves dated reports as workflow artifacts and keeps a latest report in the repository.

## Important data limitations
The public bulk endpoints are daily snapshots, not full historical databases:
- TWSE: `https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL`
- TPEx: `https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes`

The workflow will fail loudly when an endpoint/schema is unavailable or returns implausibly few rows. This initial release does **not** claim complete multi-year backfill or adjusted historical prices. A robust historical backtest is a separate phase.

## Run
GitHub Actions runs weekdays at 06:30 UTC (14:30 Taiwan time; 17:30 Sydney during AEDT / 16:30 during AEST), and can also be started manually from the Actions tab.

## VCP methodology (initial defaults)
- Trend: close above 50/150/200-day averages; 50-day above 150-day; 150-day above 200-day.
- Base: recent 60 trading sessions split into three 20-session windows; local peak-to-trough contractions should generally become shallower.
- Volume: recent 10-session average volume should contract relative to 50 sessions.
- Breakout: close above the prior 20-session pivot high, with volume expansion.
- Relative strength: 63-session stock return compared with TAIEX proxy where index data is available.
- Score: 100-point transparent heuristic, not a calibrated probability.

Thresholds live in `config.yaml`.

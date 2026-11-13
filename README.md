# AI-VCP Taiwan

Cloud-based Taiwan stock VCP research scanner. GitHub Actions updates a rolling two-year daily OHLCV dataset, evaluates the VCP-style rules, and publishes a ranked report.

## Current implementation
- Builds a first-pass TWSE/TPEx common-stock universe from the exchanges' latest daily lists.
- Downloads two years of daily adjusted OHLCV from Yahoo Finance through `yfinance` (TWSE suffix `.TW`, TPEx suffix `.TWO`).
- Persists history under `data/history/` and updates it on scheduled runs.
- Calculates a transparent 100-point VCP-style heuristic and writes `reports/latest.md` and `reports/latest.csv`.
- Scheduled weekdays at 06:30 UTC (14:30 Taiwan time; 17:30 Sydney during AEDT / 16:30 during AEST).

## Data-source caveats
The exchange OpenAPI endpoints used for the latest universe/snapshot are:
- TWSE: https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL
- TPEx: https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes

Historical bars currently come from Yahoo Finance, **not directly from the exchanges**. Coverage, ticker mapping, adjustments and missing records must be checked. A failed/empty download is skipped; the workflow stops if fewer than 300 symbols update. This is not a complete survivorship-bias-free historical database.

Official historical lookup pages document historical availability (TPEx mainboard history from Nov 20, 2013; individual stock daily info since 1994): https://www.tpex.org.tw/en-us/mainboard/trading/info/statistics/day.html and https://www.tpex.org.tw/en-us/mainboard/trading/info/stock-pricing.html

## VCP scoring (prototype)
- Trend template: close > 50-day MA > 150-day MA > 200-day MA.
- Three consecutive 20-session contraction windows across the recent 60 sessions.
- Volume dry-up: 10-day average volume relative to 50-day average.
- Breakout: close above prior 20-session high with volume expansion.
- Relative strength proxy: positive 63-session stock return (not yet benchmark-relative).
- Score is a heuristic ranking, not a probability or a buy recommendation.

## Not yet implemented / validation required
- Official adjusted historical price feed and reconciliation against exchange records.
- Historical universe membership and delisted securities (survivorship-bias control).
- Proper benchmark-relative strength against TAIEX.
- More faithful pivot/contraction detection and corporate-action event validation.
- Transaction-cost-aware, point-in-time backtesting and walk-forward validation.

Do not use this prototype alone to make trades. Review the actual report's data date and coverage before interpreting a signal.

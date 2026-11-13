from __future__ import annotations

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import yaml

DATA = Path("data/history")
OUT = Path("reports")
HORIZONS = (5, 10, 20, 60)


def find_signals(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Vectorized VCP signal detection. No per-day Python/pandas rolling loop."""
    d = df.sort_index().dropna(subset=["open", "close", "high", "low", "volume"]).copy()
    if len(d) < 220 + max(HORIZONS) + 1:
        return np.array([], dtype=int)

    c = d["close"].astype(float)
    h = d["high"].astype(float)
    l = d["low"].astype(float)
    v = d["volume"].astype(float)

    ma50 = c.rolling(50, min_periods=50).mean()
    ma150 = c.rolling(150, min_periods=150).mean()
    ma200 = c.rolling(200, min_periods=200).mean()
    trend = (c > ma50) & (ma50 > ma150) & (ma150 > ma200) & (ma200 > ma200.shift(20))

    base = int(cfg["vcp"]["base_days"])
    segs = int(cfg["vcp"]["segments"])
    if base % segs:
        raise ValueError("base_days must be divisible by segments")
    width = base // segs
    # For signal day i, use the three consecutive prior windows:
    # [i-base, i-base+width), ..., [i-1]. shift excludes signal day.
    depths = []
    for offset in (base - width + 1, width + 1, 1):
        peak = h.rolling(width, min_periods=width).max().shift(offset)
        trough = l.rolling(width, min_periods=width).min().shift(offset)
        depths.append((peak - trough) / peak * 100)

    min_c = float(cfg["vcp"]["min_contraction_pct"])
    max_c = float(cfg["vcp"]["max_contraction_pct"])
    tol = float(cfg["vcp"]["contraction_tolerance"])
    contraction = pd.Series(True, index=d.index)
    for depth in depths:
        contraction &= depth.between(min_c, max_c)
    for a, b in zip(depths, depths[1:]):
        contraction &= b <= a * tol

    avg10 = v.shift(1).rolling(10, min_periods=10).mean()
    avg50 = v.shift(1).rolling(50, min_periods=50).mean()
    dry = (avg50 > 0) & (avg10 / avg50 <= float(cfg["vcp"]["volume_dryup_ratio"]))
    pivot_n = int(cfg["vcp"]["pivot_lookback"])
    pivot = h.rolling(pivot_n, min_periods=pivot_n).max().shift(1)
    breakout = (c > pivot) & (avg50 > 0) & (v / avg50 >= float(cfg["vcp"]["breakout_volume_ratio"]))

    avg_value = (c * v).shift(1).rolling(20, min_periods=20).mean()
    liquid = avg_value >= float(cfg["universe"]["min_avg_value_20d_twd"])
    mask = trend & contraction & dry & breakout & liquid

    # Preserve the prior study's requirement for a complete 60-session outcome.
    eligible = np.zeros(len(d), dtype=bool)
    eligible[220: len(d) - max(HORIZONS) - 1] = True
    candidates = np.flatnonzero(mask.to_numpy() & eligible)

    # Suppress repeat signals within 20 trading sessions, per stock.
    selected = []
    last = -999
    for i in candidates:
        if i - last >= 20:
            selected.append(int(i))
            last = int(i)
    return np.asarray(selected, dtype=int)


def main():
    started = time.time()
    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    costs = cfg.get("backtest", {})
    fee_tax = float(costs.get("round_trip_fee_tax_pct", 0.585)) / 100
    slippage = float(costs.get("round_trip_slippage_pct", 0.20)) / 100
    total_cost = fee_tax + slippage

    universe = pd.read_csv("data/universe.csv", dtype={"code": str})
    events = []
    scanned = 0
    missing = 0
    for item in universe.itertuples(index=False):
        path = DATA / f"{item.code}.csv"
        if not path.exists():
            missing += 1
            continue
        try:
            df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
            if len(df) < 281:
                continue
            signals = find_signals(df, cfg)
            if not len(signals):
                scanned += 1
                continue

            dates = pd.to_datetime(df["date"])
            opens = df["open"].to_numpy(dtype=float)
            closes = df["close"].to_numpy(dtype=float)
            highs = df["high"].to_numpy(dtype=float)
            lows = df["low"].to_numpy(dtype=float)
            for i in signals:
                entry_i = i + 1
                entry = opens[entry_i]
                if not np.isfinite(entry) or entry <= 0:
                    continue
                row = {
                    "code": item.code, "name": item.name, "market": item.market,
                    "signal_date": str(dates.iloc[i].date()),
                    "entry_date": str(dates.iloc[entry_i].date()),
                    "entry_price": entry,
                    "signal_close": closes[i],
                }
                for horizon in HORIZONS:
                    exit_close = closes[entry_i + horizon]
                    gross = exit_close / entry - 1
                    row[f"return_{horizon}d_gross_pct"] = gross * 100
                    row[f"return_{horizon}d_net_pct"] = (gross - total_cost) * 100
                future_high = highs[entry_i:entry_i + 20]
                future_low = lows[entry_i:entry_i + 20]
                row["mfe_20d_pct"] = (np.nanmax(future_high) / entry - 1) * 100
                row["mae_20d_pct"] = (np.nanmin(future_low) / entry - 1) * 100
                events.append(row)
            scanned += 1
        except Exception as exc:
            print(f"Backtest skip {item.code}: {exc}", flush=True)

        if scanned and scanned % 100 == 0:
            print(f"Progress: {scanned} stocks scanned; {len(events)} events; elapsed {time.time()-started:.1f}s", flush=True)

    OUT.mkdir(exist_ok=True)
    detail = pd.DataFrame(events)
    detail.to_csv(OUT / "backtest_events.csv", index=False, encoding="utf-8-sig")
    summary = []
    for horizon in HORIZONS:
        col = f"return_{horizon}d_net_pct"
        vals = detail[col].dropna() if not detail.empty else pd.Series(dtype=float)
        summary.append({
            "horizon_days": horizon,
            "trades": int(len(vals)),
            "win_rate_pct": round(float((vals > 0).mean() * 100), 2) if len(vals) else None,
            "avg_net_return_pct": round(float(vals.mean()), 2) if len(vals) else None,
            "median_net_return_pct": round(float(vals.median()), 2) if len(vals) else None,
        })
    summ = pd.DataFrame(summary)
    summ.to_csv(OUT / "backtest_summary.csv", index=False, encoding="utf-8-sig")
    meta = {
        "signals": len(detail), "stocks_scanned": scanned, "missing_history_files": missing,
        "elapsed_seconds": round(time.time() - started, 1),
        "cost_assumption_pct": round(total_cost * 100, 3),
        "fee_tax_pct": round(fee_tax * 100, 3), "slippage_pct": round(slippage * 100, 3),
        "method": "vectorized event study; next-session open entry; fixed horizon close exits; repeat signals per stock suppressed within 20 sessions",
        "limitations": ["Yahoo Finance adjusted OHLCV", "current universe only (survivorship bias)", "not a portfolio simulation", "no stop-loss/position sizing", "fees are configurable approximations"],
    }
    (OUT / "backtest_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "backtest_summary.md").write_text(
        "# AI-VCP historical event study\n\n"
        f"- Signals: **{len(detail)}**\n- Stocks scanned: **{scanned}**\n"
        f"- Elapsed: **{meta['elapsed_seconds']} seconds**\n"
        f"- Estimated round-trip cost: **{total_cost*100:.3f}%**\n"
        "- Entry: next trading session open. Exit: close after the stated horizon.\n"
        "- This is an event study, not a portfolio backtest; survivorship bias remains.\n\n"
        + summ.to_markdown(index=False) + "\n", encoding="utf-8"
    )
    print(summ.to_string(index=False), flush=True)
    print(f"Completed: {scanned} stocks, {len(detail)} events, {meta['elapsed_seconds']} seconds", flush=True)


if __name__ == "__main__":
    main()

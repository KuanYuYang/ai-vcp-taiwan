from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import yaml

DATA = Path("data/history")
OUT = Path("reports")


def signal_at(df: pd.DataFrame, i: int, cfg: dict) -> dict | None:
    # All features are calculated using data available at the signal-day close.
    hist = df.iloc[: i + 1]
    if len(hist) < 220:
        return None
    c, h, l, v = (hist[x].astype(float) for x in ["close", "high", "low", "volume"])
    price = float(c.iloc[-1])
    mas = [float(c.rolling(n).mean().iloc[-1]) for n in (50, 150, 200)]
    ma200_prev = float(c.rolling(200).mean().iloc[-21])
    trend = price > mas[0] > mas[1] > mas[2] and mas[2] > ma200_prev
    if not trend:
        return None

    base = int(cfg["vcp"]["base_days"])
    segs = int(cfg["vcp"]["segments"])
    width = base // segs
    base_df = hist.iloc[-base-1:-1]  # exclude signal day from the contraction base
    if len(base_df) < base:
        return None
    depths = []
    for j in range(segs):
        part = base_df.iloc[j * width:(j + 1) * width]
        peak, trough = float(part["high"].max()), float(part["low"].min())
        if peak <= 0:
            return None
        depths.append((peak - trough) / peak * 100)
    contraction_ok = all(
        cfg["vcp"]["min_contraction_pct"] <= x <= cfg["vcp"]["max_contraction_pct"]
        for x in depths
    ) and all(
        depths[j + 1] <= depths[j] * cfg["vcp"]["contraction_tolerance"]
        for j in range(len(depths) - 1)
    )
    if not contraction_ok:
        return None

    prior_v = v.iloc[:-1]
    avg10, avg50 = float(prior_v.tail(10).mean()), float(prior_v.tail(50).mean())
    if avg50 <= 0 or avg10 / avg50 > cfg["vcp"]["volume_dryup_ratio"]:
        return None
    pivot_n = int(cfg["vcp"]["pivot_lookback"])
    pivot = float(h.iloc[-pivot_n-1:-1].max())
    breakout_vol = float(v.iloc[-1] / avg50)
    if price <= pivot or breakout_vol < cfg["vcp"]["breakout_volume_ratio"]:
        return None
    avg_value = float((c * v).iloc[-21:-1].mean())
    if avg_value < float(cfg["universe"]["min_avg_value_20d_twd"]):
        return None
    return {"pivot": pivot, "signal_close": price, "base_low": float(base_df["low"].min())}


def main():
    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    costs = cfg.get("backtest", {})
    # Approximate full-commission common-stock round trip:
    # 0.1425% buy + 0.1425% sell + 0.30% sell-side transaction tax.
    fee_tax = float(costs.get("round_trip_fee_tax_pct", 0.585)) / 100
    slippage = float(costs.get("round_trip_slippage_pct", 0.20)) / 100
    total_cost = fee_tax + slippage
    horizons = [5, 10, 20, 60]
    universe = pd.read_csv("data/universe.csv", dtype={"code": str})
    events = []
    for item in universe.itertuples(index=False):
        path = DATA / f"{item.code}.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
            df = df.dropna(subset=["open", "close", "high", "low", "volume"])
            last_signal_i = -999
            for i in range(220, len(df) - max(horizons) - 1):
                if i - last_signal_i < 20:
                    continue
                sig = signal_at(df, i, cfg)
                if sig is None:
                    continue
                # No look-ahead: entry is next session open, after signal-day close.
                entry = float(df["open"].iloc[i + 1])
                if not np.isfinite(entry) or entry <= 0:
                    continue
                row = {
                    "code": item.code, "name": item.name, "market": item.market,
                    "signal_date": str(df.index[i].date()),
                    "entry_date": str(df.index[i + 1].date()),
                    "entry_price": entry, "signal_close": sig["signal_close"],
                    "pivot": sig["pivot"], "base_low": sig["base_low"],
                }
                for h in horizons:
                    exit_close = float(df["close"].iloc[i + 1 + h])
                    gross = exit_close / entry - 1
                    row[f"return_{h}d_gross_pct"] = gross * 100
                    row[f"return_{h}d_net_pct"] = (gross - total_cost) * 100
                # Maximum adverse/favourable excursion over next 20 sessions, from entry.
                future = df.iloc[i + 1:i + 21]
                row["mfe_20d_pct"] = (float(future["high"].max()) / entry - 1) * 100
                row["mae_20d_pct"] = (float(future["low"].min()) / entry - 1) * 100
                events.append(row)
                last_signal_i = i
        except Exception as exc:
            print(f"Backtest skip {item.code}: {exc}")

    OUT.mkdir(exist_ok=True)
    detail = pd.DataFrame(events)
    detail.to_csv(OUT / "backtest_events.csv", index=False, encoding="utf-8-sig")
    summary = []
    for h in horizons:
        col = f"return_{h}d_net_pct"
        if detail.empty:
            summary.append({"horizon_days": h, "trades": 0, "win_rate_pct": None, "avg_net_return_pct": None, "median_net_return_pct": None})
        else:
            vals = detail[col].dropna()
            summary.append({
                "horizon_days": h, "trades": int(len(vals)),
                "win_rate_pct": round(float((vals > 0).mean() * 100), 2),
                "avg_net_return_pct": round(float(vals.mean()), 2),
                "median_net_return_pct": round(float(vals.median()), 2),
            })
    summ = pd.DataFrame(summary)
    summ.to_csv(OUT / "backtest_summary.csv", index=False, encoding="utf-8-sig")
    meta = {
        "signals": len(detail), "cost_assumption_pct": round(total_cost * 100, 3),
        "fee_tax_pct": round(fee_tax * 100, 3), "slippage_pct": round(slippage * 100, 3),
        "method": "event study; next-session open entry; fixed horizon close exits; overlapping signals per stock suppressed for 20 sessions",
        "limitations": ["Yahoo Finance adjusted OHLCV", "current universe only (survivorship bias)", "not a portfolio simulation", "no stop-loss/position sizing", "fees are configurable approximations"],
    }
    (OUT / "backtest_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "backtest_summary.md").write_text(
        "# AI-VCP historical event study\n\n"
        f"- Signals: **{len(detail)}**\n- Estimated round-trip cost: **{total_cost*100:.3f}%**\n"
        "- Entry: next trading session open. Exit: close after the stated horizon.\n"
        "- This is an event study, not a portfolio backtest; survivorship bias remains.\n\n"
        + summ.to_markdown(index=False) + "\n", encoding="utf-8"
    )
    print(summ.to_string(index=False))
    if len(detail) == 0:
        print("No historical signals; no performance inference can be made.")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

DATA = Path("data/history")
REPORTS = Path("reports")


def evaluate(df: pd.DataFrame, benchmark: pd.Series, cfg: dict):
    df = df.sort_index().dropna(subset=["close"])
    if len(df) < 220:
        return None
    c, h, l, v = (df[x].astype(float) for x in ["close", "high", "low", "volume"])
    price = float(c.iloc[-1])
    ma50, ma150, ma200 = [float(c.rolling(n).mean().iloc[-1]) for n in (50, 150, 200)]
    ma200_prev = float(c.rolling(200).mean().iloc[-21]) if len(c) >= 220 else np.nan
    trend = bool(price > ma50 > ma150 > ma200 and ma200 > ma200_prev)

    base = int(cfg["vcp"]["base_days"])
    segments = int(cfg["vcp"]["segments"])
    width = base // segments
    depths = []
    # Match the backtest: contraction base uses the prior sessions only,
    # excluding the current signal/breakout day.
    base_df = df.iloc[-base-1:-1]
    if len(base_df) < base:
        return None
    for i in range(segments):
        start = i * width
        end = start + width if i < segments - 1 else base
        window = base_df.iloc[start:end]
        if window.empty:
            return None
        peak, trough = float(window["high"].max()), float(window["low"].min())
        depths.append((peak - trough) / peak * 100 if peak > 0 else np.nan)

    contraction_ok = (
        all(np.isfinite(x) and cfg["vcp"]["min_contraction_pct"] <= x <= cfg["vcp"]["max_contraction_pct"] for x in depths)
        and all(depths[i + 1] <= depths[i] * cfg["vcp"]["contraction_tolerance"] for i in range(len(depths) - 1))
    )

    # Measure pre-breakout contraction volume without including today's possible breakout bar.
    prior = v.iloc[:-1]
    avg10_pre = float(prior.tail(10).mean())
    avg50_pre = float(prior.tail(50).mean())
    dry_ratio = avg10_pre / avg50_pre if avg50_pre > 0 else np.nan

    pivot_n = int(cfg["vcp"]["pivot_lookback"])
    pivot = float(h.iloc[-pivot_n-1:-1].max())
    breakout = bool(price > pivot)
    breakout_vol_ratio = float(v.iloc[-1] / avg50_pre) if avg50_pre > 0 else np.nan
    dry_ok = bool(np.isfinite(dry_ratio) and dry_ratio <= cfg["vcp"]["volume_dryup_ratio"])
    volume_expand = bool(np.isfinite(breakout_vol_ratio) and breakout_vol_ratio >= cfg["vcp"]["breakout_volume_ratio"])

    rs_n = int(cfg["vcp"]["rs_lookback"])
    stock_ret = float(price / c.iloc[-(rs_n + 1)] - 1) if len(c) > rs_n and c.iloc[-(rs_n + 1)] else np.nan
    aligned = pd.concat([c.rename("stock"), benchmark.rename("index")], axis=1).dropna()
    if len(aligned) > rs_n:
        bench_ret = float(aligned["index"].iloc[-1] / aligned["index"].iloc[-(rs_n + 1)] - 1)
        relative_strength = stock_ret - bench_ret if np.isfinite(stock_ret) else np.nan
    else:
        bench_ret, relative_strength = np.nan, np.nan

    # Liquidity guard: exclude names with very low average traded value.
    avg_value20 = float((c * v).tail(20).mean())
    min_value = float(cfg["universe"]["min_avg_value_20d_twd"])
    liquid = avg_value20 >= min_value

    trend_pts = 25 if trend else 0
    contraction_pts = 30 if contraction_ok else 0
    volume_pts = 15 if dry_ok else (7 if np.isfinite(dry_ratio) and dry_ratio < 0.9 else 0)
    rs_pts = 10 if np.isfinite(relative_strength) and relative_strength > 0 else 0
    near_pivot = price <= pivot and price >= pivot * 0.95
    breakout_pts = 20 if breakout and volume_expand and trend and contraction_ok and dry_ok else (10 if near_pivot and trend and contraction_ok else 0)
    liquidity_pts = 0 if liquid else -20
    score = max(0, min(100, trend_pts + contraction_pts + volume_pts + rs_pts + breakout_pts + liquidity_pts))

    confirmed_breakout = bool(liquid and trend and contraction_ok and dry_ok and breakout and volume_expand)
    if confirmed_breakout:
        status = "VCP確認突破"
    elif liquid and trend and contraction_ok and dry_ok and near_pivot:
        status = "VCP整理觀察"
    else:
        status = "條件不足"

    return {
        "score": int(score), "status": status, "close": round(price, 2),
        "ma50": round(ma50, 2), "ma150": round(ma150, 2), "ma200": round(ma200, 2),
        "ma200_rising": bool(ma200 > ma200_prev),
        "contractions_pct": "/".join(f"{x:.1f}" for x in depths),
        "contraction_ok": bool(contraction_ok),
        "pre_break_volume_ratio_10v50": round(dry_ratio, 3) if np.isfinite(dry_ratio) else None,
        "pivot": round(pivot, 2),
        "breakout_volume_ratio": round(breakout_vol_ratio, 3) if np.isfinite(breakout_vol_ratio) else None,
        "return_63d_pct": round(stock_ret * 100, 2) if np.isfinite(stock_ret) else None,
        "taiwan_index_63d_pct": round(bench_ret * 100, 2) if np.isfinite(bench_ret) else None,
        "relative_strength_vs_taiex_pct": round(relative_strength * 100, 2) if np.isfinite(relative_strength) else None,
        "avg_value_20d_twd": round(avg_value20),
        "last_date": str(df.index[-1].date()),
    }


def main():
    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    universe_path = Path("data/universe.csv")
    benchmark_path = Path("data/benchmark_twii.csv")
    if not universe_path.exists() or not benchmark_path.exists():
        raise RuntimeError("Missing universe or TAIEX benchmark; run src/history.py first.")
    universe = pd.read_csv(universe_path, dtype={"code": str})
    benchmark_df = pd.read_csv(benchmark_path, parse_dates=["date"]).set_index("date")
    benchmark = benchmark_df["close"].astype(float)
    rows = []
    for item in universe.itertuples(index=False):
        path = DATA / f"{item.code}.csv"
        if not path.exists():
            continue
        try:
            hist = pd.read_csv(path, parse_dates=["date"]).set_index("date")
            result = evaluate(hist, benchmark, cfg)
            if result:
                rows.append({"code": item.code, "name": item.name, "market": item.market, **result})
        except Exception as exc:
            print(f"Skip {item.code}: {exc}")
    if not rows:
        raise RuntimeError("No valid stocks evaluated; inspect history files and benchmark.")
    report = pd.DataFrame(rows).sort_values(["score", "relative_strength_vs_taiex_pct"], ascending=False)
    REPORTS.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    report.to_csv(REPORTS / f"vcp_{stamp}.csv", index=False, encoding="utf-8-sig")
    report.to_csv(REPORTS / "latest.csv", index=False, encoding="utf-8-sig")
    top = report[report["status"].isin(["VCP確認突破", "VCP整理觀察"])].head(int(cfg["report"]["top_n"]))
    confirmed = int((report["status"] == "VCP確認突破").sum())
    watch = int((report["status"] == "VCP整理觀察").sum())
    lines = [
        f"# AI-VCP Taiwan — {stamp}", "",
        f"- Stocks evaluated: **{len(report)}**",
        f"- Confirmed VCP breakouts: **{confirmed}**",
        f"- VCP watchlist: **{watch}**",
        "- Relative strength is measured against ^TWII over 63 sessions.",
        "- Breakout requires trend + contraction + pre-break volume dry-up + close above pivot + volume expansion.",
        "- Data: Yahoo Finance adjusted OHLCV; this remains a prototype and needs source reconciliation and backtesting.", "",
        "## VCP candidates", "",
        top.to_markdown(index=False), ""
    ]
    (REPORTS / "latest.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:10]))


if __name__ == "__main__":
    main()

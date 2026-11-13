from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import yaml

TWSE_URL = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
TPEX_URL = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; AI-VCP-Taiwan/0.1)"}


def get_json(url: str) -> Any:
    response = requests.get(url, headers=HEADERS, timeout=45)
    response.raise_for_status()
    if "json" not in response.headers.get("content-type", "").lower():
        raise RuntimeError(f"Expected JSON from {url}; got {response.headers.get('content-type')}")
    return response.json()


def first(row: dict, names: list[str], default=None):
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return default


def number(value) -> float:
    if value is None:
        return float("nan")
    s = str(value).replace(",", "").replace("--", "").strip()
    if s in ("", "-", "X", "除權息", "停止交易"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def parse_rows(rows: Any, market: str) -> pd.DataFrame:
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"{market} endpoint returned no rows")
    parsed = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if market == "TWSE":
            code = first(row, ["Code", "證券代號", "股票代號"])
            name = first(row, ["Name", "證券名稱", "股票名稱"], "")
            close = number(first(row, ["ClosingPrice", "收盤價"]))
            open_ = number(first(row, ["OpeningPrice", "開盤價"]))
            high = number(first(row, ["HighestPrice", "最高價"]))
            low = number(first(row, ["LowestPrice", "最低價"]))
            volume = number(first(row, ["TradeVolume", "成交股數"]))
            value = number(first(row, ["TradeValue", "成交金額"]))
            date = first(row, ["Date", "日期"])
        else:
            code = first(row, ["SecuritiesCompanyCode", "Code", "代號", "股票代號"])
            name = first(row, ["CompanyName", "Name", "名稱", "股票名稱"], "")
            close = number(first(row, ["Close", "ClosingPrice", "收盤價"]))
            open_ = number(first(row, ["Open", "OpeningPrice", "開盤價"]))
            high = number(first(row, ["High", "HighestPrice", "最高價"]))
            low = number(first(row, ["Low", "LowestPrice", "最低價"]))
            volume = number(first(row, ["TradingShares", "TradeVolume", "成交股數", "成交量"]))
            value = number(first(row, ["TransactionAmount", "TradeValue", "成交金額"]))
            date = first(row, ["Date", "資料日期", "年月日", "日期"])
        if code is None or not str(code).strip().isdigit():
            continue
        parsed.append({
            "code": str(code).strip(), "name": str(name).strip(), "market": market,
            "date": str(date or ""), "open": open_, "high": high, "low": low,
            "close": close, "volume": volume, "value": value,
        })
    df = pd.DataFrame(parsed)
    if df.empty:
        raise RuntimeError(f"Could not parse {market} schema. Inspect upstream endpoint fields.")
    df = df.dropna(subset=["close", "volume"])
    df = df[(df["close"] > 0) & (df["volume"] > 0)]
    if len(df) < 100:
        raise RuntimeError(f"{market} returned only {len(df)} valid securities; refusing to publish likely-bad data")
    return df


def load_snapshot() -> pd.DataFrame:
    twse = parse_rows(get_json(TWSE_URL), "TWSE")
    tpex = parse_rows(get_json(TPEX_URL), "TPEX")
    # Both are expected to represent the latest trading session; mismatches are a data-quality warning.
    if twse["date"].nunique() == 1 and tpex["date"].nunique() == 1:
        if twse["date"].iloc[0] != tpex["date"].iloc[0]:
            raise RuntimeError(f"Market date mismatch: TWSE={twse['date'].iloc[0]}, TPEx={tpex['date'].iloc[0]}")
    return pd.concat([twse, tpex], ignore_index=True)


def score_series(df: pd.DataFrame, cfg: dict) -> dict | None:
    df = df.sort_values("date").reset_index(drop=True)
    # Snapshot endpoints provide only one session. Until historical bars are backfilled,
    # this function deliberately refuses to invent history.
    if len(df) < int(cfg["universe"]["min_history_days"]):
        return None
    c, h, l, v = (df[x].astype(float) for x in ("close", "high", "low", "volume"))
    n = len(df)
    ma50, ma150, ma200 = c.rolling(50).mean().iloc[-1], c.rolling(150).mean().iloc[-1], c.rolling(200).mean().iloc[-1]
    price = c.iloc[-1]
    trend = price > ma50 > ma150 > ma200 and ma50 > c.iloc[- deixou] if False else price > ma50 > ma150 > ma200
    if not trend:
        trend_points = 0
    else:
        trend_points = 30

    base = int(cfg["vcp"]["base_days"])
    segs = int(cfg["vcp"]["segments"])
    width = base // segs
    contractions = []
    for i in range(segs):
        start = n - base + i * width
        end = n - base + (i + 1) * width
        if end <= start:
            return None
        peak = h.iloc[start:end].max()
        trough = l.iloc[start:end].min()
        if not np.isfinite(peak) or peak <= 0:
            return None
        depth = (peak - trough) / peak * 100
        contractions.append(float(depth))
    contraction_ok = all(
        cfg["vcp"]["min_contraction_pct"] <= d <= cfg["vcp"]["max_contraction_pct"]
        for d in contractions
    ) and all(
        contractions[i + 1] <= contractions[i] * cfg["vcp"]["contraction_tolerance"]
        for i in range(len(contractions) - 1)
    )
    contraction_points = 25 if contraction_ok else max(0, 15 - 3 * sum(
        contractions[i + 1] > contractions[i] * cfg["vcp"]["contraction_tolerance"]
        for i in range(len(contractions) - 1)
    ))

    avg10, avg50 = v.tail(10).mean(), v.tail(50).mean()
    dry_ratio = avg10 / avg50 if avg50 > 0 else np.nan
    dry_ok = np.isfinite(dry_ratio) and dry_ratio <= cfg["vcp"]["volume_dryup_ratio"]
    volume_points = 15 if dry_ok else (8 if np.isfinite(dry_ratio) and dry_ratio < 0.9 else 0)

    lookback = int(cfg["vcp"]["pivot_lookback"])
    pivot = h.iloc[-lookback-1:-1].max() if n > lookback else np.nan
    breakout = bool(np.isfinite(pivot) and price > pivot)
    breakout_ratio = v.iloc[-1] / avg50 if avg50 > 0 else np.nan
    breakout_points = 20 if breakout and breakout_ratio >= cfg["vcp"]["breakout_volume_ratio"] else (12 if breakout else 0)

    rs_n = int(cfg["vcp"]["rs_lookback"])
    ret = price / c.iloc[-rs_n] - 1 if n > rs_n and c.iloc[-rs_n] > 0 else np.nan
    rs_points = 10 if np.isfinite(ret) and ret > 0 else 0

    total = trend_points + contraction_points + volume_points + breakout_points + rs_points
    return {
        "score": int(total), "trend_ok": trend, "contractions_pct": "/".join(f"{x:.1f}" for x in contractions),
        "volume_dryup_ratio": round(float(dry_ratio), 3) if np.isfinite(dry_ratio) else None,
        "pivot": round(float(pivot), 2) if np.isfinite(pivot) else None,
        "breakout": breakout, "breakout_volume_ratio": round(float(breakout_ratio), 3) if np.isfinite(breakout_ratio) else None,
        "return_63d_pct": round(float(ret * 100), 2) if np.isfinite(ret) else None,
        "status": "突破" if breakout and breakout_ratio >= cfg["vcp"]["breakout_volume_ratio"] else ("接近整理" if total >= 55 else "未符合"),
    }


def main():
    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    raw = load_snapshot()
    excluded = tuple(cfg["universe"].get("exclude_code_prefixes", []))
    raw = raw[~raw["code"].str.startswith(excluded)]
    raw = raw[raw["close"] >= cfg["universe"]["min_close_twd"]]
    raw["trade_date"] = raw["date"]
    # This first phase only has one-day snapshots, so no false multi-session signals are emitted.
    results = []
    for _, row in raw.iterrows():
        results.append({
            "code": row["code"], "name": row["name"], "market": row["market"],
            "trade_date": row["trade_date"], "close": row["close"],
            "score": None, "status": "歷史資料尚未建置",
            "note": "公開 bulk API 僅提供最新日快照；歷史 OHLCV 回補完成前不計算 VCP。",
        })
    report = pd.DataFrame(results).sort_values(["market", "code"])
    out = Path("reports")
    out.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    report.to_csv(out / f"universe_{stamp}.csv", index=False, encoding="utf-8-sig")
    report.to_csv(out / "latest.csv", index=False, encoding="utf-8-sig")
    summary = [
        f"# AI-VCP Taiwan — {stamp}",
        "",
        f"- TWSE + TPEx valid securities: **{len(report)}**",
        "- Status: **Data snapshot collected; historical bars not yet backfilled. No VCP ranking is issued in this phase.**",
        "",
        "This is an honest data-ingestion baseline. The scanner intentionally does not compute 50/150/200-day indicators from one-day data.",
        "",
        "## Snapshot preview",
        "",
        report.head(30).to_markdown(index=False),
    ]
    (out / "latest.md").write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary[:8]))


if __name__ == "__main__":
    main()

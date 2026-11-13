from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

DATA = Path("data/history")
REPORTS = Path("reports")


def evaluate(df: pd.DataFrame, cfg: dict):
    df = df.sort_index().dropna(subset=["close"])
    if len(df) < 220:
        return None
    c, h, l, v = (df[x].astype(float) for x in ["close", "high", "low", "volume"])
    price = float(c.iloc[-1])
    ma50, ma150, ma200 = [float(c.rolling(n).mean().iloc[-1]) for n in (50, 150, 200)]
    trend = price > ma50 > ma150 > ma200
    base = int(cfg["vcp"]["base_days"])
    segments = int(cfg["vcp"]["segments"])
    width = base // segments
    depths = []
    for i in range(segments):
        window = df.iloc[-base + i * width : -base + (i + 1) * width if i < segments - 1 else None]
        if window.empty:
            return None
        peak, trough = float(window["high"].max()), float(window["low"].min())
        depths.append((peak - trough) / peak * 100 if peak > 0 else np.nan)
    contraction_ok = all(
        cfg["vcp"]["min_contraction_pct"] <= x <= cfg["vcp"]["max_contraction_pct"] for x in depths
    ) and all(depths[i + 1] <= depths[i] * cfg["vcp"]["contraction_tolerance"] for i in range(len(depths) - 1))
    avg10, avg50 = float(v.tail(10).mean()), float(v.tail(50).mean())
    dry = avg10 / avg50 if avg50 else np.nan
    pivot_n = int(cfg["vcp"]["pivot_lookback"])
    pivot = float(h.iloc[-pivot_n-1:-1].max())
    breakout = price > pivot
    vol_ratio = float(v.iloc[-1] / avg50) if avg50 else np.nan
    ret63 = float(price / c.iloc[-64] - 1) if len(c) >= 64 and c.iloc[-64] else np.nan
    score = 0
    score += 30 if trend else 0
    score += 25 if contraction_ok else 0
    score += 15 if np.isfinite(dry) and dry <= cfg["vcp"]["volume_dryup_ratio"] else (7 if np.isfinite(dry) and dry < 0.9 else 0)
    score += 20 if breakout and vol_ratio >= cfg["vcp"]["breakout_volume_ratio"] else (12 if breakout else 0)
    score += 10 if np.isfinite(ret63) and ret63 > 0 else 0
    status = "放量突破" if breakout and vol_ratio >= cfg["vcp"]["breakout_volume_ratio"] else ("整理觀察" if trend and contraction_ok else "未符合")
    return {
        "score": score, "status": status, "close": round(price, 2),
        "ma50": round(ma50, 2), "ma150": round(ma150, 2), "ma200": round(ma200, 2),
        "contractions_pct": "/".join(f"{x:.1f}" for x in depths),
        "volume_dryup_ratio": round(dry, 3) if np.isfinite(dry) else None,
        "pivot": round(pivot, 2), "breakout_volume_ratio": round(vol_ratio, 3) if np.isfinite(vol_ratio) else None,
        "return_63d_pct": round(ret63 * 100, 2) if np.isfinite(ret63) else None,
        "last_date": str(df.index[-1].date()),
    }


def main():
    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    universe_path = Path("data/universe.csv")
    if not universe_path.exists():
        raise RuntimeError("Missing data/universe.csv; run src/history.py first.")
    universe = pd.read_csv(universe_path, dtype={"code": str})
    rows = []
    for item in universe.itertuples(index=False):
        path = DATA / f"{item.code}.csv"
        if not path.exists():
            continue
        try:
            hist = pd.read_csv(path, parse_dates=["date"]).set_index("date")
            result = evaluate(hist, cfg)
            if result:
                rows.append({"code": item.code, "name": item.name, "market": item.market, **result})
        except Exception as exc:
            print(f"Skip {item.code}: {exc}")
    if not rows:
        raise RuntimeError("No valid VCP candidates; history may not have loaded correctly.")
    report = pd.DataFrame(rows).sort_values(["score", "return_63d_pct"], ascending=False)
    REPORTS.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    report.to_csv(REPORTS / f"vcp_{stamp}.csv", index=False, encoding="utf-8-sig")
    report.to_csv(REPORTS / "latest.csv", index=False, encoding="utf-8-sig")
    top = report.head(int(cfg["report"]["top_n"]))
    lines = [
        f"# AI-VCP Taiwan — {stamp}", "",
        f"- Stocks evaluated: **{len(report)}**",
        f"- Breakouts: **{int((report['status'] == '放量突破').sum())}**",
        "- Data: Yahoo Finance daily adjusted OHLCV; provider coverage and corporate-action handling must be monitored.",
        "- Score is a heuristic ranking, not a probability or a buy recommendation.", "",
        "## Top VCP candidates", "",
        top.to_markdown(index=False), ""
    ]
    (REPORTS / "latest.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:8]))


if __name__ == "__main__":
    main()

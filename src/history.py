from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; AI-VCP-Taiwan/0.2)"}
TWSE = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
TPEX = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"
DATA = Path("data/history")
UNIVERSE = Path("data/universe.csv")


def get_json(url):
    r = requests.get(url, headers=HEADERS, timeout=40)
    r.raise_for_status()
    return r.json()


def pick(row, keys, default=""):
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return default


def build_universe():
    sources = [(TWSE, "TWSE"), (TPEX, "TPEX")]
    out = []
    for url, market in sources:
        rows = get_json(url)
        if not isinstance(rows, list):
            raise RuntimeError(f"{market}: unexpected market response")
        for row in rows:
            code = str(pick(row, ["Code", "SecuritiesCompanyCode", "證券代號", "代號"])).strip()
            name = str(pick(row, ["Name", "CompanyName", "證券名稱", "名稱"])).strip()
            if not code.isdigit() or len(code) != 4:
                continue
            # First-pass common-share universe. Exclude fund/warrant-style code families.
            if code.startswith(("00", "020", "030", "040", "050", "060", "070", "080", "090")):
                continue
            out.append({"code": code, "name": name, "market": market})
    df = pd.DataFrame(out).drop_duplicates("code")
    if len(df) < 500:
        raise RuntimeError(f"Universe unexpectedly small ({len(df)}); inspect upstream API schemas.")
    UNIVERSE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(UNIVERSE, index=False, encoding="utf-8-sig")
    return df


def save_symbol_data(code, symbol, frame):
    if frame is None or frame.empty:
        return False
    frame = frame.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame.columns = [str(c).lower().replace(" ", "_") for c in frame.columns]
    required = ["open", "high", "low", "close", "volume"]
    if not all(c in frame.columns for c in required):
        return False
    frame = frame[required].dropna(subset=["close"]).copy()
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame.index.name = "date"
    path = DATA / f"{code}.csv"
    DATA.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path, parse_dates=["date"]).set_index("date")
        frame = pd.concat([old, frame]).sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
    frame.to_csv(path, float_format="%.4f")
    return True


def main():
    universe = build_universe()
    symbols = []
    for row in universe.itertuples(index=False):
        suffix = ".TW" if row.market == "TWSE" else ".TWO"
        symbols.append((row.code, row.code + suffix))
    # Chunked Yahoo Finance requests; adjusted OHLC is used to reduce split/dividend discontinuities.
    # This is a secondary data vendor, not an official exchange feed.
    ok = 0
    failed = []
    for start in range(0, len(symbols), 40):
        batch = symbols[start:start + 40]
        yahoo = [s for _, s in batch]
        try:
            # First load two years for new symbols; update existing symbols incrementally.
            period = "5d" if all((DATA / f"{code}.csv").exists() for code, _ in batch) else "2y"
            data = yf.download(
                tickers=yahoo, period=period, interval="1d",
                auto_adjust=True, group_by="ticker", threads=True,
                progress=False, timeout=30
            )
            for code, ticker in batch:
                try:
                    frame = data[ticker] if isinstance(data.columns, pd.MultiIndex) else data
                    if save_symbol_data(code, ticker, frame):
                        ok += 1
                    else:
                        failed.append(code)
                except Exception:
                    failed.append(code)
        except Exception:
            failed.extend(code for code, _ in batch)
        time.sleep(1.0)
    # Taiwan Weighted Index benchmark for true relative-strength comparison.
    try:
        benchmark = yf.download("^TWII", period="2y", interval="1d", auto_adjust=True, progress=False, timeout=40)
        if isinstance(benchmark.columns, pd.MultiIndex):
            benchmark.columns = benchmark.columns.get_level_values(0)
        if "Close" in benchmark.columns:
            b = benchmark[["Close"]].rename(columns={"Close": "close"}).dropna()
            b.index = pd.to_datetime(b.index).tz_localize(None)
            b.index.name = "date"
            Path("data").mkdir(exist_ok=True)
            b.to_csv("data/benchmark_twii.csv", float_format="%.4f")
        else:
            raise RuntimeError("TAIEX benchmark Close field missing")
    except Exception as exc:
        raise RuntimeError(f"Unable to download ^TWII benchmark; RS scoring must not silently proceed: {exc}")

    summary = {
        "universe_count": int(len(universe)),
        "history_files_updated": int(ok),
        "failed_or_empty": int(len(set(failed))),
        "provider": "Yahoo Finance via yfinance",
        "adjusted_ohlc": True,
        "history_window": "2y rolling",
    }
    Path("data").mkdir(exist_ok=True)
    Path("data/history_status.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if ok < 300:
        raise RuntimeError(f"Too few symbols updated ({ok}); refusing to publish VCP scan.")


if __name__ == "__main__":
    main()

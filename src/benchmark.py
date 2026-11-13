from pathlib import Path
import yfinance as yf
import pandas as pd

df = yf.download("^TWII", period="2y", interval="1d", auto_adjust=True, progress=False, timeout=40)
if df.empty:
    raise RuntimeError("TAIEX benchmark download returned no data")
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
if "Close" not in df.columns:
    raise RuntimeError("TAIEX Close column missing")
out = df[["Close"]].rename(columns={"Close":"close"}).dropna()
out.index = pd.to_datetime(out.index).tz_localize(None)
out.index.name = "date"
Path("data").mkdir(exist_ok=True)
out.to_csv("data/benchmark_twii.csv", float_format="%.4f")
print(f"TAIEX benchmark rows: {len(out)}; latest: {out.index[-1].date()}")

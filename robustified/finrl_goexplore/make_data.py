import argparse
import os
import pandas as pd
import sys

import yfinance as yf
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.config import INDICATORS


def _fetch_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    # yfinance returns an index of dates; normalize to FinRL's expected schema.
    hist = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
        actions=False,
    )

    # yfinance may return MultiIndex columns even for a single ticker.
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)

    if hist is None or len(hist) == 0:
        raise ValueError(f"no data fetched for ticker={ticker}")

    hist = hist.reset_index()

    # Some yfinance versions use 'Date', some use 'Datetime'
    date_col = "Date" if "Date" in hist.columns else ("Datetime" if "Datetime" in hist.columns else None)
    if date_col is None:
        raise ValueError(f"unexpected yfinance columns for {ticker}: {list(hist.columns)}")

    hist = hist.rename(
        columns={
            date_col: "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    hist["tic"] = ticker

    need = ["date", "open", "high", "low", "close", "volume", "tic"]
    missing = [c for c in need if c not in hist.columns]
    if missing:
        raise ValueError(f"missing columns from yfinance for {ticker}: {missing} got={list(hist.columns)}")

    return hist[need]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+", default=["AAPL"])
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--end", default="2020-01-01")
    p.add_argument("--out", default="finrl_goexplore/data/finrl.csv")
    args = p.parse_args()

    dfs = []
    failed = []
    for t in args.tickers:
        try:
            dfs.append(_fetch_yfinance(t, args.start, args.end))
        except Exception as e:
            failed.append((t, str(e)))
            print(f"WARN: skipping ticker={t} due to error: {e}", file=sys.stderr)

    if len(dfs) == 0:
        raise ValueError(f"no data fetched for any tickers={args.tickers}")

    df = pd.concat(dfs, ignore_index=True)

    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False,
    )
    df = fe.preprocess_data(df)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "tic"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out} rows={len(df)} tickers={df['tic'].nunique()}")
    if failed:
        print(f"WARN: failed tickers: {[t for t, _ in failed]}", file=sys.stderr)


if __name__ == "__main__":
    main()

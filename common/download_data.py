import yfinance as yf

TICKERS = "WHIRLPOOL.NS"
INTERVAL = "1h"
PERIOD = "720d"
ticker_file = f"datasets/{TICKERS}"

yf.download(
            tickers=TICKERS,
            period=PERIOD,
            interval=INTERVAL,
            group_by="Ticker",
            auto_adjust=True,
            prepost=True,
        ).reset_index().to_parquet(ticker_file, index=False, engine="fastparquet")
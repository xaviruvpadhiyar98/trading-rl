import numpy as np
import polars as pl
from pathlib import Path

def load_close_prices(TICKER):
    TRAIN_FILE = Path("datasets") / f"{TICKER}"

    CLOSE_PRICES = np.array(
        pl
        .read_parquet(TRAIN_FILE)
        .with_columns(index=pl.int_range(0, end=pl.count(), eager=False))
        .sort("index")
        .set_sorted("index")
        .group_by_dynamic(
            "index", every="1i", period="40i", include_boundaries=True, closed="right"
        )
        .agg(pl.col("Close"))
        .with_columns(pl.col("Close").list.len().alias("Count"))
        .filter(pl.col("Count") == 40)
        ["Close"]
        .to_list()
    )
    return CLOSE_PRICES
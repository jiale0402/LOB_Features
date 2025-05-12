from lob_features.features.rr import realized_returns

import polars as pl


def realized_volatility(X: pl.DataFrame, window: int) -> pl.Series:
    ret = realized_returns(X, 1)
    return ret.rolling_std(window_size=window).cast(pl.Float32)

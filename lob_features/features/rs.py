from lob_features.features.rr import realized_returns

import polars as pl


def realized_return_skew(X: pl.DataFrame, window: int) -> pl.Series:
    ret = realized_returns(X, 1)
    return ret.rolling_skew(window_size=window).cast(pl.Float32)

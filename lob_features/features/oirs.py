from lob_features.features.oir import order_imbalance_ratio

import polars as pl


def order_imbalance_ratio_skew(X: pl.DataFrame, window: int = 10) -> pl.Series:
    return order_imbalance_ratio(X).rolling_skew(window_size=window)

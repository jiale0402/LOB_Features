from lob_features.features.oir import order_imbalance_ratio

import numpy as np
import polars as pl


def order_imbalance_ratio_kurtosis(X: pl.DataFrame, window: int = 10) -> pl.Series:
    oir = order_imbalance_ratio(X)
    rolling_std = oir.rolling_std(window_size=window)
    rolling_mean = oir.rolling_mean(window_size=window)
    rolling_kurtosis = (
        np.power(oir.to_numpy() - rolling_mean.to_numpy(), 3)
        / window
        / np.power(rolling_std.to_numpy(), 3)
    )
    return pl.Series(name="order_imbalance_ratio_kurtosis", values=rolling_kurtosis).cast(
        pl.Float32
    )

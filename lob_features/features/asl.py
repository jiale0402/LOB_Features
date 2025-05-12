from lob_features.dispatch.config import ASK_PRICES, ASK_VOLUMES
from lob_features.features.sprd import get_ticksize

import numpy as np
import polars as pl


def ask_slope(X: pl.DataFrame, levels: int = 5) -> pl.Series:
    ask_volumes = X.select(ASK_VOLUMES[:levels]).to_numpy()
    ask_prices = X.select(ASK_PRICES[:levels]).to_numpy()
    ask_slope_arr = np.zeros(ask_prices.shape[0])
    ticksize = get_ticksize(X)
    for i in range(ask_prices.shape[1] - 1):
        askprices_1 = ask_prices[:, i]
        askprices_2 = ask_prices[:, i + 1]
        askvolumes_1 = ask_volumes[:, i]
        askvolumes_2 = ask_volumes[:, i + 1]
        ask_slope_arr += (askvolumes_2 - askvolumes_1) / ((askprices_2 - askprices_1) / ticksize)
    ask_slope_arr /= ask_prices.shape[1]
    return pl.Series(name="ask_slope", values=ask_slope_arr).cast(pl.Float32)

from lob_features.dispatch.config import BID_PRICES, BID_VOLUMES
from lob_features.features.sprd import get_ticksize

import numpy as np
import polars as pl


def bid_slope(X: pl.DataFrame, levels: int = 5) -> pl.Series:
    bid_volumes = X.select(BID_VOLUMES[:levels]).to_numpy()
    bid_prices = X.select(BID_PRICES[:levels]).to_numpy()
    bid_slope_arr = np.zeros(bid_prices.shape[0])
    ticksize = get_ticksize(X)
    for i in range(bid_prices.shape[1] - 1):
        bidprices_1 = bid_prices[:, i]
        bidprices_2 = bid_prices[:, i + 1]
        bidvolumes_1 = bid_volumes[:, i]
        bidvolumes_2 = bid_volumes[:, i + 1]
        bid_slope_arr += (bidvolumes_2 - bidvolumes_1) / ((bidprices_2 - bidprices_1) / ticksize)
    bid_slope_arr /= bid_prices.shape[1]
    return pl.Series(name="bid_slope", values=bid_slope_arr).cast(pl.Float32)

from lob_features.dispatch.config import BID_PRICES
from lob_features.features.vwabp import vwabp

import numpy as np
import polars as pl


def bid_price_stability(X: pl.DataFrame, levels: int = 5) -> pl.Series:
    _vwabp = vwabp(X, levels).to_numpy()
    bid_prices = X.select(BID_PRICES[:levels]).to_numpy() - _vwabp.reshape(-1, 1)
    bps = np.sqrt(np.sum(np.square(bid_prices), axis=1) / levels)
    return pl.Series(bps)

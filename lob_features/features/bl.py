from lob_features.dispatch.config import BID_PRICES, BID_VOLUMES

import numpy as np
import polars as pl


def bid_liquidity(X: pl.DataFrame, levels: int = 5) -> pl.Series:
    bid_vols = X.select(BID_VOLUMES[:levels]).to_numpy()
    bid_prices = X.select(BID_PRICES[:levels]).to_numpy()
    with np.errstate(invalid="ignore"):
        al = np.nanmean(bid_vols / bid_prices, axis=1)
    return pl.Series(al)

from lob_features.dispatch.config import ASK_PRICES, BID_PRICES

import numpy as np
import polars as pl


def get_ticksize(X: pl.DataFrame) -> float:
    sprd = (X[ASK_PRICES[0]] - X[BID_PRICES[0]]).to_numpy().ravel()
    return np.min(sprd[sprd > 0])


def spread(X: pl.DataFrame, window: int) -> pl.Series:
    sprd = X[ASK_PRICES[0]] - X[BID_PRICES[0]]
    return pl.Series(name="sprd", values=sprd).cast(pl.Float32)

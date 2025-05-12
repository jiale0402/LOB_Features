from lob_features.dispatch.config import BID_PRICES, BID_VOLUMES

import numpy as np
import polars as pl


def vwabp(X: pl.DataFrame, levels: int = 5) -> pl.Series:
    sum_v = X.select(BID_VOLUMES[:levels]).to_numpy().sum(axis=1)
    vwabp = (
        np.sum(
            X.select(BID_PRICES[:levels]).to_numpy() * X.select(BID_VOLUMES[:levels]).to_numpy(),
            axis=1,
        )
        / sum_v
    )
    return pl.Series(vwabp.ravel())

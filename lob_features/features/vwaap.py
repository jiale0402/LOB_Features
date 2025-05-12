from lob_features.dispatch.config import ASK_PRICES, ASK_VOLUMES

import numpy as np
import polars as pl


def vwaap(X: pl.DataFrame, levels: int = 5) -> pl.Series:
    sum_v = X.select(ASK_VOLUMES[:levels]).to_numpy().sum(axis=1)
    vwaap = (
        np.sum(
            X.select(ASK_VOLUMES[:levels]).to_numpy() * X.select(ASK_PRICES[:levels]).to_numpy(),
            axis=1,
        )
        / sum_v
    )
    return pl.Series(vwaap.ravel())

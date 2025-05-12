from lob_features.dispatch.config import ASK_VOLUMES, BID_VOLUMES

import numpy as np
import polars as pl


def hump(X: pl.DataFrame, levels=5) -> pl.Series:
    weighted_volumes = X.select(
        [list(reversed(BID_VOLUMES[:levels])) + ASK_VOLUMES[:levels]]
    ).to_numpy()
    hump = (np.argmax(weighted_volumes, axis=1) + 1) / (2 * levels) - 0.5
    return pl.Series(hump)

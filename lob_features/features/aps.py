from lob_features.dispatch.config import ASK_PRICES
from lob_features.features.vwaap import vwaap

import numpy as np
import polars as pl


def ask_price_stability(X: pl.DataFrame, levels: int = 5) -> pl.Series:
    _vwaap = vwaap(X, levels).to_numpy()
    ask_prices = X.select(ASK_PRICES[:levels]).to_numpy() - _vwaap.reshape(-1, 1)
    aps = np.sqrt(np.sum(np.square(ask_prices), axis=1) / levels)
    return pl.Series(aps)

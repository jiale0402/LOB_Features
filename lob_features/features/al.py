from lob_features.dispatch.config import ASK_PRICES, ASK_VOLUMES

import numpy as np
import polars as pl


def ask_liquidity(X: pl.DataFrame, levels: int = 5) -> pl.Series:
    ask_vols = X.select(ASK_VOLUMES[:levels]).to_numpy()
    ask_prices = X.select(ASK_PRICES[:levels]).to_numpy()
    with np.errstate(invalid="ignore"):
        al = np.nanmean(ask_vols / ask_prices, axis=1)
    return pl.Series(al)

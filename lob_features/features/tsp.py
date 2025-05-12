from lob_features.dispatch.config import ASK_VOLUMES, BID_VOLUMES
from lob_features.features.vwaap import vwaap
from lob_features.features.vwabp import vwabp

import polars as pl


def total_spread(X: pl.DataFrame) -> pl.Series:
    vwabp_arr = vwabp(X).to_numpy()
    vwaap_arr = vwaap(X).to_numpy()
    ask_volumes = X.select(ASK_VOLUMES).to_numpy()
    bid_volumes = X.select(BID_VOLUMES).to_numpy()
    vol_sum = (ask_volumes + bid_volumes).sum(axis=1)
    ask_weights = ask_volumes.sum(axis=1) / vol_sum
    bid_weights = 1 - ask_weights
    return pl.Series((ask_weights * vwabp_arr + bid_weights * vwaap_arr).ravel())

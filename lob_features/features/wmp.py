from lob_features.features.vwaap import vwaap
from lob_features.features.vwabp import vwabp

import polars as pl


def weighted_midprice(X: pl.DataFrame) -> pl.Series:
    vwabp_arr = vwabp(X).to_numpy()
    vwaap_arr = vwaap(X).to_numpy()
    return pl.Series((vwabp_arr + vwaap_arr) / 2)

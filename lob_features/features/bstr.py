from lob_features.dispatch.config import ASK_VOLUMES, BID_VOLUMES
from lob_features.features.vwabp import vwabp

import numpy as np
import polars as pl


def bid_strength(X: pl.DataFrame, levels: int = 5, ema_steps: int = 10) -> pl.Series:
    vwabp_arr = vwabp(X).ewm_mean(span=ema_steps, adjust=False).to_numpy()
    ask_vol_sum = pl.Series(X.select(ASK_VOLUMES[:levels]).to_numpy().sum(axis=1))
    bid_vol_sum = pl.Series(X.select(BID_VOLUMES[:levels]).to_numpy().sum(axis=1))

    with np.errstate(divide="ignore", invalid="ignore"):
        bid_ask_vol_ratio = (
            pl.Series(
                np.nan_to_num(
                    ask_vol_sum.to_numpy() / bid_vol_sum.to_numpy(), nan=1.0, posinf=1.0, neginf=1.0
                )
            )
            .ewm_mean(span=ema_steps, adjust=False)
            .to_numpy()
        )

    return pl.Series(
        name="bid_strength", values=(vwabp_arr + bid_ask_vol_ratio + bid_vol_sum)
    ).cast(pl.Float32)

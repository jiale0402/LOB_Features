from lob_features.dispatch.config import ASK_VOLUMES, BID_VOLUMES
from lob_features.features.vwaap import vwaap

import numpy as np
import polars as pl


def ask_strength(X: pl.DataFrame, levels: int = 5, ema_steps: int = 10) -> pl.Series:
    vwaap_arr = vwaap(X).ewm_mean(span=ema_steps, adjust=False).to_numpy()
    ask_vol_sum = pl.Series(X.select(ASK_VOLUMES[:levels]).to_numpy().sum(axis=1))
    bid_vol_sum = pl.Series(X.select(BID_VOLUMES[:levels]).to_numpy().sum(axis=1))

    with np.errstate(divide="ignore", invalid="ignore"):
        ask_bid_vol_ratio = (
            pl.Series(
                np.nan_to_num(
                    ask_vol_sum.to_numpy() / bid_vol_sum.to_numpy(), nan=1.0, posinf=1.0, neginf=1.0
                )
            )
            .ewm_mean(span=ema_steps, adjust=False)
            .to_numpy()
        )

    return pl.Series(
        name="ask_strength", values=(vwaap_arr + ask_bid_vol_ratio + ask_vol_sum)
    ).cast(pl.Float32)

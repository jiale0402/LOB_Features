from lob_features.dispatch.config import ASK_PRICES, ASK_VOLUMES, BID_PRICES, BID_VOLUMES
from lob_features.features.sprd import get_ticksize

import numpy as np
import polars as pl


def volume_order_imbalance(X: pl.DataFrame, window: int) -> pl.Series:
    tol = get_ticksize(X) / 5
    curr_bid_price = X[BID_PRICES[0]].to_numpy()
    curr_bid_vol = X[BID_VOLUMES[0]].to_numpy()
    curr_ask_price = X[ASK_PRICES[0]].to_numpy()
    curr_ask_vol = X[ASK_VOLUMES[0]].to_numpy()
    prev_bid_price = X[BID_PRICES[0]].shift(window).to_numpy()
    prev_bid_vol = X[BID_VOLUMES[0]].shift(window).to_numpy()
    prev_ask_price = X[ASK_PRICES[0]].shift(window).to_numpy()
    prev_ask_vol = X[ASK_VOLUMES[0]].shift(window).to_numpy()
    delta_vtb = np.where(
        np.abs(curr_bid_price - prev_bid_price) < tol,
        curr_bid_vol - prev_bid_vol,
        np.where(curr_bid_price < prev_bid_price, 0, curr_bid_vol),
    )
    delta_vta = np.where(
        np.abs(curr_ask_price - prev_ask_price) < tol,
        curr_ask_vol - prev_ask_vol,
        np.where(curr_ask_price < prev_ask_price, curr_ask_vol, 0),
    )
    voi = delta_vtb - delta_vta
    return pl.Series(name="voi", values=voi.astype(np.float32))

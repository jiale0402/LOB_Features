from lob_features.dispatch.config import ASK_PRICES, BID_PRICES
from lob_features.features.sprd import get_ticksize

import numpy as np
import polars as pl

from pathlib import Path

ACTION_SIZE = 100.0


def backtest(
    symbol: str,
    data_dir: Path,
    predictions: np.ndarray,
    fees: float = 0.00018,
) -> list[float]:
    """
    Run a simple taker-only backtest. The rules are: if
    prediction[t] > spread + fees → long; if prediction[t] < -(spread + fees) → short; else flat.
    Each action is capped at 100 dollars.
    """
    depths = (
        pl.read_parquet(data_dir / "cleaned_datasets" / f"{symbol}.parquet")
        .select(ASK_PRICES[:1] + BID_PRICES[:1])
        .to_numpy()
    )
    n_steps = depths.shape[0]
    assert n_steps == predictions.shape[0], "Depths and predictions must align"
    position_hist = np.zeros(n_steps)
    pnl_hist = np.zeros(n_steps)
    curr_posn = 0.0
    curr_cash = 0.0
    ticksize = get_ticksize(depths)

    for t in range(n_steps):
        ask, bid = depths[t]
        mid = 0.5 * (ask + bid)
        expected_edge = predictions[t] * mid
        if expected_edge > ticksize + fees:
            curr_cash -= (1 + fees) * ACTION_SIZE * ask
            curr_posn += ACTION_SIZE / ask
        elif expected_edge < -(ticksize + fees):
            curr_cash += (1 - fees) * ACTION_SIZE * bid
            curr_posn -= ACTION_SIZE / bid
        pnl_hist[t] = curr_cash + curr_posn * mid
        position_hist[t] = curr_posn
    return pnl_hist.tolist()

import numpy as np
import polars as pl


def realized_returns(X: pl.DataFrame, window: int) -> pl.Series:
    midprices = X["mid_price"].to_numpy()
    prev_midprices = X["mid_price"].shift(window, fill_value=np.nan).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.log(midprices) - np.log(prev_midprices)
        ret = np.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)
    return pl.Series(name=f"ret-{window}", values=ret).cast(pl.Float32)

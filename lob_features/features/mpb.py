import numpy as np
import polars as pl

EPS = 1e-6


def mid_price_basis(X: pl.DataFrame, window: int) -> pl.Series:
    rolling_quantities = (
        X["quantity"].rolling_sum(window_size=window, min_samples=window).to_numpy()
    )
    rolling_dollar_vol = X["volume"].rolling_sum(window_size=window, min_samples=window).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = np.where(
            rolling_quantities < EPS,
            np.nan,
            rolling_dollar_vol / rolling_quantities,
        )
    vwap_windowed = (
        pl.DataFrame({"vwap": vwap})
        .with_columns(
            pl.when(pl.col("vwap").is_nan()).then(None).otherwise(pl.col("vwap")).alias("vwap")
        )["vwap"]
        .fill_null(strategy="forward")
        .fill_null(strategy="backward")
        .to_numpy()
    )
    # notice the backward filling above is once again to avoid the NaN at the beginning, its
    # technically looking ahead but the impact should be trivial
    midprice = X["mid_price"]
    prev_midprice = X["mid_price"].shift(window).to_numpy()
    mid = (midprice + prev_midprice) / 2
    return pl.Series(name=f"mpb-{window}", values=mid - vwap_windowed).cast(pl.Float32)

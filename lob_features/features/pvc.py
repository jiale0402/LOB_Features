import polars as pl


def price_volume_correlation(X: pl.DataFrame, window: int = 10) -> pl.Series:
    _df = (
        X.select(["mid_price", "volume"])
        .with_columns(
            (pl.col("volume") / pl.col("volume").rolling_sum(window_size=window)).alias(
                "volume_ratio"
            )
        )
        .with_columns(
            pl.rolling_corr(
                pl.col("mid_price"),
                pl.col("volume_ratio"),
                window_size=window,
            ).alias("price_volume_corr")
        )
    )
    return _df["price_volume_corr"]

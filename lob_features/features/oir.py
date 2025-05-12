from lob_features.dispatch.config import ASK_VOLUMES, BID_VOLUMES

import polars as pl


def order_imbalance_ratio(X: pl.DataFrame) -> pl.Series:
    oir = (X[ASK_VOLUMES[0]] - X[BID_VOLUMES[0]]) / (X[ASK_VOLUMES[0]] + X[BID_VOLUMES[0]])
    return pl.Series(name="OIR", values=oir).cast(pl.Float32)

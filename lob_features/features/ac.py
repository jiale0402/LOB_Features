from lob_features.dispatch.config import ASK_VOLUMES

import numpy as np
import polars as pl


def ask_concentration(X: pl.DataFrame, levels=5) -> pl.Series:
    ask_vols = X.select(ASK_VOLUMES[:levels]).to_numpy()
    max_vol = np.max(ask_vols, axis=1)
    tot_vol = np.sum(ask_vols, axis=1)
    ac = np.nan_to_num(max_vol / tot_vol, nan=0, posinf=0, neginf=0)
    return pl.Series(ac)

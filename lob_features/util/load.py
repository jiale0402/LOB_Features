import numpy as np
import polars as pl

from pathlib import Path

import os


def load_data(feature_dir: Path, features: list[str], symbol: str) -> pl.DataFrame:
    all_symbol_features = os.listdir(feature_dir)
    all_symbol_features = [f for f in all_symbol_features if f.startswith(symbol)]
    data: dict[str, np.ndarray] = {}
    for feature in features:
        if f"{symbol}-{feature}.parquet" not in all_symbol_features:
            raise ValueError(f"Feature {feature} not found for symbol {symbol}")
        data[feature] = pl.read_parquet(feature_dir / f"{symbol}-{feature}.parquet").to_numpy()
    data = pl.from_dict(data)
    return data

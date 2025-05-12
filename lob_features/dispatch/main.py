from lob_features.dispatch.catalog import NAME_TO_FEATURE
from lob_features.dispatch.config import ASK_PRICES, ASK_VOLUMES, BID_PRICES, BID_VOLUMES

from tqdm import tqdm

import numpy as np
import pandas as pd
import polars as pl

from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path

import logging
import os

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.NOTSET)
COLS_TO_USE = (
    ASK_VOLUMES
    + BID_VOLUMES
    + BID_PRICES
    + ASK_PRICES
    + ["open", "high", "low", "close", "volume", "quantity", "vwap", "mid_price"]
)


def _write_to_disk(
    values: np.ndarray,
    symbol: str,
    feature_name: str,
    output_dir: Path,
) -> None:
    pd.DataFrame({feature_name: values}).to_parquet(
        output_dir / f"{symbol}-{feature_name}.parquet", compression="zstd"
    )


def compute_features(
    dataset_path: Path,
    output_dir: Path,
    symbol: str,
    n_workers: int = 1,
    force_recompute: bool = False,
) -> None:
    ds = pl.scan_parquet(dataset_path).select(COLS_TO_USE).collect()
    futs: list[Future] = []
    if n_workers == -1:
        LOGGER.warning("No n_workers set, using all available cores")
        n_workers = max(len(NAME_TO_FEATURE), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for name, (fn, kwargs) in tqdm(
            NAME_TO_FEATURE.items(), desc=f"Computing features for {symbol}"
        ):
            out_f = output_dir / f"{symbol}-{name}.parquet"
            if out_f.exists() and not force_recompute:
                LOGGER.info("Feature %s already exists → skip", name)
                continue
            values = fn(ds, **kwargs)  # type: ignore
            futs.append(executor.submit(_write_to_disk, values, symbol, name, output_dir))
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                LOGGER.error("Error processing feature %s: %s", f, e)
                raise e


def driver(symbol_file: Path, out_dir: Path, n_workers: int, force_recompute: bool) -> None:
    symbol = symbol_file.stem
    try:
        LOGGER.info("Processing %s", symbol)
        compute_features(
            symbol_file,
            out_dir,
            symbol,
            n_workers=n_workers,
            force_recompute=force_recompute,
        )
    except Exception as e:
        LOGGER.error("Error processing %s: %s", symbol, e)
        raise e


def main() -> None:
    import argparse
    import multiprocessing

    multiprocessing.set_start_method("forkserver", force=True)

    p = argparse.ArgumentParser("Dispatch LOB features")
    p.add_argument("--warehouse", required=True, help="Directory containing cleaned up datasets.")
    p.add_argument("--n_workers", type=int, default=-1)
    p.add_argument("--force_recompute", action="store_true")
    args = p.parse_args()

    warehouse = Path(args.warehouse)
    assert warehouse.exists()
    data_dir = warehouse / "cleaned_datasets"
    out_dir = warehouse / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    force_recompute = args.force_recompute
    files = list(data_dir.glob("*.parquet"))
    LOGGER.info("Found %d file(s)", len(files))

    for f in files:
        driver(f, out_dir, args.n_workers, force_recompute)


if __name__ == "__main__":
    main()

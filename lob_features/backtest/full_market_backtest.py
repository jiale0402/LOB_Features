from lob_features.backtest.backtest import backtest
import polars as pl
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

from concurrent.futures import ProcessPoolExecutor, as_completed

def full_backtest(data_directory: Path, fees: float, predictions: dict[str, np.ndarray], n_workers: int, save_fig: bool = True) -> None:
    data_directory = Path(data_directory) / "cleaned_datasets"
    symbols = [f.stem.split("-")[0] for f in data_directory.glob("*.parquet")]
    symbols = list(set(symbols))
    LOGGER.info("Found %d symbols", len(symbols))
    futures = []
    with ProcessPoolExecutor() as executor:
        for symbol in symbols:
            futures.append(executor.submit(backtest, symbol, data_directory, predictions[symbol], fees))
        for future in tqdm(as_completed(futures), total=len(futures), desc="Backtesting"):
            symbol = futures[future]
            try:
                future.result()
            except Exception as e:
                LOGGER.error("Error processing symbol %s: %s", symbol, e)
    
    if save_fig:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set(style="whitegrid")
        for symbol in symbols:
            plt.figure(figsize=(10, 6))
            plt.plot(futures[symbol], label=symbol)
            plt.title(f"Backtest results for {symbol}")
            plt.xlabel("Time")
            plt.ylabel("PnL")
            plt.legend()
            plt.savefig(f"{symbol}_backtest.png")
            plt.close()
    LOGGER.info("Backtest completed for all symbols")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Full market backtest")
    parser.add_argument("--data_directory", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--fees", type=float, default=0.00018, help="Transaction fees")
    parser.add_argument("--n_workers", type=int, default=4, help="Number of workers for parallel processing")
    parser.add_argument("--save_fig", action="store_true", help="Save figures of the backtest results")
    args = parser.parse_args()

    # Example predictions dictionary, to be replaced with actual cached prediction feeds
    predictions = {symbol: np.random.rand(1000) for symbol in ["AAPL", "GOOGL"]}

    full_backtest(args.data_directory, args.fees, predictions, args.n_workers, args.save_fig)
from lob_features.dispatch.config import OB_LEVELS
from lob_features.features.ac import ask_concentration
from lob_features.features.al import ask_liquidity
from lob_features.features.aps import ask_price_stability
from lob_features.features.asl import ask_slope
from lob_features.features.astr import ask_strength
from lob_features.features.bc import bid_concentration
from lob_features.features.bl import bid_liquidity
from lob_features.features.bps import bid_price_stability
from lob_features.features.bsl import bid_slope
from lob_features.features.bstr import bid_strength
from lob_features.features.hump import hump
from lob_features.features.mpb import mid_price_basis
from lob_features.features.oir import order_imbalance_ratio
from lob_features.features.oirk import order_imbalance_ratio_kurtosis
from lob_features.features.oirs import order_imbalance_ratio_skew
from lob_features.features.oirv import order_imbalance_ratio_volatility
from lob_features.features.pvc import price_volume_correlation
from lob_features.features.rr import realized_returns
from lob_features.features.rs import realized_return_skew
from lob_features.features.rv import realized_volatility
from lob_features.features.sprd import spread
from lob_features.features.tsp import total_spread
from lob_features.features.voi import volume_order_imbalance
from lob_features.features.wmp import weighted_midprice

NAME_TO_FEATURE = {
    "OIR": (order_imbalance_ratio, {}),
    "VOI": (volume_order_imbalance, {"window": 1}),
    "BPS": (bid_price_stability, {"levels": OB_LEVELS}),
    "APS": (ask_price_stability, {"levels": OB_LEVELS}),
    "AC": (ask_concentration, {"levels": OB_LEVELS}),
    "BC": (bid_concentration, {"levels": OB_LEVELS}),
    "MPB": (mid_price_basis, {"window": 1}),
    "BL": (bid_liquidity, {"levels": OB_LEVELS}),
    "AL": (ask_liquidity, {"levels": OB_LEVELS}),
    "RR": (realized_returns, {"window": 30}),
    "RV": (realized_volatility, {"window": 30}),
    "RS": (realized_return_skew, {"window": 30}),
    "ASTR": (ask_strength, {"levels": 5}),
    "BSTR": (bid_strength, {"levels": 5}),
    "HUMP": (hump, {"levels": 5}),
    "ASL": (ask_slope, {"levels": 5}),
    "BSL": (bid_slope, {"levels": 5}),
    "OIRS": (order_imbalance_ratio_skew, {"window": 30}),
    "OIRV": (order_imbalance_ratio_volatility, {"window": 30}),
    "TSP": (total_spread, {}),
    "OIRK": (order_imbalance_ratio_kurtosis, {"window": 30}),
    "SPRD": (spread, {}),
    "WMP": (weighted_midprice, {}),
    "PVC": (price_volume_correlation, {"window": 30}),
}

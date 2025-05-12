OB_LEVELS = 5
BID_PRICES = [f"bids[{i}].price" for i in range(OB_LEVELS)]
ASK_PRICES = [f"asks[{i}].price" for i in range(OB_LEVELS)]
BID_VOLUMES = [f"bids[{i}].amount" for i in range(OB_LEVELS)]
ASK_VOLUMES = [f"asks[{i}].amount" for i in range(OB_LEVELS)]

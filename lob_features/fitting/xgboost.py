from xgboost import XGBRegressor


def fit_xgboost(
    X_train,
    y_train,
    X_test,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_child_weight=1,
    gamma=0,
    subsample=1,
    colsample_bytree=1,
    device="cuda",
):
    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        gamma=gamma,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        device=device,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

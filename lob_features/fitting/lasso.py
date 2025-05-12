from sklearn.linear_model import Lasso


def fit_lasso(X_train, y_train, X_test, alpha=1.0):
    model = Lasso(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

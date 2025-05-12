from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import numpy as np


def train_test_split(data: np.ndarray, test_size: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    n_samples = data.shape[0]
    n_test_samples = int(n_samples * test_size)
    return data[:-n_test_samples], data[-n_test_samples:]


def mean_impute_and_scale(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    trainX, testX = train_test_split(X)
    trainy, testy = train_test_split(y)
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    trainX = imputer.fit_transform(trainX)
    trainX = scaler.fit_transform(trainX)
    testX = imputer.transform(testX)
    testX = scaler.transform(testX)
    return trainX, testX, trainy, testy

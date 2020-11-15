import numpy as np

EPS = 1e-06


def binary_cross_entropy(y_real: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return -(y_real * np.log(y_pred + EPS) + (1 - y_real) * np.log(1 - y_pred + EPS))


def binary_cross_entropy_back(y_real: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return - y_real / y_pred + (1 - y_real) / (1 - y_pred + EPS)

import numpy as np

EPS = 1e-06


class Loss:
    def __init__(self):
        pass

    def __call__(self, y_real: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return self.forward(y_real, y_pred)

    def forward(self, y_real: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Forward pass of this loss function has not been implemented"
        )

    def backward(self, y_real: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Backward pass of this loss function has not been implemented"
        )


class BinaryCrossEntropy(Loss):
    def forward(self, y_real: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return -(
            y_real * np.log(y_pred + EPS) + (1 - y_real) * np.log(1 - y_pred + EPS)
        )

    def backward(self, y_real: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return -y_real / y_pred + (1 - y_real) / (1 - y_pred + EPS)

import numpy as np

EPS = 1e-07


class Loss:
    """Generic class to define a loss function"""

    def __init__(self):
        pass

    def __call__(self, y_real: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return self.forward(y_real, y_pred)

    def forward(self, y_real: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculates and returns the loss value."""
        raise NotImplementedError(
            "Forward pass of this loss function has not been implemented"
        )

    def backward(self, y_real: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculates the gradient of the output predictions"""
        raise NotImplementedError(
            "Backward pass of this loss function has not been implemented"
        )


class BinaryCrossEntropy(Loss):
    """Binary Cross Entropy loss used for binary classification

    Calculates negative log likelihood i.e., the entropy between the real
    distribution and the predicted distribution.
    """

    def forward(self, y_real: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return -(
            y_real * np.log(y_pred + EPS) + (1 - y_real) * np.log(1 - y_pred + EPS)
        )

    def backward(self, y_real: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return -(y_real / (y_pred + EPS)) + (1 - y_real) / (1 - y_pred + EPS)


def norm(a):
    return np.sqrt(np.einsum('ij,ij->i', a, a))


class CosineLoss(Loss):
    def forward(self, y_real: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # XY_dot = (y_real * y_pred).sum()
        # X_norm = np.sqrt((y_real*y_real).sum())
        # Y_norm = np.sqrt((y_pred*y_pred).sum())
        # similarity = XY_dot / (X_norm * Y_norm)
        # d_loss = 1 - similarity
        # return d_loss
            # Add a small constant to avoid 0 vectors
        yh = y_pred + 1e-8
        y = y_real + 1e-8
        # https://math.stackexchange.com/questions/1923613/partial-derivative-of-cosine-similarity
        # xp = get_array_module(yh)
        norm_yh = np.linalg.norm(yh, axis=1, keepdims=True)
        # print(norm_yh.shape)
        norm_y = np.linalg.norm(y, axis=1, keepdims=True)
        # print(norm_y.shape)
        mul_norms = norm_yh * norm_y
        # print(mul_norms.shape)
        cosine = (yh * y).sum(axis=1, keepdims=True) / mul_norms + EPS
        # print(cosine.shape)
        # print(cosine)
        # d_yh = (y / mul_norms) - (cosine * (yh / norm_yh**2))
        loss = np.abs(1-cosine).sum()
        # print(loss)
        return loss

    def backward(self, y_real: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # XY_dot = (y_real * y_pred).sum()
        # X_norm = np.sqrt((y_real*y_real).sum())
        # Y_norm = np.sqrt((y_pred*y_pred).sum())
        # similarity = XY_dot / ((X_norm * Y_norm) + EPS)
        # d_loss = 1 - similarity

        # d_XY_dot = -d_loss / similarity
        # d_X_norm = d_loss * XY_dot / ((similarity ** 2) + EPS)
        # dX = d_XY_dot * y_pred + d_X_norm * (2.0 * X_norm)
        # # dX = (X / (X_norm*Y_norm)) - (similarity * (X / X_norm**2))

            # Add a small constant to avoid 0 vectors
        yh = y_pred + 1e-8
        y = y_real + 1e-8
        # https://math.stackexchange.com/questions/1923613/partial-derivative-of-cosine-similarity
        # xp = get_array_module(yh)
        norm_yh = np.linalg.norm(yh, axis=1, keepdims=True)
        norm_y = np.linalg.norm(y, axis=1, keepdims=True)
        mul_norms = norm_yh * norm_y
        cosine = (yh * y).sum(axis=1, keepdims=True) / mul_norms + EPS
        d_yh = (y / mul_norms) - (cosine * (yh / norm_yh**2))
        # loss = np.abs(1-cosine).sum()
        return -d_yh


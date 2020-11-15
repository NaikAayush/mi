import numpy as np


class Activation:
    def __init__(self):
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Forward pass not implemented")

    def backward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Backward pass not implemented")

    def __call__(self, x: np.ndarray, back: bool = False) -> np.ndarray:
        if back:
            return self.backward(x)
        return self.forward(x)


class Identity(Activation):
    def forward(self, x):
        return x

    def backward(self, x):
        return np.ones_like(x)


class ReLU(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        z = x.copy()
        z[z < 0] = 0
        return z

    def backward(self, x: np.ndarray) -> np.ndarray:
        x_grad = x.copy()
        x_grad[x_grad < 0] = 0
        x_grad[x_grad > 0] = 1
        return x_grad


class Sigmoid(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def backward(self, x: np.ndarray) -> np.ndarray:
        act = self(x)
        return act * (1 - act)


class LeakyReLU(Activation):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        z = x.copy()
        z[z < 0] *= self.alpha
        return z

    def backward(self, x: np.ndarray) -> np.ndarray:
        x_g = x.copy()
        x_g[x_g <= 0] = self.alpha
        x_g[x_g > 0] = 1
        return x_g

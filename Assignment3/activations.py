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
    def forward(self, x):
        x[x < 0] = 0
        return x

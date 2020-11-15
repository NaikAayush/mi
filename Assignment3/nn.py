import numpy as np
from activations import Activation, Identity

class Dense:
    def __init__(self,
                 in_len: int,
                 out_len: int,
                 activation: Activation=Identity):
        self.in_len = in_len
        self.out_len = out_len

        self.w = np.random.randn(out_len, in_len)
        self.b = np.random.randn(out_len, 1)
        self.activation = activation()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        assert x.shape[0] == self.in_len
        if (len(x.shape) < 2):
            x = x.reshape((x.shape[0], 1))
        return self.activation(np.einsum("ij,jk->ik", self.w, x) + self.b)

    def backward(self, dA: np.ndarray, z: np.ndarray, A_prev: np.ndarray) -> np.ndarray:
        dz = dA * self.activation(z, back=True)
        dW = np.einsum("ij,kj->ik", dz, A_prev)
        # TODO
        # db = ...
        # dA = ...
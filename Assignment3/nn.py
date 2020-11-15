import numpy as np
from activations import Activation, Identity


class Dense:
    """A fully connected dense layer"""
    def __init__(self,
                 in_len: int,
                 out_len: int,
                 activation: Activation = Identity):
        self.in_len = in_len
        self.out_len = out_len

        self.w = np.random.randn(out_len, in_len)
        self.b = np.random.randn(out_len)
        self.activation = activation()

        self.dw = None
        self.db = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        assert x.shape[1] == self.in_len, \
               f"Input must be 2D array with size (m, {self.in_len})"
        if len(x.shape) < 2:
            x = x.reshape((1, x.shape[0]))
        # Tested
        return self.activation(np.einsum("oi,mi->mo", self.w, x) + self.b)

    def backward(self, dA: np.ndarray, z: np.ndarray, A_prev: np.ndarray) -> np.ndarray:
        """Backward pass"""
        # All tested
        dz = dA * self.activation(z, back=True)
        self.dw = np.einsum("mo,mi->oi", dz, A_prev) / dz.shape[0]
        self.db = np.einsum("mo->o", dz) / dz.shape[0]
        dA_prev = np.einsum("oi,mo->mi", self.w, dz)
        return dA_prev

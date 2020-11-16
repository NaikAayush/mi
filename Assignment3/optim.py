from typing import Tuple
import numpy as np

from nn import Param


class Optimizer:
    def __init__(self, params: Tuple[Param, ...]):
        self.params = params

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.fill(0.0)

    def step(self):
        pass


class SGD(Optimizer):
    def __init__(
        self, lr: float, params: Tuple[Param, ...], l2_lambda=None, beta1=None
    ):
        super().__init__(params=params)
        self.lr = lr

        self.l2_lambda = l2_lambda
        self.l2_reg = l2_lambda is not None

        self.momentum = beta1 is not None
        self.beta1 = beta1

        velcs = []
        if self.momentum:
            for param in params:
                velcs.append(np.zeros_like(param.data))
        self.velcs = tuple(velcs)

    def step(self):
        for ind, param in enumerate(self.params):
            weight_scale = 1
            if self.l2_reg:
                weight_scale = 1 - self.lr * self.l2_lambda

            grad = param.grad
            if self.momentum:
                grad = self.beta1 * self.velcs[ind] + (1 - self.beta1) * param.grad

            param.data = weight_scale * param.data - self.lr * grad

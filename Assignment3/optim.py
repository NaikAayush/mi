from typing import Tuple

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
    def __init__(self, lr: float, params: Tuple[Param, ...]):
        super().__init__(params=params)
        self.lr = lr

    def step(self):
        for param in self.params:
            param.data = param.data - self.lr * param.grad

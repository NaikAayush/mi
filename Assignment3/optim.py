from typing import Tuple

from nn import Param


class SGD:
    def __init__(self, lr: float, params: Tuple[Param, ...]):
        self.lr = lr
        self.params = params

    def step(self):
        for param in self.params:
            param.data = param.data - self.lr * param.grad

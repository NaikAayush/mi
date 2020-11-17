"""
Design of a Neural Network from scratch

*************<IMP>*************
Mention hyperparameters used and describe functionality in detail in this space
- carries 1 mark
"""
from typing import Iterator, Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import os

### ACTIVATIONS ###


class Activation:
    """Generic class to define an activation function.

    Usage:

    ```
    a = Activation()
    y = a(x)
    y_grad = a(x, back=True)
    ```
    """

    def __init__(self):
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass of the activation function

        :param x: inputs
        """
        raise NotImplementedError("Forward pass not implemented")

    def backward(self, x: np.ndarray) -> np.ndarray:
        """Backward pass of the activation function

        :param x: inputs
        :returns: Gradient of the inputs
        """
        raise NotImplementedError("Backward pass not implemented")

    def __call__(self, x: np.ndarray, back: bool = False) -> np.ndarray:
        if back:
            return self.backward(x)
        return self.forward(x)


class Identity(Activation):
    """Activation function that returns the input

    y = x
    """

    def forward(self, x):
        return x

    def backward(self, x):
        return np.ones_like(x)


class ReLU(Activation):
    """Rectified Linear Unit activation function

    y = 0 for x < 0, x for x >= 0
    """

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
    """Sigmoid function, specifically the logistic function

    y = 1 / (1 + e^-x)
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def backward(self, x: np.ndarray) -> np.ndarray:
        act = self(x)
        return act * (1 - act)


class LeakyReLU(Activation):
    """Variation of ReLU that has a small constant slope on the negative side

    y = alpha*x for x < 0, x for x >= 0

    :param alpha: The negative slope parameter
    """

    def __init__(self, alpha: float = 0.01):
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


####### END #######

### LOSSES ####
EPS = 1e-06


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


##### END #####


### NEURAL NETWORK ###


class Param:
    """Stores a trainable parameter and its gradient

    :param data: value to store
    """

    def __init__(self, data: np.ndarray):
        self.data = data
        self.grad = None


class Module:
    """A general module to define a layer or multiple layers of a neural network"""

    def __init__(self):
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass

        :param x: Input for forward pass
        """
        raise NotImplementedError("This module does not have a forward pass")

    def backward(self, dA: np.ndarray, *args):
        """Backward pass. Calculates and stores gradients.

        :param dA: gradient of the output of this module
        """
        raise NotImplementedError("This module does not have a backward pass")

    @staticmethod
    def _add_param(params: List, attr):
        if isinstance(attr, Param):
            params.append(attr)
        elif isinstance(attr, Module):
            params.extend(attr.parameters())

    def parameters(self) -> Tuple[Param, ...]:
        """Returns all trainable parameters"""
        params = []
        for attr in vars(self).values():
            self._add_param(params, attr)
            # check if attr is iterablel
            try:
                iterator = iter(attr)
                for x in iterator:
                    self._add_param(params, x)
            except TypeError:
                pass
        return tuple(params)


class Dense(Module):
    """A fully connected dense layer

    :param in_len: Size of input
    :type in_len: int
    :param out_len: Size of output
    :type out_len: int
    :param activation: Activation function. Default is identity.
    :param xavier_init: Whether to use Xavier initialisation.
        The weights are multiplied by sqrt(2 / in_len)
    """

    def __init__(
        self,
        in_len: int,
        out_len: int,
        activation: Activation = Identity,
        xavier_init: bool = False,
    ):
        super().__init__()
        self.in_len = in_len
        self.out_len = out_len

        self.w = Param(np.random.randn(out_len, in_len))
        if xavier_init:
            self.w.data *= np.sqrt(2 / in_len)
        self.b = Param(np.random.randn(out_len))
        self.activation = activation()

        self.z = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        assert (
            x.shape[1] == self.in_len
        ), f"Input must be 2D array with size (m, {self.in_len})"
        if len(x.shape) < 2:
            x = x.reshape((1, x.shape[0]))
        # Tested
        self.z = np.einsum("oi,mi->mo", self.w.data, x) + self.b.data
        return self.activation(self.z)

    # pylint: disable=arguments-differ
    def backward(self, dA: np.ndarray, A_prev: np.ndarray) -> np.ndarray:
        """Backward pass"""
        # All tested
        dz = dA * self.activation(self.z, back=True)
        self.w.grad = np.einsum("mo,mi->oi", dz, A_prev) / dz.shape[0]
        self.b.grad = np.einsum("mo->o", dz) / dz.shape[0]
        dA_prev = np.einsum("oi,mo->mi", self.w.data, dz)
        return dA_prev


class Sequential(Module):
    """Sequential layers to stack multiple layers one after another.
    All layers must be passed as different parameters.
    """

    def __init__(self, *modules: Iterator[Module]):
        super().__init__()
        self.modules = tuple(modules)
        self.outs = []

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.outs = [x]
        for module in self.modules:
            a = module(self.outs[-1])
            self.outs.append(a)
        return self.outs[-1]

    # pylint: disable=arguments-differ
    def backward(self, dA: np.ndarray):
        for module, prev_out in zip(reversed(self.modules), reversed(self.outs[:-1])):
            dA = module.backward(dA, prev_out)


######## END #########


### OPTIMIZER ###


class Optimizer:
    """Generic class to define an optimizer for gradient descent

    :param params: trainable parameters to optimize
    """

    def __init__(self, params: Tuple[Param, ...]):
        self.params = params

    def zero_grad(self):
        """Zeros out all the gradients.
        To be run before doing a backward pass
        """
        for param in self.params:
            if param.grad is not None:
                param.grad.fill(0.0)

    def step(self):
        """Calculates one step of the optimization process
        and updates the parameters
        """


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer with support for L2 regularization
    and momentum

    :param params: trainable parameters to optimize
    :param l2_lambda: (optional) lambda parameter of L2 regularization. If not given,
        L2 regularization is not used (default).
    :param beta1: (optional) beta parameter of momentum
        (exponentially averaged velocity).
        If not given, momentum is not used.
    """

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


##### END #######

# For reproducibility
np.random.seed(53)


class NN:

    """ X and Y are dataframes """

    def __init__(self):
        self.model = Sequential(
            Dense(9, 20, ReLU, xavier_init=True),
            Dense(20, 20, LeakyReLU, xavier_init=True),
            Dense(20, 1, Sigmoid),
        )
        self.optimizer = SGD(0.05, self.model.parameters(), l2_lambda=0.01)
        self.loss_fun = BinaryCrossEntropy()

    def fit(self, X, Y):
        """
        Function that trains the neural network by taking
        x_train and y_train samples as input
        """

        steps = 1000
        for step in range(steps):
            self.optimizer.zero_grad()
            a = self.model(X)

            if step % 100 == 0:
                loss = self.loss_fun(Y, a)
                print(f"Loss at step {step}: {loss.mean()}")

            dA = self.loss_fun.backward(Y, a)
            self.model.backward(dA)

            self.optimizer.step()

    def accuracies(self, X_train, X_test, y_train, y_test):
        train_pred = self.predict(X_train)
        train_acc = sum(train_pred == y_train) / len(y_train)
        train_acc = train_acc[0]
        print(f"\nTrain accuracy: {train_acc*100:.03f}%")

        test_pred = self.predict(X_test)
        test_acc = sum(test_pred == y_test) / len(y_test)
        test_acc = test_acc[0]
        print(f"Test accuracy: {test_acc*100:.03f}%\n")

        return train_acc, test_acc

    def predict(self, X):

        """
        The predict function performs a simple feed forward of weights
        and outputs yhat values

        yhat is a list of the predicted value for df X
        """

        a = self.model(X)
        y_hat = np.around(a)
        return y_hat

    def CM(self, y_test, y_test_obs):
        """
        Prints confusion matrix
        y_test is list of y values in the test dataset
        y_test_obs is list of y values predicted by the model

        """

        for i in range(len(y_test_obs)):
            if y_test_obs[i] > 0.6:
                y_test_obs[i] = 1
            else:
                y_test_obs[i] = 0

        cm = [[0, 0], [0, 0]]
        fp = 0
        fn = 0
        tp = 0
        tn = 0

        # print(y_test, y_test_obs)
        for i in range(len(y_test)):
            if y_test[i] == 1 and y_test_obs[i] == 1:
                tp = tp + 1
            if y_test[i] == 0 and y_test_obs[i] == 0:
                tn = tn + 1
            if y_test[i] == 1 and y_test_obs[i] == 0:
                fp = fp + 1
            if y_test[i] == 0 and y_test_obs[i] == 1:
                fn = fn + 1
        cm[0][0] = tn
        cm[0][1] = fp
        cm[1][0] = fn
        cm[1][1] = tp

        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = (2 * p * r) / (p + r)

        print("Confusion Matrix : ")
        print(cm)
        print("\n")
        print(f"Precision : {p}")
        print(f"Recall : {r}")
        print(f"F1 SCORE : {f1}")


# df = pd.read_csv("./processed.csv")
dir_path = os.path.dirname(os.path.realpath(__file__))
df = pd.read_csv(os.path.join(dir_path, "../data/processed.csv"))

columns = list(df.columns)
columns.remove("Result")

X = df[columns].to_numpy()
y = df[["Result"]].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)

nn = NN()

nn.fit(X_train, y_train)

y_pred = nn.predict(X_test)

nn.accuracies(X_train, X_test, y_train, y_test)
nn.CM(y_test, y_pred)

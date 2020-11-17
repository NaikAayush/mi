import numpy as np

from nn import Dense, Sequential
from activations import LeakyReLU, ReLU, Sigmoid
from data import X_train, X_test, y_train, y_test
import losses
import optim

np.random.seed(94)

model = Sequential(
    Dense(9, 20, ReLU, xavier_init=True),
    Dense(20, 20, LeakyReLU, xavier_init=True),
    Dense(20, 1, Sigmoid),
)
optimizer = optim.SGD(0.05, model.parameters(), l2_lambda=0.01)
loss_fun = losses.BinaryCrossEntropy()

steps = 1000
for _ in range(steps):
    optimizer.zero_grad()
    a = model(X_train)

    loss = loss_fun(y_train, a)

    dA = loss_fun.backward(y_train, a)
    model.backward(dA)

    optimizer.step()

a = model(X_train)
y_pred = np.around(a)
tr_a = sum(y_pred == y_train) / len(y_train)

a = model(X_test)
y_pred = np.around(a)
te_a = sum(y_pred == y_test) / len(y_test)
print("Train accuracy: ", tr_a)
print("Test accuracy: ", te_a)


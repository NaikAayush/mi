import numpy as np
from tqdm.auto import tqdm

from nn import Dense, Sequential
from activations import ReLU, Sigmoid
from data import X_train, X_test, y_train, y_test
import losses
import optim

model = Sequential(Dense(9, 20, ReLU), Dense(20, 20, ReLU), Dense(20, 1, Sigmoid))
optimizer = optim.SGD(0.2, model.parameters())

t = tqdm(total=100)
for _ in range(100):
    optimizer.zero_grad()
    a = model(X_train)

    loss = losses.binary_cross_entropy(y_train, a)
    print("Loss: ", loss.mean())

    dA = losses.binary_cross_entropy_back(y_train, a)
    model.backward(dA)

    optimizer.step()
    t.set_postfix(
        loss=loss.mean(),
        w_mean=model.modules[0].w.data.mean(),
        dw_mean=model.modules[0].w.grad.mean(),
    )
    t.update()
t.close()

a = model(X_train)
y_pred = np.around(a)
print("Train accuracy: ", sum(y_pred == y_train) / len(y_train))

a = model(X_test)
y_pred = np.around(a)
print("Test accuracy: ", sum(y_pred == y_test) / len(y_test))

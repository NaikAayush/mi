import numpy as np
from tqdm.auto import tqdm

from nn import Dense, Sequential
from activations import LeakyReLU, ReLU, Sigmoid
from data import X_train, X_test, y_train, y_test
import losses
import optim



for i in range(100):
    np.random.seed(i)
    # print("Seed: ",i)

    model = Sequential(
        Dense(9, 20, ReLU, xavier_init=False),
        Dense(20, 20, LeakyReLU, xavier_init=False),
        Dense(20, 1, Sigmoid),
    )
    optimizer = optim.SGD(0.01, model.parameters())
    loss_fun = losses.BinaryCrossEntropy()

    steps = 2000
    # t = tqdm(total=steps)
    for _ in range(steps):
        optimizer.zero_grad()
        a = model(X_train)

        loss = loss_fun(y_train, a)

        dA = loss_fun.backward(y_train, a)
        model.backward(dA)

        optimizer.step()
        # t.set_postfix(
        #     loss=loss.mean(),
        #     w_mean=model.modules[0].w.data.mean(),
        #     dw_mean=model.modules[0].w.grad.mean(),
        # )
        # t.update()
    # t.close()

    a = model(X_train)
    y_pred = np.around(a)
    tr_a = sum(y_pred == y_train) / len(y_train)
    # print("Train accuracy: ", sum(y_pred == y_train) / len(y_train))

    a = model(X_test)
    y_pred = np.around(a)
    te_a = sum(y_pred == y_test) / len(y_test)
    # print("Test accuracy: ", sum(y_pred == y_test) / len(y_test))
    if tr_a and te_a > 0.85:
        print("Seed: ",i)
        print("Train accuracy: ", tr_a)
        print("Test accuracy: ", te_a)
        print(i,tr_a,te_a,sep=',',file=open('brr.csv','a'))
    
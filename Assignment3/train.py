from nn import Dense
from activations import ReLU, Sigmoid
from data import X, y
import losses

d1 = Dense(9, 20, ReLU)
d2 = Dense(20, 20, ReLU)
d3 = Dense(20, 1, Sigmoid)

z1, a1 = d1(X)
z2, a2 = d2(a1)
z3, a3 = d3(a2)

loss = losses.binary_cross_entropy(y, a3)
print(loss.mean())

dA3 = losses.binary_cross_entropy_back(y, a3)
dA2 = d3.backward(dA3, z3, a2)
dA1 = d2.backward(dA2, z2, a1)
dA0 = d1.backward(dA1, z1, X)


import numpy as np

from sigmoid import sigmoid


# X = np.array([1.0, 0.5])
# print(f"X: {X}")
# print()

# W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
# B1 = np.array([0.1, 0.2, 0.3])
# A1 = np.dot(X, W1) + B1
# Z1 = sigmoid(A1)
# print(f"W1: \n{W1}")
# print(f"B1: {B1}")
# print(f"A1: {A1}")
# print(f"Z1: {Z1}")
# print()

# W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
# B2 = np.array([0.1, 0.2])
# A2 = np.dot(Z1, W2) + B2
# Z2 = sigmoid(A2)
# print(f"W2: \n{W2}")
# print(f"B2: {B2}")
# print(f"A2: {A2}")
# print(f"Z2: {Z2}")
# print()


def identity_function(x: np.ndarray) -> np.ndarray:
    return x


# W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
# B3 = np.array([0.1, 0.2])
# A3 = np.dot(Z2, W3) + B3
# Y = identity_function(A3)  # 恒等関数を出力層のactivation functionとする (Y = A3)
# print(f"W3: \n{W3}")
# print(f"B3: {B3}")
# print(f"A3: {A3}")
# print(f"Y: {Y}")


def init_network():
    network = {}
    network["W1"] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network["b1"] = np.array([0.1, 0.2, 0.3])
    network["W2"] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network["b2"] = np.array([0.1, 0.2])
    network["W3"] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network["b3"] = np.array([0.1, 0.2])
    return network


def forward(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    return y


network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)

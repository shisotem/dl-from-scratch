import os, sys
import pickle
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data() -> tuple[np.ndarray, np.ndarray]:
    (_, _), (x_test, t_test) = load_mnist(
        normalize=True, flatten=True, one_hot_label=False
    )
    return x_test, t_test


def init_network() -> dict[str, np.ndarray]:
    with open(os.path.join(current_dir, "sample_weight.pkl"), "rb") as f:
        network: dict[str, np.ndarray] = pickle.load(f)

    return network


# Number of unit: 784 -> 50 -> 100 -> 10
def predict(network: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = x @ W1 + b1
    z1 = sigmoid(a1)
    a2 = z1 @ W2 + b2
    z2 = sigmoid(a2)
    a3 = z2 @ W3 + b3
    y = softmax(a3)

    return y


if __name__ == "__main__":
    x, t = get_data()
    network = init_network()

    batch_size = 100
    accuracy_cnt = 0
    for i in range(0, len(x), batch_size):
        x_batch = x[i : i + batch_size]
        y_batch = predict(network, x_batch)
        p_batch = np.argmax(y_batch, axis=1)  # 予測した0-9の数字のバッチ

        t_batch = t[i : i + batch_size]
        accuracy_cnt += np.sum(p_batch == t_batch)

    print(f"Accuracy: {accuracy_cnt / len(x)}")  # Accuracy: 0.9352 (10000件でtest)

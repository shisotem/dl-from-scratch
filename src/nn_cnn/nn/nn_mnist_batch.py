import os, sys
import pickle
import numpy as np

from sigmoid import sigmoid
from softmax import softmax

# MNISTデータセットのモジュールがあるディレクトリへのパスを追加
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from dataset.mnist import load_mnist
except ImportError:
    print("Error: Could not import 'load_mnist' from 'dataset.mnist'.")
    print(
        "Please ensure that the 'dataset' directory is in the parent directory of this script."
    )
    sys.exit(1)


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

    accuracy_cnt = 0
    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y)  # 予測した0-9の数字
        if p == t[i]:
            accuracy_cnt += 1

    print(f"Accuracy: {accuracy_cnt / len(x)}")  # Accuracy: 0.9352 (10000件でtest)

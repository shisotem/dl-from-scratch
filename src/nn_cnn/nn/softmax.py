import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x)
    y = exp_x / np.sum(exp_x)
    return y


if __name__ == "__main__":
    x = np.array([0.3, 2.9, 4.0])
    y = softmax(x)
    print(y)
    print(np.sum(y))

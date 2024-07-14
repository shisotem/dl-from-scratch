import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    """
    y = exp(x_k) / Σexp(x_i)
    = (exp(c) * exp(x_k)) / (exp(c) * Σexp(x_i))
    = exp(x_k + c) / Σexp(x_i + c)
    (cは任意の定数: overflow対策)
    """
    c = np.max(x)
    y = np.exp(x - c) / np.sum(np.exp(x - c))
    return y


if __name__ == "__main__":
    x = np.array([1010, 1000, 990])  # 通常の実装ではoverflowしてしまう (exp(x)->inf)
    y = softmax(x)
    print(y)
    print(np.sum(y))

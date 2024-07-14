import numpy as np
import matplotlib.pyplot as plt


def step_function(x: np.ndarray) -> np.ndarray:
    return np.array(x > 0, dtype=int)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    x = np.arange(-5, 5.1, 0.1)
    y_step = step_function(x)
    y_sig = sigmoid(x)
    plt.plot(x, y_step, "k--")
    plt.plot(x, y_sig)
    plt.ylim(-0.1, 1.1)
    plt.show()

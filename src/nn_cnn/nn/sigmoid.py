import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    x = np.arange(-5, 5.1, 0.1)
    y = sigmoid(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.grid()
    plt.show()

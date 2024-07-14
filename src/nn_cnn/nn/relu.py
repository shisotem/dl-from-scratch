import numpy as np
import matplotlib.pyplot as plt


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


if __name__ == "__main__":
    x = np.arange(-5, 5.1, 0.1)
    y = relu(x)
    plt.plot(x, y)
    plt.xlim(-6, 6)
    plt.ylim(-1, 6)
    plt.grid()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt


def step_function(x: np.ndarray) -> np.ndarray:
    return np.array(x > 0, dtype=int)


if __name__ == "__main__":
    x = np.arange(-5, 5.1, 0.1)
    y = step_function(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()

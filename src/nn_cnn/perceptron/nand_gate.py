import numpy as np
from typing import Literal


def NAND(x1: Literal[0, 1], x2: Literal[0, 1]) -> Literal[0, 1]:
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x) + b
    return 0 if tmp <= 0 else 1  # activation function -> step function


if __name__ == "__main__":
    print(NAND(0, 0))
    print(NAND(1, 0))
    print(NAND(0, 1))
    print(NAND(1, 1))

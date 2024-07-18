import numpy as np
from typing import Literal


def OR(x1: Literal[0, 1], x2: Literal[0, 1]) -> Literal[0, 1]:
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b
    return 0 if tmp <= 0 else 1


if __name__ == "__main__":
    print(OR(0, 0))
    print(OR(1, 0))
    print(OR(0, 1))
    print(OR(1, 1))

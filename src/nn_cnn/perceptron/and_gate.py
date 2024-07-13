# 学習データ (e.g., (x1,x2)=(1,1), y=1) から人の手でパラメータを考えた (期待する入出力を満たすように適切な重みを考えた)
# (-> MLの学習では自動でパラメータを決定する)

# Point: 同じ構造のパーセプトロンが、パラメータを調整することで、ANDにもNANDにもORにも変身できる

# def AND(x1, x2):
#     w1, w2, theta = 0.5, 0.5, 0.7
#     tmp = x1 * w1 + x2 * w2
#     return 0 if tmp <= theta else 1

import numpy as np


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7  # thetaを左辺に移項、-thetaをbと置く (バイアスは発火のしやすさを制御する, パイアス=下駄はき)
    tmp = np.sum(w * x) + b
    return 0 if tmp <= 0 else 1


if __name__ == "__main__":
    print(AND(0, 0))
    print(AND(1, 0))
    print(AND(0, 1))
    print(AND(1, 1))

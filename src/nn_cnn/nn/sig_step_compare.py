import numpy as np
import matplotlib.pyplot as plt


# 相違点
# パーセプトロンではニューロン間を 0 か 1 の二値信号が流れていたのに対し、NNでは連続的な実数値の信号が流れる

# 共通点
# 入力信号が重要な情報であれば大きな値を出力し、入力信号が重要でなければ小さな値を出力する
# 非常に小さい/大きい入力信号がきても、出力信号の値を0から1の間に押し込める
# ともに非線形関数


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

import os
import sys
import numpy as np
from PIL import Image

# MNISTデータセットのモジュールがあるディレクトリへのパスを追加
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from dataset.mnist import load_mnist
except ImportError:
    print("Error: Could not import 'load_mnist' from 'dataset.mnist'.")
    print(
        "Please ensure that the 'dataset' directory is in the parent directory of this script."
    )
    sys.exit(1)


def img_show(img: np.ndarray) -> None:
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img: np.ndarray = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)
print(img.shape)  # (28, 28)

img_show(img)

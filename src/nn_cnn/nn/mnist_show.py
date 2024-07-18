import os
import sys
import numpy as np
from PIL import Image

sys.path.append(os.pardir)
from dataset.mnist import load_mnist


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

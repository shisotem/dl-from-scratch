from and_gate import AND
from nand_gate import NAND
from or_gate import OR


# 単層のパーセプトロンでは線形領域しか表現できず、非線形領域 (e.g., XOR) を分割できない
# -> 層を重ねる (MLP) ことでパーセプトロンは非線形な表現が可能になる
# (パーセプトロンは層を深くすることでより柔軟な表現が可能になったもといえる)


def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


if __name__ == "__main__":
    print(XOR(0, 0))
    print(XOR(1, 0))
    print(XOR(0, 1))
    print(XOR(1, 1))

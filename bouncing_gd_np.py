# بسم الله الرحمن الرحيم و به نستعين

import numpy as np
from scipy.sparse import csr_matrix
from tqdm import trange
from typing import Callable
import numpy.typing as npt

type Vec = npt.NDArray  # can represent a matrix as well
type Input = csr_matrix | Vec
type LossFunc = Callable[[Input, Vec], float]
type GradFunc = Callable[[Input, Vec], Vec]


def gd(
    data: Input, loss_f: LossFunc, gradient: GradFunc, lr=1, epochs=50, seed=0
) -> tuple[Vec, list[float]]:
    np.random.seed(seed)
    weight = np.random.randn(data.shape[-1])
    losses = [loss_f(data, weight)]

    for _ in trange(epochs):
        g = gradient(data, weight)
        weight -= g * lr

        losses.append(loss_f(data, weight))
    return weight, losses


def bouncy_gd(
    data: Input,
    loss_f: LossFunc,
    gradient: GradFunc,
    lr=1,
    epochs=50,
    TH=0.7,
    seed=0,
    beta=0.995,
) -> tuple[Vec, list[float]]:
    np.random.seed(seed)
    feature_dimensions = data.shape[-1]
    weight = np.random.randn(feature_dimensions)
    losses = [loss_f(data, weight)]
    lr = np.ones(feature_dimensions) * lr  # per weight (parameter) adaptive learning rate
    sw = 1  # v_t
    e = 1e-08

    def dist(g1: Vec, g2: Vec) -> Vec:
        flatness_1, flatness_2 = np.linalg.norm(g1), np.linalg.norm(g2)
        dists = np.array([flatness_2, flatness_1])
        return dists / (dists.sum() + e)

    for _ in trange(epochs):
        g = gradient(data, weight)
        sw = beta * sw + np.abs(g)  # second moment

        oracle = weight - g * lr
        g_orc = gradient(data, oracle)
        if g @ g_orc < 0:
            # print('bounce!')

            d1, d2 = dist(g, g_orc)
            if d1 > TH:  # Implies that we're approaching some minima
                # print("Cut!")
                lr /= sw

            weight = weight * d1 + oracle * d2

        else:
            weight = oracle - g_orc * lr

        losses.append(loss_f(data, weight))

    return weight, losses


def main():
    mem = Memory("./mycache")
    # plt.style.use('seaborn')
    plt.rcParams["figure.autolayout"] = True

    EPOCHS: int = 50
    LR: int = 1000

    @mem.cache
    def get_data(filePath):
        data_set = load_svmlight_file(filePath)
        return data_set[0], data_set[1]

    data, target = get_data("news20.binary.bz2")
    n, d = data.shape
    # data = data.toarray()
    # data = np.append(data, np.ones((n, 1)), 1)
    # d += 1
    print(f"We have {n} samples, each has {d} features")

    def logistic_loss(x: Input, w: Vec) -> float:
        return np.mean(np.log(1 + np.exp(-target * (x @ w))))

    def logistic_grad(x: Input, w: Vec) -> Vec:
        e = np.exp(-target * (x @ w))
        gradient = ((-e / (1 + e) * target) @ x) / n
        return gradient

    _, losses_gd = gd(data, logistic_loss, logistic_grad, lr=LR, epochs=EPOCHS)
    print(f"Loss GD: {losses_gd[-1]:.4}")
    plt.semilogy(losses_gd, label="Vanilla GD")

    _, losses_bgd = bouncy_gd(data, logistic_loss, logistic_grad, lr=LR, epochs=EPOCHS)
    print(f"Loss BGD: {losses_bgd[-1]:.4}")
    plt.semilogy(losses_bgd, label="Bouncing GD")

    plt.ylabel("Loss", rotation="horizontal")
    plt.xlabel("Iterations")
    plt.gca().text(15, 3, f"$LR={LR}$", c="pink", size=20)
    plt.legend(prop={"size": 15})
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":  # Such that when you import a function from this script, the whole script doesn't run automatically
    from joblib import Memory
    from sklearn.datasets import load_svmlight_file
    import matplotlib.pyplot as plt

    main()

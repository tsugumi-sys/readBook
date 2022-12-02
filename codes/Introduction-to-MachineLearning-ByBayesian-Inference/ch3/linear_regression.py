from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def linear_regression(ndim: int, error_std: float) -> Tuple:
    # initialize m, A
    e = np.random.normal(0, 1 / error_std, 1)
    m = np.zeros((ndim))
    A = np.diag(np.ones((ndim)))
    w = np.random.multivariate_normal(m, A)
    # w = softmax(w)

    return w, e


if __name__ == "__main__":
    W, err = linear_regression(ndim=4, error_std=10)
    x = np.linspace(-1, 1, 100)
    y = []
    for i in x:
        X = [1, i, i**2, i**3]
        y.append(np.dot(W, X) + err)

    plt.plot(x, y)
    plt.savefig("ch3-linear-regression.png")
    plt.close()

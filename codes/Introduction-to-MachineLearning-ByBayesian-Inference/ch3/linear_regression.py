from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np


# def softmax(x: np.ndarray) -> np.ndarray:
#     return np.exp(x) / np.sum(np.exp(x), axis=0)


class LinearRegression:
    def __init__(self, ndim: int, error_std: float) -> None:
        self.ndim = ndim
        self.error_std = error_std
        self.error = np.random.normal(0, 1 / error_std, 1)
        self.mean = np.zeros((ndim))
        self.vc_matrix = np.diag(np.ones((ndim)))
        self.weights = np.random.multivariate_normal(self.mean, self.vc_matrix)

    def forward(self, x) -> float:
        return np.dot(self.weights, x) + self.error

    def update(self, inputs: List[Tuple]) -> None:
        """
        Args:
            inputs (List[Tuple]): Lists of (x, y)
        """
        sum_x = 0
        for x, _ in inputs:
            sum_x += np.dot(x, x.T)


def linear_regression(ndim: int, error_std: float) -> Tuple:
    # initialize m, A
    e = np.random.normal(0, 1 / error_std, 1)
    m = np.zeros((ndim))
    A = np.diag(np.ones((ndim)))
    w = np.random.multivariate_normal(m, A)

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

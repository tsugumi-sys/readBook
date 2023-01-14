import numpy as np

# def softmax(x: np.ndarray) -> np.ndarray:
#     return np.exp(x) / np.sum(np.exp(x), axis=0)


class LinearRegression1D:
    def __init__(self, ndim: int, _lambda: float = 10.0) -> None:
        self.ndim = ndim
        self._lambda = _lambda
        self.std = 1 / _lambda
        self.w_mean = np.zeros((ndim))
        self.w_vc_matrix = np.diag(np.ones((ndim)))

        # Initialize weights
        self.update_weights(self.w_mean, self.w_vc_matrix)

    def forward(self, x: np.ndarray) -> float:
        if x.ndim != self.ndim:
            raise ValueError(
                f"The dimention of x must be {self.ndim}, instead of {x.ndim}"
            )

        print(x, self.weights)
        return np.random.normal(np.dot(self.weights, x), self.std)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Args:
            inputs (List[Tuple]): Lists of (x, y)
        """
        # Calculate posterior weights' mean and vc matrix.
        print(self.w_mean, self.w_vc_matrix)
        prior_w_vc_matrix = self.w_vc_matrix.copy()
        self.w_vc_matrix = self.posterior_vc_matrix(x, prior_w_vc_matrix)
        self.w_mean = self.posterior_mean(
            x, y, self.w_vc_matrix, prior_w_vc_matrix, self.w_mean
        )
        print(self.w_mean, self.w_vc_matrix)
        # Update posterior of weights and std
        self.update_weights(self.w_mean, self.w_vc_matrix)
        self.update_std(self.std, x, self.w_vc_matrix)

    def posterior_vc_matrix(
        self, x: np.ndarray, prior_w_vc_matrix: np.ndarray
    ) -> np.ndarray:
        sum_xx = np.zeros((self.ndim, self.ndim))
        for idx in range(x.shape[0]):
            x_n = x[idx]
            sum_xx += np.outer(x_n, x_n.T)
        return self._lambda * sum_xx + prior_w_vc_matrix

    def posterior_mean(
        self,
        x: np.ndarray,
        y: np.ndarray,
        posterior_w_vc_matrix: np.ndarray,
        prior_w_vc_matrix: np.ndarray,
        prior_mean: np.ndarray,
    ) -> np.ndarray:
        sum_yx = np.zeros((self.ndim))
        for idx in range(y.shape[0]):
            sum_yx += y[idx] * x[idx]
        return (
            np.linalg.inv(posterior_w_vc_matrix)
            * (self._lambda * sum_yx + prior_w_vc_matrix * prior_mean)
        )[0]

    def update_weights(self, mean, vc_matrix) -> None:
        self.weights = np.random.multivariate_normal(mean, vc_matrix)

    def update_std(
        self, prior_std: float, x: np.ndarray, vc_matrix: np.ndarray
    ) -> None:
        self.std = prior_std + x.T * vc_matrix * x


def input_vec(ndim: int, x: float) -> np.ndarray:
    return [x**i for i in range(ndim)]


def generate_training_data(ndim: int, N: int = 10) -> (np.ndarray, np.ndarray):
    x_points = np.random.uniform(0, 5, N)
    inputs = np.zeros((N, ndim))
    for idx, x in enumerate(x_points):
        inputs[idx] = input_vec(ndim, x)
    return inputs, np.sin(x_points)


def main():
    np.random.seed(43)
    # Configuration
    N = 10
    # ndims = [1, 2, 3, 4, 5, 10]
    ndim = 1
    X, y = generate_training_data(ndim, N)
    model = LinearRegression(ndim=ndim)
    model.fit(X, y)


if __name__ == "__main__":
    a1 = np.ones((1, 1))
    a2 = np.random.rand(1, 1)
    print((a1 * a1).shape)
    print(np.multiply(a1, a2).shape)
    # main()

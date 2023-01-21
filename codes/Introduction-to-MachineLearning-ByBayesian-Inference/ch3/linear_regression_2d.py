from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class LinearRegression2D:
    def __init__(self, ndim: int, _lambda: float = 10.0) -> None:
        self.ndim = ndim
        self._lambda = _lambda
        self.std = 1 / _lambda
        self.w_mean = np.zeros(ndim)
        self.w_vc_matrix = np.diag(np.ones(ndim))

    def predict(self, x: np.ndarray) -> Tuple[float, float]:
        """
        Return:
            Tuple[float, float]: The mean and standard deviation
                of predicted normal distribution.
        """
        mean = np.dot(self.w_mean, x)
        std = self.std + x.T @ np.linalg.inv(self.w_vc_matrix) @ x
        return (mean, std)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        # Calculate posterior weights' mean and vc matrix.
        prior_w_vc_matrix = self.w_vc_matrix.copy()
        self.w_vc_matrix = self.posterior_vc_matrix(x, prior_w_vc_matrix)
        self.w_mean = self.posterior_mean(
            x, y, self.w_vc_matrix, prior_w_vc_matrix, self.w_mean
        )

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
        return np.matmul(
            np.linalg.inv(posterior_w_vc_matrix),
            self._lambda * sum_yx + np.matmul(prior_w_vc_matrix, prior_mean),
        )


def input_vec(ndim: int, x: float) -> np.ndarray:
    return np.asarray([x**i for i in range(ndim)])


def generate_training_data(
    ndim: int,
    x_points: List[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            input_vectors like (1, x, x**2, ...) and y (sin value of x points)
    """
    input_vectors = np.zeros((len(x_points), ndim))
    for idx, x in enumerate(x_points):
        input_vectors[idx] = input_vec(ndim, x)
    return input_vectors, np.sin(x_points)


def create_plot(ax, ndim, x, label, pred_x, pred_means, pred_stds, ylim) -> None:
    ax.set_title(f"M = {ndim}")
    ax.set(ylim=ylim)
    # plot label and x with black dot
    sns.scatterplot(x=x, y=label, color="black", ax=ax)
    # plot predicts
    sns.lineplot(x=pred_x, y=pred_means, color="purple", ax=ax)
    # plot upper and lower std^(1/2) line
    upper_std_line = [
        pred_means[i] + np.sqrt(np.abs(pred_stds[i])) for i in range(len(pred_means))
    ]
    lower_std_line = [
        pred_means[i] - np.sqrt(np.abs(pred_stds[i])) for i in range(len(pred_means))
    ]
    sns.lineplot(x=pred_x, y=upper_std_line, color="blue", linestyle="--", ax=ax)
    sns.lineplot(x=pred_x, y=lower_std_line, color="blue", linestyle="--", ax=ax)


def main():
    np.random.seed(43)

    # Configuration
    n_samples = 10
    input_x_min, input_x_max = 0, 7
    input_x = np.random.uniform(input_x_min, input_x_max, n_samples)
    pred_x = np.linspace(-0, 8, 40)
    ylim = (-4, 4)
    ndims = [1, 2, 3, 4, 5, 10]
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 10))
    axes = axes.flatten()

    for ndim, ax in zip(ndims, axes):
        print("=" * 10, f"M = {ndim}", "=" * 10)
        # Fit
        input_vectors, y = generate_training_data(ndim, input_x)
        model = LinearRegression2D(ndim=ndim)
        model.fit(input_vectors, y)
        # Inference
        pred_means, pred_stds = [], []
        for x_new in pred_x:
            mean_new, std_new = model.predict(input_vec(ndim, x_new))
            pred_means.append(mean_new)
            pred_stds.append(std_new)

        # Create plot
        create_plot(ax, ndim, input_x, y, pred_x.tolist(), pred_means, pred_stds, ylim)

    plt.tight_layout()
    plt.savefig("prediction-distribution-of-polynomial-regression-model.png")
    plt.close()


if __name__ == "__main__":
    main()

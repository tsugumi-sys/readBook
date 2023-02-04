import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scistats
import pandas as pd
import seaborn as sns

sys.path.append("..")
from utils.common import ApproximateMethod  # noqa: E402

rng = np.random.default_rng()


class GaussianMixtureModel:
    def __init__(
        self,
        num_clusters: int,
        data_dim: int,
        init_means: Optional[np.ndarray] = None,
        init_beta: Optional[np.ndarray] = None,
        init_freedom_deg: Optional[np.ndarray] = None,
        init_wishart_W: Optional[np.ndarray] = None,
        init_dirichlet_alpha: Optional[np.ndarray] = None,
    ):
        """
        Args:
            num_clusters: The number of clusters.
            data_dim: The dimentions of data.
            init_means: initial value of mean parameters for mu's normal distribution.
            init_beta: initial value of a coefficient parameter for vc matrix.
            init_freedom_deg: initial value of the freedom degree for wishart
                 distribution.
            init_wishart_W: initial value of a positive definite matrix for wishart
                distribution.
            init_dirichlet_alpha: The initial value of dirichlet distribution.
        """
        self.num_clusters = num_clusters
        self.data_dim = data_dim

        if init_means is not None and init_means.shape[0] != num_clusters:
            raise ValueError(f"The length of `init_m` should be {num_clusters}")
        if init_means is None:
            init_means = np.zeros((num_clusters, data_dim))
        self.init_means = init_means[0]
        self.means = init_means

        if init_beta is not None and init_beta.shape[0] != num_clusters:
            raise ValueError(f"The length of `init_beta` should be {num_clusters}")
        if init_beta is None:
            init_beta = np.ones(num_clusters) * 0.1
        self.init_beta = init_beta[0]
        self.beta = init_beta

        if init_freedom_deg is not None and init_freedom_deg.shape[0] != num_clusters:
            raise ValueError(
                f"The first dim of `init_freedom_deg` should be {num_clusters}."
            )
        if init_freedom_deg is None:
            init_freedom_deg = np.ones(num_clusters) + data_dim
        self.init_freedom_deg = init_freedom_deg[0]
        self.freedom_deg = init_freedom_deg

        if init_wishart_W is not None and init_wishart_W.shape != (
            num_clusters,
            data_dim,
            data_dim,
        ):
            raise ValueError(
                f"The shape of `init_W` should be ({data_dim}, {data_dim}), "
                f"instead of {init_wishart_W.shape}"
            )
        if init_wishart_W is None:
            init_wishart_W = np.zeros((num_clusters, data_dim, data_dim))
            for k in range(num_clusters):
                init_wishart_W[k] = np.diag(np.ones(data_dim))
        self.init_wishart_W = init_wishart_W[0]
        self.wishart_W = init_wishart_W

        if init_dirichlet_alpha is None:
            init_dirichlet_alpha = np.ones(num_clusters)
        self.init_dirichlet_alpha = init_dirichlet_alpha
        self.dirichlet_alpha = init_dirichlet_alpha

    def fit(
        self, x: np.ndarray, max_iter: int = 100, approximate_method: str = "gibbs"
    ) -> List:
        if not ApproximateMethod.valid(approximate_method):
            raise ValueError(
                f"`approximate_method` should be in {ApproximateMethod.members()},"
                f" instead of {approximate_method}"
            )

        elbo_scores = []
        for step in range(max_iter):
            if approximate_method == ApproximateMethod.gibbs:
                self.gibbs(x)
        return elbo_scores

    def gibbs(self, x) -> None:
        # Sampling s
        print(self.means)
        data_size = x.shape[0]
        wishart = self.wisharts()
        mixing_ratios = self.mixsing_ratios()
        x_minus_means = x - self.means[:, None]  # (num_clusters, num_x, data_dims)
        etas = np.zeros((data_size, self.num_clusters))
        for k in range(self.num_clusters):
            lambda_k = wishart[k]
            det_lambda_k = np.linalg.det(lambda_k)
            ln_mixing_ratio_k = np.log(mixing_ratios[0, k])
            for i in range(data_size):
                etas[i, k] = np.exp(
                    -0.5 * x_minus_means[k, i].T @ lambda_k @ x_minus_means[k, i]
                    + 0.5 * det_lambda_k
                    + ln_mixing_ratio_k
                )
        etas /= np.sum(etas, axis=1)[:, None]
        selected_clusters = rng.multinomial(1, pvals=etas, size=data_size)

        # Update parameters
        count_selected_clusters = np.sum(selected_clusters, axis=0)
        self.beta = count_selected_clusters + self.init_beta
        means = np.zeros((self.num_clusters, self.data_dim))
        wishart_W = np.zeros((self.num_clusters, self.data_dim, self.data_dim))
        for k in range(self.num_clusters):
            selected_cluster_k = selected_clusters[:, k]
            means_k = (
                np.sum(selected_cluster_k[:, None] * x, axis=0)
                + self.init_beta * self.init_means
            ) / self.beta[k]
            means[k] = means_k
            wishart_W[k] = np.linalg.inv(
                (
                    np.sum(
                        selected_cluster_k * np.square(np.linalg.norm(x, axis=1)),
                        axis=0,
                    )
                    + self.init_beta * np.dot(self.init_means, self.init_means)
                    - self.beta[k] * np.dot(means_k, means_k)
                    + np.linalg.inv(self.init_wishart_W)
                )
            )
        self.means = means
        self.wishart_W = wishart_W
        self.freedom_deg = count_selected_clusters + self.init_freedom_deg
        self.dirichlet_alpha = count_selected_clusters + self.init_dirichlet_alpha

    def mixsing_ratios(self) -> np.ndarray:
        return rng.dirichlet(self.dirichlet_alpha, 1)

    def mus(self) -> np.ndarray:
        """Mean parameters of x"""
        mus = np.zeros((self.num_clusters, self.data_dim))
        wisharts = self.wisharts()
        for k in range(self.num_clusters):
            mus[k] = rng.multivariate_normal(
                self.means[k], np.linalg.inv(self.beta[k] * wisharts[k])
            )
        return mus

    def wisharts(self) -> np.ndarray:
        wisharts = np.zeros((self.num_clusters, self.data_dim, self.data_dim))
        for k in range(self.num_clusters):
            wisharts[k] = scistats.wishart.rvs(
                self.freedom_deg[k], self.wishart_W[k], size=1
            )
        return wisharts


def generate_gaussian_mixture_data(
    n_samples: int,
    data_dims: int,
    mixing_ratios: np.ndarray,
    means: np.ndarray,
    vc_matrixs: np.ndarray,
) -> Dict:
    """Geberate data by sampling from Gaussian Mixture model.

    Args:
        data_dims: The dimention of data.
        mixing_ratios: The mixing ratios of each clusters. The samples of each clusters
             are calcualted with `mixing_ratios` and `n_samples`.
        means: The mean parameters of each cluster.
            The shape should be like (num_clusters, data_dim)
        vc_matrixs: The variance-covariance matrix of each clusters.
            The shape should be like (num_clusters, data_dim, data_dim)
    """
    num_clusters = mixing_ratios.shape[0]
    if means.shape[0] != num_clusters:
        raise ValueError("The cluster number is different from `mixin_ratios` and `mu`")
    if vc_matrixs.shape[0] != num_clusters:
        raise ValueError(
            "The cluster number is different from `mixin_ratios` and `vc_matricxs`"
        )
    if means.shape != (num_clusters, data_dims):
        raise ValueError(
            f"The shape of `mu` should be like ({num_clusters}, {data_dims})"
        )
    if vc_matrixs.shape != (num_clusters, data_dims, data_dims):
        raise ValueError(
            "The shape of `vc_matrixs` "
            f"should be like ({num_clusters}, {data_dims}, {data_dims})"
        )
    data = {}
    for c_idx in range(num_clusters):
        cluster_n_samples = int(n_samples * mixing_ratios[c_idx])
        data[str(c_idx)] = rng.multivariate_normal(
            means[c_idx], vc_matrixs[c_idx], size=cluster_n_samples
        )
    return data


def visualize_gmm_data(data: Dict):
    """
    Args:
        data (Dict): {cluster_idx(str): data(np.ndarray)}
    """
    df = pd.DataFrame()
    for cluster_idx, c_data in data.items():
        c_df = pd.DataFrame(
            c_data, columns=[f"x{i + 1}" for i in range(c_data.shape[1])]
        )
        c_df["cluster_idx"] = int(cluster_idx) + 1
        df = pd.concat([c_df, df], axis=0)
    _, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x="x1", y="x2", hue="cluster_idx", ax=ax)
    plt.savefig("Generated-Gauusian-Mixture-Model-data.png")
    plt.close()


if __name__ == "__main__":
    n_samples = 10
    data_dims = 2
    n_clusters = 3
    mixing_raitos = np.array([0.3, 0.3, 0.4])
    means = np.array([[-6.5, 1], [-1, 4], [4, 0]])
    vc_matrixs = np.array(
        [
            [
                [1, -0.3],
                [-0.3, 1],
            ],
            [
                [1, -0.7],
                [-0.7, 1],
            ],
            [
                [1, 0.3],
                [0.3, 1],
            ],
        ]
    )
    data = generate_gaussian_mixture_data(
        n_samples=n_samples,
        data_dims=data_dims,
        mixing_ratios=mixing_raitos,
        means=means,
        vc_matrixs=vc_matrixs,
    )
    # visualize_gmm_data(data)
    model = GaussianMixtureModel(num_clusters=n_clusters, data_dim=data_dims)
    model.fit(np.concatenate([a for a in data.values()], axis=0), max_iter=10)
    print("label means:", means)

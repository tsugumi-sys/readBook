import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append("..")
from utils.common import ApproximateMethod

rng = np.random.default_rng()


class GaussianMixtureModel:
    def __init__(
        self,
        num_clusters: int,
        data_dim: int,
        init_m: Optional[np.ndarray] = None,
        init_beta: float = 0.1,
        init_freedom_deg: Optional[int] = None,
        init_W: Optional[np.ndarray] = None,
        dirichlet_alpha: Optional[np.ndarray] = None,
    ):
        """
        Args:
            init_m: initial value of mean parameters for mu's normal distribution.
            init_beta: initial value of a coefficient parameter for vc matrix.
            init_freedom_deg: initial value of the freedom degree for wishart
                 distribution.
            init_W: initial value of a positive definite matrix for wishart
                distribution.
        """
        self.num_clusters = num_clusters
        self.data_dim = data_dim

        if init_m is not None and init_m.shape[0] != num_clusters:
            raise ValueError(f"The length of `init_m` should be {num_clusters}")
        if init_m is None:
            init_m = np.zeros(data_dim)
        self.init_m = init_m

        self.init_beta = init_beta

        if init_freedom_deg is None:
            init_freedom_deg = data_dim + 1
        self.init_freedom_deg = init_freedom_deg

        if init_W is not None and init_W.shape != (data_dim, data_dim):
            raise ValueError(
                f"The shape of `init_W` should be ({data_dim}, {data_dim}), "
                f"instead of {init_W.shape}"
            )
        if init_W is None:
            init_W = np.diag(np.ones(data_dim))

        if dirichlet_alpha is None:
            dirichlet_alpha = np.ones(num_clusters)
        self.dirichlet_alpha


def generate_gaussian_mixture_data(
    n_samples: int,
    data_dims: int,
    mixing_ratios: np.ndarray,
    means: np.ndarray,
    vc_matrixs: np.ndarray,
) -> Dict:
    """
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
    n_samples = 250
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
        n_samples=250,
        data_dims=data_dims,
        mixing_ratios=mixing_raitos,
        means=means,
        vc_matrixs=vc_matrixs,
    )
    visualize_gmm_data(data)

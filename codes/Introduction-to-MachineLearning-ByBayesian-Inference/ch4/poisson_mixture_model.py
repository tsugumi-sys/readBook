from typing import Tuple, List
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

rng = np.random.default_rng()


class ApproximateMethod(str, Enum):
    gibbs = "gibbs"
    variational = "variational"
    collapsed_gibbs = "collapsed_gibbs"

    @staticmethod
    def members() -> List[str]:
        return [v.value for v in ApproximateMethod.__members__.values()]

    @staticmethod
    def valid(name: str) -> bool:
        return name in ApproximateMethod.members()


class PoissonMixtureModel:
    def __init__(
        self,
        num_clusters: int,
        gamma_a: float = 2.0,
        gamma_b: float = 2.0,
        dirichlet_alpha: np.ndarray = np.array([1, 1]),
    ):
        self.num_clusters = num_clusters
        self.gamma_a = gamma_a
        self.gamma_b = gamma_b
        if dirichlet_alpha.shape[0] != num_clusters:
            raise ValueError(f"The length of dirichlet_alpha should be {num_clusters}.")
        self.dirichlet_alpha = dirichlet_alpha

    def sample(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        selected_clusters = self.select_clusters(size)
        _, cluster_idxs = np.where(selected_clusters == 1)
        return (
            rng.poisson(lam=self.cluster_lambdas[cluster_idxs], size=size),
            self.cluster_lambdas,
        )

    def fit(
        self,
        x: np.ndarray,
        max_iter: int = 100,
        approximate_method: str = "gibbs",
    ) -> None:
        if not ApproximateMethod.valid(approximate_method):
            raise ValueError(
                f"Invalid `approximate_method` {approximate_method}, "
                f"should be in {ApproximateMethod.members}"
            )

        for step in range(max_iter):
            if approximate_method == ApproximateMethod.gibbs:
                if step == 0:
                    cluster_lambdas = self.cluster_lambdas()
                    cluster_mixing_ratios = self.mixing_ratios()
                self.gibss(x, cluster_lambdas, cluster_mixing_ratios)
                # Sampling
                cluster_lambdas = self.cluster_lambdas()
                cluster_mixing_ratios = self.mixing_ratios()

    def gibss(
        self,
        x: np.ndarray,
        cluster_lambdas: np.ndarray,
        cluster_mixing_ratios: np.ndarray,
    ) -> None:
        # Sampling s_n
        mu = np.exp(
            x[:, None] * np.log(cluster_lambdas)
            - cluster_lambdas
            + np.log(cluster_mixing_ratios)
        )
        mu = mu / mu.sum(axis=1)[:, None]
        selected_clusters = rng.multinomial(1, pvals=mu, size=x.shape[0])
        # Update parameters
        self.gamma_a = np.sum(selected_clusters.T * x, axis=1) + self.gamma_a
        self.gamma_b = np.sum(selected_clusters, axis=0) + self.gamma_b
        self.dirichlet_alpha = np.sum(selected_clusters, axis=0) + self.dirichlet_alpha

    def cluster_lambdas(self) -> np.ndarray:
        return rng.gamma(self.gamma_a, 1 / self.gamma_b, self.num_clusters)

    def select_clusters(self, size: int) -> np.ndarray:
        return rng.multinomial(1, self.mixing_ratios, size)

    def mixing_ratios(self) -> np.ndarray:
        return rng.dirichlet(self.dirichlet_alpha, 1)


def generate_poisson_mixture_data(
    lambdas: np.ndarray, mixing_ratios: np.ndarray, n_samples: int
) -> np.ndarray:
    if lambdas.ndim != 1 or mixing_ratios.ndim != 1:
        raise ValueError(
            "`lamdbas` or `mixing_ratios` should be a single dimention vectors"
            f", instead of lambdas.ndim={lambdas.ndim}, "
            f"mixing_ratios.ndim={mixing_ratios.ndim}."
        )
    if lambdas.shape[0] != mixing_ratios.shape[0]:
        raise ValueError("`lambads` and `mixing_ratios should be the same shape`")

    selected_clusters = np.random.multinomial(n=1, pvals=mixing_ratios, size=n_samples)
    _, cluster_idxs = np.where(selected_clusters == 1)
    return rng.poisson(lam=lambdas[cluster_idxs], size=n_samples)


def plot_poisson_mixutre_data(data: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(x=data, ax=ax)
    ax.set_title("Plot poisson mixture data")
    fig.savefig("ch4_poisson_mixutre_data.png")


if __name__ == "__main__":
    cluster_lambdas = np.asarray([25, 55])
    cluster_mixing_ratios = np.asarray([0.6, 0.4])
    n_samples = 250
    data = generate_poisson_mixture_data(
        cluster_lambdas, cluster_mixing_ratios, n_samples
    )
    # plot_poisson_mixutre_data(
    #     generate_poisson_mixture_data(cluster_lambdas, cluster_mixing_ratios, n_samples)
    # )
    pcm = PoissonMixtureModel(num_clusters=2)
    pcm.fit(data)
    print(pcm.cluster_lambdas())

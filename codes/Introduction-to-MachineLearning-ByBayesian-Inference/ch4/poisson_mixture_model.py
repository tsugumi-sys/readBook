import sys
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import digamma
from scipy.stats import nbinom

sys.path.append("..")
from utils.common import ApproximateMethod  # noqa

rng = np.random.default_rng()


class PoissonMixtureModel:
    def __init__(
        self,
        num_clusters: int,
        init_gamma_a: Optional[np.ndarray] = None,
        init_gamma_b: Optional[np.ndarray] = None,
        init_dirichlet_alpha: Optional[np.ndarray] = None,
    ):
        self.num_clusters = num_clusters

        if init_gamma_a is not None and init_gamma_a.shape[0] != num_clusters:
            raise ValueError(f"The length of `gamma_a` should be {num_clusters}.")

        if init_gamma_b is not None and init_gamma_b.shape[0] != num_clusters:
            raise ValueError(f"The length of `gamma_b` should be {num_clusters}.")

        if init_gamma_a is None:
            init_gamma_a = np.ones(num_clusters).astype(float) * (1 / num_clusters)
        self.init_gamma_a = init_gamma_a
        self.gamma_a = init_gamma_a

        if init_gamma_b is None:
            init_gamma_b = np.ones(num_clusters)
        self.init_gamma_b = init_gamma_b
        self.gamma_b = init_gamma_b

        if (
            init_dirichlet_alpha is not None
            and init_dirichlet_alpha.shape[0] != num_clusters
        ):
            raise ValueError(f"The length of dirichlet_alpha should be {num_clusters}.")
        if init_dirichlet_alpha is None:
            init_dirichlet_alpha = np.ones(num_clusters)
        self.init_dirichlet_alpha = init_dirichlet_alpha
        self.dirichlet_alpha = init_dirichlet_alpha

    def sample(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        selected_clusters = self.select_clusters(size)
        _, cluster_idxs = np.where(selected_clusters == 1)
        cluster_lambdas = self.cluster_lambdas()
        return (
            rng.poisson(lam=cluster_lambdas[cluster_idxs], size=size),
            cluster_lambdas,
        )

    def fit(
        self,
        x: np.ndarray,
        max_iter: int = 100,
        approximate_method: str = "gibbs",
    ) -> List:
        if not ApproximateMethod.valid(approximate_method):
            raise ValueError(
                f"Invalid `approximate_method` {approximate_method}, "
                f"should be in {ApproximateMethod.members()}"
            )

        elbo_scores = []
        for step in range(max_iter):
            old_gamma_a, old_gamma_b, old_dirichlet_alpha = (
                self.gamma_a.copy(),
                self.gamma_b.copy(),
                self.dirichlet_alpha.copy(),
            )
            if approximate_method == ApproximateMethod.gibbs:
                cluster_lambdas = self.cluster_lambdas()
                cluster_mixing_ratios = self.mixing_ratios()
                s_mean_parameters = self.gibbs(
                    x, cluster_lambdas, cluster_mixing_ratios
                )
            elif approximate_method == ApproximateMethod.variational:
                # Inititalize parameter distributions
                if step == 0:
                    _ = self.initialzie_parameter_dist(x)
                s_mean_parameters = self.variational(x)
            else:
                # Initialize parameter distributions
                if step == 0:
                    selected_clusters = self.initialzie_parameter_dist(x)
                s_mean_parameters, selected_clusters = self.collapsed_gibbs(
                    x, selected_clusters
                )
            # Calculate ELBO score
            elbo_scores.append(
                self.elbo(
                    x,
                    s_mean_parameters,
                    old_gamma_a,
                    old_gamma_b,
                    old_dirichlet_alpha,
                    self.gamma_a,
                    self.gamma_b,
                    self.dirichlet_alpha,
                )
            )
        return elbo_scores

    def initialzie_parameter_dist(self, x) -> np.ndarray:
        selected_clusters = self.select_clusters(x.shape[0])
        self.gamma_a = np.sum(selected_clusters.T * x, axis=1) + self.init_gamma_a
        each_clusters_selected_count = np.sum(selected_clusters, axis=0)
        self.gamma_b = each_clusters_selected_count + self.init_gamma_b
        self.dirichlet_alpha = each_clusters_selected_count + self.init_dirichlet_alpha
        return selected_clusters

    def gibbs(
        self,
        x: np.ndarray,
        cluster_lambdas: np.ndarray,
        cluster_mixing_ratios: np.ndarray,
    ) -> np.ndarray:
        # Sampling s_n
        s_mean_parameters = np.exp(
            x[:, None] * np.log(cluster_lambdas)
            - cluster_lambdas
            + np.log(cluster_mixing_ratios)
        )
        s_mean_parameters /= np.sum(s_mean_parameters, axis=1)[:, None]
        selected_clusters = rng.multinomial(1, pvals=s_mean_parameters, size=x.shape[0])
        # Update parameters
        self.gamma_a = np.sum(selected_clusters.T * x, axis=1) + self.init_gamma_a
        each_clusters_selected_count = np.sum(selected_clusters, axis=0)
        self.gamma_b = each_clusters_selected_count + self.init_gamma_b
        self.dirichlet_alpha = each_clusters_selected_count + self.init_dirichlet_alpha
        # Calculate ELBO
        return s_mean_parameters

    def variational(self, x) -> np.ndarray:
        # Update the means of s (mu)
        s_mean_parameters = np.exp(
            x[:, None] * (digamma(self.gamma_a) - np.log(self.gamma_b))
            - (self.gamma_a / self.gamma_b)
            + (digamma(self.dirichlet_alpha) - digamma(np.sum(self.dirichlet_alpha)))
        )
        s_mean_parameters /= np.sum(s_mean_parameters, axis=1)[:, None]
        # Update parameters
        self.gamma_a = np.sum(s_mean_parameters.T * x, axis=1) + self.init_gamma_a
        each_cluster_mparams = np.sum(s_mean_parameters, axis=0)
        self.gamma_b = each_cluster_mparams + self.init_gamma_b
        self.dirichlet_alpha = each_cluster_mparams + self.init_dirichlet_alpha
        return s_mean_parameters

    def collapsed_gibbs(
        self, x: np.ndarray, selected_clusters: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        sampled_selected_clusters = np.zeros((x.shape[0], self.num_clusters))
        s_mean_parameters = np.zeros((x.shape[0], self.num_clusters))
        for idx, x_i in enumerate(x):
            # subtract statistics of x_i
            self.gamma_a -= selected_clusters[idx] * x_i
            self.gamma_b -= selected_clusters[idx]
            self.dirichlet_alpha -= selected_clusters[idx]
            # Sample s of x_i
            sampling_prob = (
                nbinom.pmf(x_i, self.gamma_a, 1 - (1 / (1 + self.gamma_b))) + 1e-7
            ) * (self.dirichlet_alpha / np.sum(self.dirichlet_alpha))
            sampling_prob /= np.sum(sampling_prob)
            s_mean_parameters[idx] = sampling_prob
            sampled_s = rng.multinomial(1, pvals=sampling_prob, size=1).reshape(
                self.num_clusters
            )
            sampled_selected_clusters[idx] = sampled_s
            # add statistics of x_i
            self.gamma_a += sampled_s * x_i
            self.gamma_b += sampled_s
            self.dirichlet_alpha += sampled_s
            # save sampled s
            sampled_selected_clusters[idx] = sampled_s
        return s_mean_parameters, sampled_selected_clusters

    def elbo(
        self,
        x: np.ndarray,
        s_mean_parameters: np.ndarray,
        old_gamma_a: np.ndarray,
        old_gamma_b: np.ndarray,
        old_dirichlet_alpha: np.ndarray,
        new_gamma_a: np.ndarray,
        new_gamma_b: np.ndarray,
        new_dirichlet_alpha: np.ndarray,
    ):
        mean_lambdas = new_gamma_a / new_gamma_b
        mean_ln_lambdas = digamma(new_gamma_a) - np.log(new_gamma_b)
        mean_ln_pis = digamma(new_dirichlet_alpha) - digamma(
            np.sum(new_dirichlet_alpha)
        )
        mean_pis = new_dirichlet_alpha / np.sum(new_dirichlet_alpha)

        ln_marginal_likelihood = (
            np.sum(s_mean_parameters * (mean_ln_lambdas * x[:, None] - mean_lambdas))
            + np.sum(s_mean_parameters * mean_ln_pis)
            - np.sum(s_mean_parameters * new_dirichlet_alpha)
        )
        kl_lambdas = np.sum(
            (new_gamma_a - old_gamma_a) * mean_ln_lambdas
            - (new_gamma_b - old_gamma_b) * mean_lambdas
        )
        kl_pis = np.sum((new_dirichlet_alpha - old_dirichlet_alpha) * mean_pis)
        return ln_marginal_likelihood - kl_lambdas - kl_pis

    def cluster_lambdas(self) -> np.ndarray:
        return rng.gamma(self.gamma_a, 1 / self.gamma_b, self.num_clusters)

    def select_clusters(self, size: int) -> np.ndarray:
        return rng.multinomial(1, self.mixing_ratios(), size)

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

    selected_clusters = rng.multinomial(n=1, pvals=mixing_ratios, size=n_samples)
    _, cluster_idxs = np.where(selected_clusters == 1)
    return rng.poisson(lam=lambdas[cluster_idxs], size=n_samples)


def plot_poisson_mixutre_data(
    data: np.ndarray,
    cluster_lambdas: np.ndarray,
    cluster_mixing_ratios: np.ndarray,
    n_samples: int,
    save_fig_path: str = "ch4_poission_mixture_model_traiing_data.png",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(x=data, ax=ax)
    ax.set_title(
        f"Plot poisson mixture data. (Lamdbas: {cluster_lambdas}, "
        f"Mixing Ratios: {cluster_mixing_ratios}, N: {n_samples})"
    )
    fig.savefig("ch4_poisson_mixutre_data.png")


def plot_training_results(
    results_df: pd.DataFrame,
    max_iter: int,
    num_clusters: int,
    n_samples: int,
    save_fig_path: str = "training_elbo_scores_each_methods.png",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.lineplot(data=results_df, x="steps", y="ELBO", hue="method")
    ax.set_title(
        f"ELBO scores (Max Iteration: {max_iter}, Number of Clusters: {num_clusters}, "
        f"Number of datasets: {n_samples})"
    )
    fig.savefig(save_fig_path)


if __name__ == "__main__":
    # Configuration
    num_clusters = 4
    cluster_lambdas = np.array([10, 30, 50, 70])
    cluster_mixing_ratios = np.array([0.35, 0.15, 0.35, 0.15])
    n_samples = 400
    max_iter = 500
    each_methods_runs = 10
    results_df = pd.DataFrame()
    data = generate_poisson_mixture_data(
        cluster_lambdas, cluster_mixing_ratios, n_samples
    )

    # Visualize training data
    plot_poisson_mixutre_data(data, cluster_lambdas, cluster_mixing_ratios, n_samples)

    # Training and save ELBO scores
    for method in ApproximateMethod.members():
        for run in range(each_methods_runs):
            print(f"Method: {method}, {run + 1} try.")
            res_df = pd.DataFrame()
            model = PoissonMixtureModel(num_clusters=num_clusters)
            elbo_scores = model.fit(data, max_iter=max_iter, approximate_method=method)
            res_df["ELBO"] = elbo_scores
            res_df["method"] = method
            res_df["steps"] = [i for i in range(max_iter)]
            results_df = pd.concat([results_df, res_df], axis=0)

    # Visualize training results
    plot_training_results(results_df, max_iter, num_clusters, n_samples)

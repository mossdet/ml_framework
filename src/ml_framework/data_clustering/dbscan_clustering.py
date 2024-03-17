import pandas as pd
import numpy as np
import optuna
import sklearn
import matplotlib.pyplot as plt
import logging

from sklearn.metrics import silhouette_score, davies_bouldin_score
from ml_framework.data_clustering.clustering import Clustering

from ml_framework.tools.helper_functions import get_workspace_path
from typing import List, Dict, Union


class DBSCAN_Clustering(Clustering):
    """
    DBSCAN_Clustering class for fitting a dbscan-clustering model as implemented in scikit-learn.

    Attributes:
        train_data (pd.DataFrame): The training data.

    Methods:
        fit(nr_iterations: int = 10): Fit the dbscan-clustering model.
    """

    def __init__(
        self,
        train_data: pd.DataFrame = None,
    ):
        """
        Initialize the DBSCAN_Clustering object.

        Args:
            train_data (pd.DataFrame): The training data.
        """
        super().__init__(
            train_data=train_data,
        )
        self.y_clustering = None
        self.X_new = None
        self.y_new = None
        self.model = None
        self.n_clusters = None

    def fit(self, nr_iterations: int = 10):
        """
        Fit the dbscan-clustering model.

        Args:
            nr_iterations (int): The number of iterations.
        """
        plt.switch_backend("agg")

        def optuna_objective_func(trial, train_data):

            params = {
                "eps": trial.suggest_float("eps", 1e-6, 1e6, log=True),
                "min_samples": trial.suggest_int("min_samples", 2, 50, step=1),
                "metric": trial.suggest_categorical(
                    "metric", ["euclidean", "manhattan"]
                ),
            }
            model = sklearn.cluster.DBSCAN(**params).fit(train_data)

            # params = {
            #    "min_samples": trial.suggest_int("min_samples", 2, 50, step=1),
            #    "min_cluster_size": trial.suggest_int(
            #        "min_cluster_size", 20, 500, step=20
            #    ),
            #    "cluster_selection_method": trial.suggest_categorical(
            #        "cluster_selection_method", ["eom", "leaf"]
            #    ),
            #    "metric": trial.suggest_categorical("metric", ["euclidean"]),
            # }
            # model = sklearn.cluster.DBSCAN(**params, copy=True).fit(train_data)

            nr_clusters = len(np.unique(model.labels_))
            cluster_ratios = {"ClusterLabel": [], "ClusterRatio": []}
            for label in np.unique(model.labels_):
                cluster_ratios["ClusterLabel"].append(label)
                cluster_ratio = np.sum(model.labels_ == label) / len(model.labels_)
                cluster_ratios["ClusterRatio"].append(cluster_ratio)

            silhouette_val = -100
            davies_bouldin_val = 100
            if (nr_clusters < 2) or (
                np.sum(np.array(cluster_ratios["ClusterRatio"]) > 0.99) > 0
            ):
                silhouette_val = -100
                davies_bouldin_val = 100
            else:
                silhouette_val = silhouette_score(train_data, model.labels_)
                davies_bouldin_val = davies_bouldin_score(train_data, model.labels_)
                pass

            trial.set_user_attr("model", model)
            trial.set_user_attr("silhouette_val", silhouette_val)
            trial.set_user_attr("davies_bouldin_val", davies_bouldin_val)

            # logging.info(
            #     f"Trial: {trial.number},\t Silhouette_Score: {silhouette_val},\t Davies_Bouldin_Score: {davies_bouldin_val},\tNr.Clusters: {nr_clusters}"
            # )

            return silhouette_val

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(direction="maximize")

        # Start optimizing with specified number of trials
        study.optimize(
            lambda trial: optuna_objective_func(trial, self.X_train),
            n_trials=nr_iterations,
        )

        self.model = study.best_trial.user_attrs["model"]
        self.y_clustering = self.model.labels_
        self.n_clusters = len(np.unique(self.model.labels_))

        davies_bouldin_val = study.best_trial.user_attrs["davies_bouldin_val"]
        silhouette_val = silhouette_score(self.X_train, self.y_clustering)

        # logging.info(
        #     f"DBSCAN, Nr.Clusters: {self.n_clusters}, Silhouette Score: {silhouette_val}, Davies-Bouldin Score: {davies_bouldin_val}"
        # )
        # for label in np.unique(self.y_clustering):
        #     logging.info(
        #         f"Cluster: {label}, Size%: {np.sum(self.y_clustering==label)/len(self.y_clustering)*100:.2f}"
        #     )

        pass

    def predict(self, new_data: pd.DataFrame = None):
        """
        Assigns new data points to one of the clusters.

        Args:
            test_data (pd.DataFrame): The new data points to be assigned to a clust.
        """

        self.X_new = new_data.to_numpy()

        nr_samples = self.X_new.shape[0]

        self.y_new = np.ones(shape=nr_samples, dtype=int) * -1

        for i in range(nr_samples):
            diff = self.model.components_ - self.X_new[i, :]  # NumPy broadcasting

            dist = np.linalg.norm(diff, axis=1)  # Euclidean distance

            shortest_dist_idx = np.argmin(dist)

            if dist[shortest_dist_idx] < self.model.eps:
                self.y_new[i] = self.model.labels_[
                    self.model.core_sample_indices_[shortest_dist_idx]
                ]

        self.n_clusters = len(np.unique(self.model.labels_))

        pass


if __name__ == "__main__":
    pass

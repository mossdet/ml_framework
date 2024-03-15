import pandas as pd
import numpy as np
import optuna
import sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
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
            if nr_clusters > 1:
                silhouette_val = silhouette_score(train_data, model.labels_)
            else:
                silhouette_val = 0

            trial.set_user_attr("model", model)

            print(
                f"Trial: {trial.number},\tSilhouetteScore: {silhouette_val},\tNr.Clusters: {nr_clusters}"
            )
            for k, v in enumerate(trial.params.items()):
                print(f"{k} {v[0]}={v[1]}")

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

        for label in np.unique(self.y_clustering):
            print(f"Cluster: {label}, Size: {np.sum(self.y_clustering==label)}")

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

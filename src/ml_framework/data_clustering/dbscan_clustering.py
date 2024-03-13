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

    def fit(self, nr_iterations: int = 10):
        """
        Fit the dbscan-clustering model.

        Args:
            nr_iterations (int): The number of iterations.
        """
        plt.switch_backend("agg")

        def optuna_objective_func(trial, train_data):
            params = {
                "eps": trial.suggest_float("eps", 1e-6, 1e9, log=True),
                "min_samples": trial.suggest_int("min_samples", 1, 60, step=3),
                "metric": trial.suggest_categorical("metric", ["euclidean"]),
            }

            model = sklearn.cluster.DBSCAN(**params).fit(train_data)

            if len(np.unique(model.labels_)) > 1:
                silhouette_val = silhouette_score(train_data, model.labels_)
            else:
                silhouette_val = 0

            trial.set_user_attr("model", model)

            return silhouette_val

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")

        # Start optimizing with specified number of trials
        study.optimize(
            lambda trial: optuna_objective_func(trial, self.X_train),
            n_trials=nr_iterations,
        )

        self.model = study.best_trial.user_attrs["model"]
        self.y_clustering = self.model.labels_

        pass

    def predict(self, new_data: pd.DataFrame = None):
        """
        Assigns new data points to one of the clusters.

        Args:
            test_data (pd.DataFrame): The new data points to be assigned to a clust.
        """

        self.X_new = new_data.to_numpy()
        # self.y_new_data = self.model.predict(self.X_new)

        nr_samples = self.X_new.shape[0]

        self.y_new_data = np.ones(shape=nr_samples, dtype=int) * -1

        for i in range(nr_samples):
            diff = self.model.components_ - self.X_new[i, :]  # NumPy broadcasting

            dist = np.linalg.norm(diff, axis=1)  # Euclidean distance

            shortest_dist_idx = np.argmin(dist)

            if dist[shortest_dist_idx] < self.model.eps:
                self.y_new_data[i] = self.model.labels_[
                    self.model.core_sample_indices_[shortest_dist_idx]
                ]

        pass


if __name__ == "__main__":
    pass

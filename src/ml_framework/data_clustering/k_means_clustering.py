import pandas as pd
import numpy as np
import optuna
import sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from ml_framework.data_clustering.clustering import Clustering

from ml_framework.tools.helper_functions import get_workspace_path
from typing import List, Dict, Union


class KMeansClustering(Clustering):
    """
    KMeansClustering class for fitting a k-means model as implemented in scikit-learn and using Optuna for hyperparameter optimization.

    Attributes:
        train_data (pd.DataFrame): The training data.

    Methods:
        fit(nr_iterations: int = 10): Fit the k-means model with
            Optuna for hyperparameter optimization.
    """

    def __init__(
        self,
        train_data: pd.DataFrame = None,
    ):
        """
        Initialize the KMeansClustering object.

        Args:
            train_data (pd.DataFrame): The training data.
        """
        super().__init__(
            train_data=train_data,
        )

    def fit(self, nr_iterations: int = 10):
        """
        Fit the k-means model with Optuna for hyperparameter optimization.

        Args:
            nr_iterations (int): The number of iterations for Optuna to search
                for the best hyperparameters.
        """
        plt.switch_backend("agg")

        models_log = {
            "inertia": [],
            "silhouette": [],
            "k": [],
            "model": [],
        }

        early_stop_history_sz = 3
        early_stop_tol = 0.05
        nr_no_improve = 0

        for k in range(2, 10):
            params = {
                "n_clusters": k,
                "init": "k-means++",
                "n_init": 50,
                "max_iter": nr_iterations,
                "random_state": 42,
            }

            model = sklearn.cluster.KMeans(**params).fit(self.X_train)

            silhouette_val = silhouette_score(self.X_train, model.labels_)

            models_log["inertia"].append(model.inertia_)
            models_log["silhouette"].append(silhouette_val)
            models_log["k"].append(k)
            models_log["model"].append(model)

            # print(f"K = {k}, silhouette_score = {silhouette_val}")

            if len(models_log["silhouette"]) > 1:
                score_diff = models_log["silhouette"][-1] / models_log["silhouette"][-2]
                if score_diff < 1 + early_stop_tol:
                    nr_no_improve += 1
                else:
                    nr_no_improve = 0

                if nr_no_improve >= early_stop_history_sz:
                    break

        best_model_idx = np.argmax(models_log["silhouette"])

        # Retrain on training+validation set
        self.model = models_log["model"][best_model_idx]
        self.y_clustering = self.model.labels_
        self.n_clusters = len(np.unique(self.model.labels_))

        # for label in np.unique(self.y_clustering):
        #     print(f"Cluster: {label}, Size: {np.sum(self.y_clustering==label)}")

        self.plot_score_evolution(
            models_log["k"], models_log["silhouette"], models_log["k"][best_model_idx]
        )

        pass

    def plot_score_evolution(
        self,
        k_ls: List[int] = None,
        score_ls: List[float] = None,
        ideal_k: int = None,
    ):
        """
        Plots the evolution of the silhouette score with respect to the number of clusters.

        Args:
            k_ls (List[int]): A list of the number of clusters used in the optimization process.
            score_ls (List[float]): A list of the silhouette scores obtained for each number of clusters.
            ideal_k (int): The number of clusters that gave the best silhouette score.

        Returns:
            None: A plot of the silhouette score versus the number of clusters is saved as an image file.
        """

        plt.plot(k_ls, score_ls)

        plt.ylabel("Silhouette Score")
        plt.xlabel("Nr. Clusters")
        plt.title(f"{type(self).__name__} Clustering Elbow Plot\nIdeal nr. K:{ideal_k}")

        plt.savefig(
            self.images_destination_path + f"Elbow_Plot_{type(self).__name__}.jpeg"
        )
        # plt.show()
        plt.close()


if __name__ == "__main__":
    pass

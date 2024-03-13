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

        def optuna_objective_func(trial, train_data, n_clusters):
            """
            Objective function for Optuna to optimize.

            Args:
                trial: An Optuna trial object.
                X_train (pd.DataFrame): The training features.
                y_train (pd.Series): The training target.
                X_valid (pd.DataFrame): The validation features.
                y_valid (pd.Series): The validation target.

            Returns:
                float: The mean F1 score of the model predictions on the validation set.
            """
            params = {
                "init": trial.suggest_categorical("init", ["k-means++", "random"]),
                "tol": trial.suggest_float("tol", 1e-9, 1e9, log=True),
                "algorithm": trial.suggest_categorical("algorithm", ["lloyd", "elkan"]),
                # Constants
                "n_clusters": trial.suggest_categorical("n_clusters", [n_clusters]),
                "n_init": trial.suggest_categorical("n_init", [10]),
                "max_iter": trial.suggest_categorical("max_iter", [1000]),
                "random_state": trial.suggest_categorical("random_state", [42]),
            }

            model = sklearn.cluster.KMeans(**params).fit(train_data)

            inertia_val = model.inertia_

            trial.set_user_attr("model", model)

            print(
                f"K: {k},\t Trial: {trial.number},\t SilhouetteScore: {silhouette_val}"
            )

            return inertia_val

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        models_log = {
            "inertia": [],
            "silhouette": [],
            "k": [],
            "optuna_trial": [],
            "model": [],
        }
        silhouette_ls = []
        k_ls = []
        early_stop_history_sz = 10
        early_stop_tol = 0.05
        for k in range(2, nr_iterations):
            study = optuna.create_study(direction="minimize")
            # Start optimizing with specified number of trials
            study.optimize(
                lambda trial: optuna_objective_func(trial, self.X_train, k),
                n_trials=nr_iterations,
            )

            model = study.best_trial.user_attrs["model"]
            inertia_val = study.best_trial.value
            silhouette_val = silhouette_score(self.X_train, model.labels_)

            models_log["inertia"].append(inertia_val)
            models_log["silhouette"].append(silhouette_val)
            models_log["k"].append(k)
            models_log["optuna_trial"].append(study.best_trial)
            models_log["model"].append(model)

            silhouette_ls.append(silhouette_val)
            k_ls.append(k)

            print(f"K = {k}, silhouette_score = {silhouette_val}")
            if len(silhouette_ls) > early_stop_history_sz:
                improvement_history = np.diff(silhouette_ls) / silhouette_ls[1:]
                avg_improvement = np.mean(improvement_history)
                if avg_improvement < early_stop_tol:
                    break

        best_model_idx = np.argmax(silhouette_ls)

        # Retrain on training+validation set
        self.model = models_log["model"][best_model_idx]
        self.y_clustering = self.model.labels_
        self.plot_score_evolution(k_ls, silhouette_ls, k_ls[best_model_idx])

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

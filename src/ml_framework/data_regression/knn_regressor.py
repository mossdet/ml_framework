import os
import pandas as pd
import numpy as np
import optuna
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

from ml_framework.data_regression.regressor import Regressor
from ml_framework.tools.helper_functions import get_workspace_path
from typing import List, Dict, Union


class KNN_Regressor(Regressor):
    """
    A class for implementing a K-Nearest Neighbors regressor.

    This class inherits from the Regressor class and provides methods for fitting
    and evaluating a KNN model as implemented in scikit-learn and using Optuna for hyperparameter optimization.

    Attributes:
        target_col_name (str): The name of the target column.
        train_data (pd.DataFrame): The training dataset.
        valid_data (pd.DataFrame): The validation dataset.
        best_k (int): The optimal number of neighbors found during optimization.

    Methods:
        fit(nr_iterations: int) -> None:
            Fits the KNN model to the training data using Optuna for hyperparameter optimization.

        get_best_k() -> int:
            Returns the optimal number of neighbors found during training.

    """

    def __init__(
        self,
        target_col_name: str = None,
        train_data: pd.DataFrame = None,
        valid_data: pd.DataFrame = None,
    ):
        """
        Initializes the KNN_Regressor.

        Args:
            target_col_name (str): The name of the target column.
            train_data (pd.DataFrame): The training dataset.
            valid_data (pd.DataFrame): The validation dataset.
        """
        super().__init__(
            target_col_name=target_col_name,
            train_data=train_data,
            valid_data=valid_data,
        )
        self.best_k = None

    def fit(self, nr_iterations: int = 10):
        """
        Fits the KNN model to the training data using Optuna for hyperparameter optimization.

        Args:
            nr_iterations (int): The number of optimization iterations.

        Returns:
            None
        """

        plt.switch_backend("agg")

        def optuna_objective_func(trial, X_train, y_train, X_valid, y_valid):
            params = {
                "n_neighbors": trial.suggest_int("n_neighbors", 5, 100, step=5),
                "weights": trial.suggest_categorical(
                    "weights", ["uniform", "distance"]
                ),
                "algorithm": trial.suggest_categorical(
                    "algorithm", ["ball_tree", "kd_tree", "brute"]
                ),
                "metric": trial.suggest_categorical(
                    "metric", ["cityblock", "euclidean", "l1", "l2", "manhattan"]
                ),
                "n_jobs": -1,
            }

            model = sklearn.neighbors.KNeighborsRegressor(**params).fit(
                X_train, y_train
            )

            y_predicted = model.predict(X_valid)
            rmse_val = sklearn.metrics.mean_squared_error(y_valid, y_predicted)

            return rmse_val

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")

        # Start optimizing with specified number of trials
        study.optimize(
            lambda trial: optuna_objective_func(
                trial, self.X_train, self.y_train, self.X_valid, self.y_valid
            ),
            n_trials=nr_iterations,
        )

        # Retrain on training+validation set
        X_train_valid = np.concatenate((self.X_train, self.X_valid))
        y_train_valid = np.concatenate((self.y_train, self.y_valid))
        best_trial = study.best_trial
        self.best_k = best_trial.params["n_neighbors"]
        self.model = sklearn.neighbors.KNeighborsClassifier(**best_trial.params).fit(
            X_train_valid, y_train_valid
        )

        print("Best K= ", self.best_k)

    def get_best_k(self):
        """
        Returns the optimal number of neighbors found during training.

        Returns:
            int: The optimal number of neighbors.
        """
        return self.best_k


if __name__ == "__main__":
    pass

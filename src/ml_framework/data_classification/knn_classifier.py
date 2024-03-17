import os
import pandas as pd
import numpy as np
import optuna
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import logging

from ml_framework.data_classification.classifier import Classifier
from ml_framework.tools.helper_functions import get_workspace_path
from typing import List, Dict, Union
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
)


class KNN_Classifier(Classifier):
    """
    A class for implementing a K-Nearest Neighbors classifier.

    This class inherits from the Classifier class and provides methods for fitting
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
        Initializes the KNN_Classifier.

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
                "n_neighbors": trial.suggest_int("n_neighbors", 10, 250, step=10),
                "weights": trial.suggest_categorical(
                    "weights", ["uniform", "distance"]
                ),
                "algorithm": trial.suggest_categorical(
                    "algorithm", ["ball_tree", "kd_tree", "brute"]
                ),
                "metric": trial.suggest_categorical(
                    "metric", ["cityblock", "euclidean", "l1", "l2", "manhattan"]
                ),
                "n_jobs": trial.suggest_categorical("n_jobs", [-1]),
            }

            model = sklearn.neighbors.KNeighborsClassifier(**params).fit(
                X_train, y_train
            )

            y_predicted = model.predict(X_valid)
            f1_val = f1_score(y_valid, y_predicted, average=None)
            f1_val = np.mean(f1_val)

            return f1_val

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")

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

        logging.info("Best K= ", self.best_k)

    def get_best_k(self):
        """
        Returns the optimal number of neighbors found during training.

        Returns:
            int: The optimal number of neighbors.
        """
        return self.best_k


if __name__ == "__main__":
    pass

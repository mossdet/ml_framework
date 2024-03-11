import os
import pandas as pd
import numpy as np
import optuna
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
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


class DecisionTreeClassifier(Classifier):
    """
    A class for implementing a DecisionTree classifier.

    This class inherits from the Classifier class and provides methods for fitting
    and evaluating a decision tree model as implemented in scikit-learn and using Optuna for hyperparameter optimization.

    Attributes:
        target_col_name (str): The name of the target column.
        train_data (pd.DataFrame): The training dataset.
        valid_data (pd.DataFrame): The validation dataset.

    Methods:
        fit(nr_iterations: int) -> None:
            Fits the decision tree model to the training data using Optuna for hyperparameter optimization.
    """

    def __init__(
        self,
        target_col_name: str = None,
        train_data: pd.DataFrame = None,
        valid_data: pd.DataFrame = None,
    ):
        """
        Initializes the DecisionTreeClassifier.

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

    def fit(self, nr_iterations: int = 10):
        """
        Fits the DecisionTree model to the training data using Optuna for hyperparameter optimization.

        Args:
            nr_iterations (int): The number of optimization iterations.

        Returns:
            None
        """

        plt.switch_backend("agg")

        def optuna_objective_func(trial, X_train, y_train, X_valid, y_valid):
            params = {
                "criterion": trial.suggest_categorical(
                    "criterion", ["gini", "entropy"]
                ),
                "splitter": trial.suggest_categorical("splitter", ["best", "random"]),
                "max_depth": trial.suggest_int("max_depth", 1, 10),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2"]
                ),
                "random_state": trial.suggest_categorical("random_state", [42]),
            }

            model = sklearn.tree.DecisionTreeClassifier(**params).fit(X_train, y_train)

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
        self.model = sklearn.tree.DecisionTreeClassifier(**best_trial.params).fit(
            X_train_valid, y_train_valid
        )


if __name__ == "__main__":
    pass

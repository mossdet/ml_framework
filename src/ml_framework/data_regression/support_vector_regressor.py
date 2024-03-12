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


class SupportVectorRegressor(Regressor):
    """
    SupportVectorRegressor class for fitting a support vector regressor model as implemented in scikit-learn and using Optuna for hyperparameter optimization.

    Attributes:
        target_col_name (str): The name of the target column.
        train_data (pd.DataFrame): The training data.
        valid_data (pd.DataFrame): The validation data.
        model: The support vector regressor model.

    Methods:
        fit(nr_iterations: int = 10): Fit the support vector regressor model with
            Optuna for hyperparameter optimization.
    """

    def __init__(
        self,
        target_col_name: str = None,
        train_data: pd.DataFrame = None,
        valid_data: pd.DataFrame = None,
    ):
        """
        Initialize the SupportVectorRegressor object.

        Args:
            target_col_name (str): The name of the target column.
            train_data (pd.DataFrame): The training data.
            valid_data (pd.DataFrame): The validation data.
        """
        super().__init__(
            target_col_name=target_col_name,
            train_data=train_data,
            valid_data=valid_data,
        )

    def fit(self, nr_iterations: int = 10):
        """
        Fit the support vector regressor model with Optuna for hyperparameter optimization.

        Args:
            nr_iterations (int): The number of iterations for Optuna to search
                for the best hyperparameters.
        """

        plt.switch_backend("agg")

        def optuna_objective_func(trial, X_train, y_train, X_valid, y_valid):
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
                "C": trial.suggest_float("C", 1e-10, 1e10, log=True),
                "kernel": trial.suggest_categorical(
                    "kernel", ["linear", "rbf", "sigmoid"]
                ),
                "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
                "max_iter": trial.suggest_categorical("max_iter", [20000]),
            }

            model = sklearn.svm.SVR(**params).fit(X_train, y_train)

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
        self.model = sklearn.svm.SVR(**best_trial.params).fit(
            X_train_valid, y_train_valid
        )

        pass


if __name__ == "__main__":
    pass

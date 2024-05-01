import os
import pandas as pd
import numpy as np
import optuna
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import logging

from xgboost import XGBRegressor
from ml_framework.data_regression.regressor import Regressor

from ml_framework.tools.helper_functions import get_workspace_path
from typing import List, Dict, Union


class XGBoostRegressor(Regressor):
    """
    XGBoostRegressor class for fitting a xgboost regressor model as implemented in scikit-learn and using Optuna for hyperparameter optimization.

    Attributes:
        target_col_name (str): The name of the target column.
        train_data (pd.DataFrame): The training data.
        valid_data (pd.DataFrame): The validation data.
        model: The xgboost regressor model.

    Methods:
        fit(nr_iterations: int = 10): Fit the xgboost regressor model with
            Optuna for hyperparameter optimization.
    """

    def __init__(
        self,
        target_col_name: str = None,
        train_data: pd.DataFrame = None,
        valid_data: pd.DataFrame = None,
    ):
        """
        Initialize the XGBoostRegressor object.

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
        Fit the xgboost regressor model with Optuna for hyperparameter optimization.

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
                "objective": "reg:squarederror",
                "max_depth": trial.suggest_int("max_depth", 5, 100, step=5),
                "n_estimators": trial.suggest_int("n_estimators", 25, 1000, step=25),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-6, 1, log=False
                ),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 1000, log=True),
                # "subsample": trial.suggest_float("subsample", 0.05, 1.0),
                # "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
                # "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            }

            model = XGBRegressor(**params, random_state=0).fit(X_train, y_train)

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
        self.model = XGBRegressor(**best_trial.params, random_state=0).fit(
            X_train_valid, y_train_valid
        )

        pass

    def save_model(self, stored_model_path:str=None)->None:
        # store the XGBoost model
        self.model.save_model(stored_model_path + type(self).__name__ + "SavedModel")

    def load_model(self, stored_model_path:str=None)->None:
        # load the XGBoost model
        self.model = XGBRegressor()
        self.model.load_model(stored_model_path + type(self).__name__ + "SavedModel")

if __name__ == "__main__":
    pass

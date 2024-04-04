import os
import pandas as pd
import numpy as np
import optuna
from xgboost import XGBClassifier
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


class XGBoostClassifier(Classifier):
    """
    XGBoostClassifier class for fitting a xgboost classifier model as implemented in scikit-learn and using Optuna for hyperparameter optimization.

    Attributes:
        target_col_name (str): The name of the target column.
        train_data (pd.DataFrame): The training data.
        valid_data (pd.DataFrame): The validation data.
        model: The xgboost classifier model.

    Methods:
        fit(nr_iterations: int = 10): Fit the xgboost classifier model with
            Optuna for hyperparameter optimization.
    """

    def __init__(
        self,
        target_col_name: str = None,
        train_data: pd.DataFrame = None,
        valid_data: pd.DataFrame = None,
    ):
        """
        Initialize the XGBoostClassifier object.

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
        Fit the xgboost classifier model with Optuna for hyperparameter optimization.

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
                "objective": "binary:logistic",
                "max_depth": trial.suggest_int("max_depth", 1, 20, step=1),
                "n_estimators": trial.suggest_int("n_estimators", 10, 1000, step=10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-3, 1, step=1e-6, log=False
                ),
                "subsample": trial.suggest_float("subsample", 0.05, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            }

            model = XGBClassifier(**params, random_state=0).fit(X_train, y_train)

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
        self.model = XGBClassifier(**best_trial.params, random_state=0).fit(
            X_train_valid, y_train_valid
        )

        pass

    def save_model(self, stored_model_path:str=None)->None:
        # store the XGBoost classifier in XGBoost format (pickle format is not always backwards and forward compatible)
        self.model.save_model(stored_model_path + type(self).__name__ + "SavedModel")

    def load_model(self, stored_model_path:str=None)->None:
        # load the XGBoost classifier in XGBoost format (pickle format is not always backwards and forward compatible)
        self.model = XGBClassifier()
        self.model.load_model(stored_model_path + type(self).__name__ + "SavedModel")


if __name__ == "__main__":
    pass

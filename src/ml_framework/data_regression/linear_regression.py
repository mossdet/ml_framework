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


class LinearRegressor(Regressor):
    """
    LinearRegressor class for fitting a linear regression model as implemented in scikit-learn.

    Attributes:
        target_col_name (str): The name of the target column.
        train_data (pd.DataFrame): The training data.
        valid_data (pd.DataFrame): The validation data.
        model: The linear regression model.

    Methods:
        fit(nr_iterations: int = 10): Fit the linear regression model
    """

    def __init__(
        self,
        target_col_name: str = None,
        train_data: pd.DataFrame = None,
        valid_data: pd.DataFrame = None,
    ):
        """
        Initialize the LinearRegressor object.

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
        Fit the linear regression model with Optuna for hyperparameter optimization.

        Args:
            nr_iterations (int): The number of iterations for Optuna to search
                for the best hyperparameters.
        """

        plt.switch_backend("agg")

        model = sklearn.linear_model.LinearRegression(n_jobs=-1).fit(
            self.X_train, self.y_train
        )
        y_predicted = model.predict(self.X_valid)
        rmse_val = sklearn.metrics.mean_squared_error(self.y_valid, y_predicted)

        # Retrain on training+validation set
        X_train_valid = np.concatenate((self.X_train, self.X_valid))
        y_train_valid = np.concatenate((self.y_train, self.y_valid))
        self.model = sklearn.linear_model.LinearRegression(n_jobs=-1).fit(
            X_train_valid, y_train_valid
        )

        pass


if __name__ == "__main__":
    pass

import pandas as pd

from ml_framework.data_regression.linear_regression import (
    LinearRegressor,
)

"""
from ml_framework.data_regression.knn_regressor import KNN_Regressor
from ml_framework.data_regression.decision_tree_regressor import (
    DecisionTreeRegressor,
)
from ml_framework.data_regression.random_forest_regressor import (
    RandomForestRegressor,
)
from ml_framework.data_regression.support_vector_regressor import (
    SupportVectorRegressor,
)
from ml_framework.data_regression.xgboost_regressor import XGBoostRegressor
from ml_framework.data_regression.ann_regressor_tf import ANN_TF_Regressor
"""


class RunRegression:
    """
    A class for running regression tasks using different regressors.

    Attributes:
        regressor_name (str): The name of the regressor to use.
        target_col_name (str): The name of the target column in the dataset.
        train_data (pd.DataFrame): The training data for the regressor.
        valid_data (pd.DataFrame): The validation data for the regressor.
        model: The regressor model instance.

    Methods:
        fit(nr_iterations): Fits the regressor model to the training data.
        predict(test_data): Predicts the target values for the test data.
        score(): Computes the score of the regressor model.
        plot_scatterplot(): Plots the scatterplot showing the  regressor's predictions and the true values.
    """

    def __init__(
        self,
        regressor_name: str = None,
        target_col_name: str = None,
        train_data: pd.DataFrame = None,
        valid_data: pd.DataFrame = None,
    ):
        """
        Initializes the RunClassification instance.

        Parameters:
            regressor_name (str): The name of the regressor to use.
            target_col_name (str): The name of the target column in the dataset.
            train_data (pd.DataFrame): The training data for the regressor.
            valid_data (pd.DataFrame): The validation data for the regressor.
        """
        self.regressor_name = regressor_name
        self.target_col_name = target_col_name
        self.model = None
        params = {
            "target_col_name": target_col_name,
            "train_data": train_data,
            "valid_data": valid_data,
        }
        self.model = eval(regressor_name + "(target_col_name, train_data, valid_data)")
        pass

    def fit(self, nr_iterations: int = None):
        """
        Fits the regressor model to the training data.

        Parameters:
            nr_iterations (int): The number of iterations for the fitting process.
        """
        self.model.fit(nr_iterations=nr_iterations)

    def predict(self, test_data: pd.DataFrame = None):
        """
        Predicts the target values for the test data.

        Parameters:
            test_data (pd.DataFrame): The test data for prediction.
        """
        self.model.predict(test_data)

    def score(self):
        """
        Computes the score of the regressor model.

        Returns:
            dict: A dictionary containing the score metrics of the regressor.
        """
        score_dict = self.model.score()
        return score_dict

    def plot_scatterplot(self):
        self.model.plot_scatterplot()


if __name__ == "__main__":
    pass

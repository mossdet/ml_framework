import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from typing import List, Dict, Union
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from ml_framework.tools.helper_functions import get_workspace_path


class Regressor:
    """
    A class for performing regression tasks.

    Attributes:
        target_col_name (str): The name of the target column in the dataset.
        train_data (pd.DataFrame): The training data for the classifier.
        valid_data (pd.DataFrame): The validation data for the classifier.
        X_train (np.ndarray): Features of the training data.
        y_train (np.ndarray): Target labels of the training data.
        X_valid (np.ndarray): Features of the validation data.
        y_valid (np.ndarray): Target labels of the validation data.
        y_test (np.ndarray): Predicted target labels.
        model: The trained classifier model.
        y_predicted (np.ndarray): Predicted target labels.
        images_destination_path (str): Path to store generated images.

    Methods:
        fit(): Trains the classifier model.
        predict(test_data): Predicts target labels for the test data.
        get_predicted_values(): Returns the predicted target labels.
        score(): Computes performance metrics of the classifier model.
        plot_scatterplot(): Plots the true and predicted values as a scatterplot.
    """

    def __init__(
        self,
        target_col_name: str = None,
        train_data: pd.DataFrame = None,
        valid_data: pd.DataFrame = None,
    ):
        """
        Initializes the Classifier instance.

        Args:
            target_col_name (str): The name of the target column in the dataset.
            train_data (pd.DataFrame): The training data for the classifier.
            valid_data (pd.DataFrame): The validation data for the classifier.
        """
        self.X_train = train_data.loc[
            :, train_data.columns != target_col_name
        ].to_numpy()
        self.y_train = (
            train_data.loc[:, train_data.columns == target_col_name].to_numpy().ravel()
        )

        self.X_valid = valid_data.loc[
            :, valid_data.columns != target_col_name
        ].to_numpy()
        self.y_valid = (
            valid_data.loc[:, valid_data.columns == target_col_name].to_numpy().ravel()
        )

        self.target_col_name = target_col_name
        self.y_test = None
        self.model = None
        self.y_predicted = None
        self.perf_metrics = None

        self.nr_classes = len(np.unique(self.y_train))

        self.images_destination_path = (
            get_workspace_path() + "Images/Regression/Regression_Modelling_Images/"
        )
        os.makedirs(self.images_destination_path, exist_ok=True)
        print("\n\n****************************************************************")
        print("\nRunning ", type(self).__name__)

        pass

    def fit(self):
        """Trains the regression model."""
        pass

    def predict(self, test_data: pd.DataFrame = None):
        """
        Predicts target values from the test data.

        Args:
            test_data (pd.DataFrame): The test data for prediction.
        """

        X_test = test_data.loc[:, test_data.columns != self.target_col_name].to_numpy()
        self.y_test = (
            test_data.loc[:, test_data.columns == self.target_col_name]
            .to_numpy()
            .ravel()
        )
        self.y_predicted = self.model.predict(X_test)

    def get_predicted_values(self):
        """Returns the predicted target labels."""
        return self.y_predicted

    def score(self):
        """
        Computes performance metrics of the classifier model.

        Returns:
            dict: A dictionary containing the computed performance metrics.
        """
        r2_val = r2_score(self.y_test, self.y_predicted)
        mse_val = mean_squared_error(self.y_test, self.y_predicted)
        rmse_al = root_mean_squared_error(self.y_test, self.y_predicted)
        mae_val = mean_absolute_error(self.y_test, self.y_predicted)
        mape_val = mean_absolute_percentage_error(self.y_test, self.y_predicted)

        print(
            f"\n{type(self).__name__}\nPerformance Metrics",
        )
        score_dict = {
            "R2_Score": r2_val,
            "Mean_Squared_Error": mse_val,
            "Root_Mean_Squared_Error": rmse_al,
            "Mean_Absolute_Error": mae_val,
            "Mean_Absolute_Percentage_Error": mape_val,
        }

        self.perf_metrics = score_dict

        for k, v in score_dict.items():
            print(f"{k}: {v}")

        return score_dict

    def plot_scatterplot(self) -> None:
        """Plots the predicted and real values as a scatterplot."""
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=self.y_predicted, y=self.y_test)
        plt.plot(self.y_test, self.y_test, "r--", linewidth=2)
        plt.ylabel("True Values")
        plt.xlabel("Predicted Values")
        plt.title(
            f"{type(self).__name__} Results\nR2:{self.perf_metrics['R2_Score']:.2f} \n Mean Absolute Percentage Error:{self.perf_metrics['Mean_Absolute_Percentage_Error']:.2f}"
        )

        suffix = "_" + type(self).__name__

        plt.savefig(
            self.images_destination_path + f"Regression_Scatterplot{suffix}.jpeg"
        )
        # plt.show()
        plt.close()


if __name__ == "__main__":
    pass

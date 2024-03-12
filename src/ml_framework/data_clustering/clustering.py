import os
import pandas as pd
import numpy as np
import optuna
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

from typing import List, Dict, Union
from sklearn.metrics import silhouette_samples, silhouette_score

from ml_framework.tools.helper_functions import get_workspace_path


class Clustering:
    """
    A class for performing clustering tasks.

    Attributes:
        target_col_name (str): The name of the target column in the dataset.
        train_data (pd.DataFrame): The training data for the clustering.
        valid_data (pd.DataFrame): The validation data for the clustering.
        X_train (np.ndarray): Features of the training data.
        y_train (np.ndarray): Target labels of the training data.
        X_valid (np.ndarray): Features of the validation data.
        y_valid (np.ndarray): Target labels of the validation data.
        y_test (np.ndarray): Predicted target labels.
        model: The trained clustering model.
        y_predicted (np.ndarray): Predicted target labels.
        confusion_matrix (np.ndarray): Confusion matrix of the predictions.
        images_destination_path (str): Path to store generated images.

    Methods:
        fit(): Trains the clustering model.
        predict(test_data): Predicts target labels for the test data.
        get_predicted_values(): Returns the predicted target labels.
        score(): Computes performance metrics of the clustering model.
        plot_confusion_matrix(): Plots the confusion matrix.
    """

    def __init__(
        self,
        target_col_name: str = None,
        train_data: pd.DataFrame = None,
    ):
        """
        Initializes the Clustering instance.

        Args:
            target_col_name (str): The name of the target column in the dataset.
            train_data (pd.DataFrame): The training data for the clustering.
            valid_data (pd.DataFrame): The validation data for the clustering.
        """
        self.X_train = train_data.to_numpy()
        self.y_train = None
        self.y_new_data = None
        self.model = None

        self.images_destination_path = (
            get_workspace_path() + "Images/Clustering/Clustering_Modelling_Images/"
        )
        os.makedirs(self.images_destination_path, exist_ok=True)
        print("\n\n****************************************************************")
        print("\nRunning ", type(self).__name__)

        pass

    def fit(self):
        """Trains the clustering model."""
        pass

    def predict(self, new_data: pd.DataFrame = None):
        """
        Assigns new data points to one of teh clusters.

        Args:
            test_data (pd.DataFrame): The new data points to be assigned to a clust.
        """

        X_test = new_data.to_numpy()
        self.y_new_data = self.model.predict(X_test)

    def get_predicted_values(self):
        """Returns the cluster labels of new data points"""
        return self.y_new_data

    def score(self):
        """
        Computes the mean Silhouette Coefficient of all samples.

        Returns:
            dict: A dictionary containing the computed performance metrics.
        """

        self.X_all_data
        self.y_all_data
        silh_val = silhouette_score(self.X_all_data, self.y_all_data)

        print(
            f"\n{type(self).__name__}\nPerformance Metrics",
        )
        score_dict = {
            "Silhouette_Coefficient": silh_val,
        }

        for k, v in score_dict.items():
            print(f"{k}: {v}")

        return score_dict


if __name__ == "__main__":
    pass

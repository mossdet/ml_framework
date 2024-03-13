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
        X_train (pd.DataFrame): The training data for the clustering.
        y_clustering (np.ndarray): the cluster labels assigned to each sample.
        X_new (pd.DataFrame): New datapoints to be clustered.
        y_new (np.ndarray): the cluster labels assigned to new data.
        model: The trained clustering model.
        images_destination_path (str): Path to store generated images.

    Methods:
        fit(): Trains the clustering model.
        predict(test_data): Predicts cluster labels for new data.
        get_predicted_values(): Returns the predicted cluster labels.
        score(): Computes the inter and/or intra cluster distance scores.
        plot_clusters(): Plots the clusters using different visualization techniques.
    """

    def __init__(
        self,
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
        self.y_clustering = None
        self.X_new = None
        self.y_new = None
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
        Assigns new data points to one of the clusters.

        Args:
            test_data (pd.DataFrame): The new data points to be assigned to a clust.
        """

        self.X_new = new_data.to_numpy()
        self.y_new_data = self.model.predict(self.X_new)
        pass

    def get_predicted_values(self):
        """Returns the cluster labels of new data points"""
        return self.y_new_data

    def score(self):
        """
        Computes the mean Silhouette Coefficient of all samples.

        Returns:
            dict: A dictionary containing the computed performance metrics.
        """
        # self.X_train, self.y_clustering, self.X_new, self.y_new
        silhouette_val = silhouette_score(self.X_train, self.y_clustering)

        print(
            f"\n{type(self).__name__}\nPerformance Metrics",
        )
        score_dict = {
            "Silhouette_Coefficient": silhouette_val,
        }

        for k, v in score_dict.items():
            print(f"{k}: {v}")

        return score_dict

    def plot_score_evolution(self):
        pass


if __name__ == "__main__":
    pass

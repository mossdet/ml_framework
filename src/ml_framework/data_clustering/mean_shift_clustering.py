import pandas as pd
import numpy as np
import optuna
import sklearn
import matplotlib.pyplot as plt
import logging

from sklearn.metrics import silhouette_score
from ml_framework.data_clustering.clustering import Clustering

from ml_framework.tools.helper_functions import get_workspace_path
from typing import List, Dict, Union


class MeanShiftClustering(Clustering):
    """
    MeanShiftClustering class for fitting a mean-shift-clustering model as implemented in scikit-learn.

    Attributes:
        train_data (pd.DataFrame): The training data.

    Methods:
        fit(nr_iterations: int = 10): Fit the mean-shift-clustering model.
    """

    def __init__(
        self,
        train_data: pd.DataFrame = None,
    ):
        """
        Initialize the MeanShiftClustering object.

        Args:
            train_data (pd.DataFrame): The training data.
        """
        super().__init__(
            train_data=train_data,
        )

    def fit(self, nr_iterations: int = 10):
        """
        Fit the mean-shift-clustering model.

        Args:
            nr_iterations (int): The number of iterations.
        """
        plt.switch_backend("agg")

        # # meanshift
        self.model = sklearn.cluster.MeanShift(
            bandwidth=None,
            seeds=None,
            bin_seeding=False,
            min_bin_freq=1,
            cluster_all=True,
            n_jobs=-1,
            max_iter=nr_iterations,
        )
        self.model = self.model.fit(self.X_train)
        self.y_clustering = self.model.labels_
        self.n_clusters = len(np.unique(self.model.labels_))

        # silhouette_val = silhouette_score(self.X_train, self.model.labels_)
        # logging.info(f"Mean-Shift Clustering\tSilhouette Score = {silhouette_val}")

        # for label in np.unique(self.y_clustering):
        #     logging.info(f"Cluster: {label}, Size: {np.sum(self.y_clustering==label)}")

        pass


if __name__ == "__main__":
    pass

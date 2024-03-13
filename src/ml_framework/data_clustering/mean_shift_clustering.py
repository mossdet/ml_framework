import pandas as pd
import numpy as np
import optuna
import sklearn
import matplotlib.pyplot as plt
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

        # meanshift
        model = sklearn.cluster.MeanShift(
            bandwidth=None,
            bin_seeding=True,
            cluster_all=True,
            max_iter=nr_iterations,
            n_jobs=-1,
        )
        self.model = model.fit(self.X_train)
        self.y_clustering = self.model.labels_

        # Retrain on training+validation set
        # self.model = models_log["model"][best_model_idx]
        # self.y_clustering = self.model.labels_
        # self.plot_score_evolution(k_ls, silhouette_ls, k_ls[best_model_idx])

        pass


if __name__ == "__main__":
    pass

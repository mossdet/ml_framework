import pandas as pd
import numpy as np
import optuna
import sklearn
import matplotlib.pyplot as plt
from ml_framework.data_clustering.clustering import Clustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram

from typing import List, Dict, Union


class AgglomerativeClustering(Clustering):

    def __init__(
        self,
        train_data: pd.DataFrame = None,
    ):
        """
        Initialize the AgglomerativeClustering object.

        Args:
            train_data (pd.DataFrame): The training data.
        """
        super().__init__(
            train_data=train_data,
        )

    def fit(self, nr_iterations: int = None):

        # setting distance_threshold=0 ensures we compute the full tree.
        self.model = sklearn.cluster.AgglomerativeClustering(
            distance_threshold=0, n_clusters=None
        )
        self.model = self.model.fit(self.X_train)
        self.y_clustering = self.model.fit_predict(self.X_train)
        self.plot_score_evolution()

        pass

    def predict(self, new_data: pd.DataFrame = None):
        self.X_new = new_data.to_numpy()
        self.y_new_data = self.model.fit_predict(self.X_new)

    def plot_score_evolution(self):

        def plot_dendrogram(model, **kwargs):
            # Create linkage matrix and then plot the dendrogram

            # create the counts of samples under each node
            counts = np.zeros(model.children_.shape[0])
            n_samples = len(model.labels_)
            for i, merge in enumerate(model.children_):
                current_count = 0
                for child_idx in merge:
                    if child_idx < n_samples:
                        current_count += 1  # leaf node
                    else:
                        current_count += counts[child_idx - n_samples]
                counts[i] = current_count

            linkage_matrix = np.column_stack(
                [model.children_, model.distances_, counts]
            ).astype(float)

            # Plot the corresponding dendrogram
            dendrogram(linkage_matrix, **kwargs)

        plt.figure(figsize=(25, 15))
        plt.title("Hierarchical Clustering Dendrogram")
        # plot the top three levels of the dendrogram
        plot_dendrogram(self.model, truncate_mode="level", p=4)
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        plt.xticks(rotation=90, fontsize=10)

        plt.savefig(
            self.images_destination_path
            + f"Hierarchical_Clustering_Dendrogram_{type(self).__name__}.jpeg"
        )
        # plt.show()
        plt.close()

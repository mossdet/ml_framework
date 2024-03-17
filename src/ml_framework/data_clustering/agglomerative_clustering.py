import pandas as pd
import numpy as np
import optuna
import sklearn
import matplotlib.pyplot as plt
import logging

from ml_framework.data_clustering.clustering import Clustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, ward
from scipy.spatial.distance import pdist

from typing import List, Dict, Union


class AgglomerativeClusteringWard:
    def __init__(self):
        self.clust_dist = None
        self.labels = None
        self.n_clusters = None
        self.centroids = None
        pass

    def fit(self, data: np.ndarray = None, n_clusters: int = None) -> np.ndarray:
        # self.clust_dist = linkage(data, "average", "euclidean")
        self.clust_dist = ward(pdist(data))
        self.labels = fcluster(Z=self.clust_dist, t=n_clusters, criterion="maxclust")
        self.n_clusters = len(np.unique(self.labels))
        self.centroids = []

        for label in np.unique(self.labels):
            self.centroids.append(np.mean(data[self.labels == label], axis=0))
            pass

        self.centroids = np.array(self.centroids)

        return self.labels

    def plot_dendrogram(self, image_filepath: str = None):

        plt.figure(figsize=(25, 15))

        plt.title("Hierarchical Clustering Dendrogram", fontsize=24)
        dendrogram(
            self.clust_dist,
            truncate_mode="lastp",  # show only the last p merged clusters
            p=self.n_clusters,  # show only the last p merged clusters
            labels=self.labels,
            show_leaf_counts=True,  # otherwise numbers in brackets are counts
            leaf_rotation=90.0,
            leaf_font_size=10.0,
            show_contracted=False,  # to get a distribution impression in truncated branches
        )

        plt.xlabel(
            "Number of points in node (or index of point if no parenthesis).",
            fontsize=20,
        )
        plt.xticks(rotation=90, fontsize=16)
        plt.yticks([])

        if image_filepath != None:
            plt.savefig(image_filepath)

        plt.close()

    def predict(self, new_data: np.ndarray = None) -> np.ndarray:
        """
        Assigns new data points to one of the clusters.

        Args:
            test_data (pd.DataFrame): The new data points to be assigned to a clust.
        """

        nr_samples = new_data.shape[0]

        label_new = np.ones(shape=nr_samples, dtype=int) * -1

        for i in range(nr_samples):
            diff = self.centroids - new_data[i, :]

            dist = np.linalg.norm(diff, axis=1)  # Euclidean distance

            shortest_dist_idx = np.argmin(dist)
            label_new[i] = self.labels[shortest_dist_idx]

        return label_new


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

    def fit(self, nr_iterations: int = 10):
        """
        Fit the AgglomerativeClustering model.

        Args:
            nr_iterations (int): The maximum number of clusters to consider.
        """
        plt.switch_backend("agg")

        models_log = {
            "silhouette": [],
            "n_clusters": [],
            "model": [],
        }

        early_stop_history_sz = 3
        early_stop_tol = 0.05
        nr_no_improve = 0

        for n_clusters in range(2, nr_iterations):

            model = AgglomerativeClusteringWard()
            labels = model.fit(self.X_train, n_clusters)
            n_clusters = len(np.unique(labels))

            silhouette_val = silhouette_score(self.X_train, labels)

            models_log["silhouette"].append(silhouette_val)
            models_log["n_clusters"].append(n_clusters)
            models_log["model"].append(model)

            # logging.info(f"K = {n_clusters}, silhouette_score = {silhouette_val}")

            if len(models_log["silhouette"]) > 1:
                score_diff = models_log["silhouette"][-1] / models_log["silhouette"][-2]
                if score_diff < 1 + early_stop_tol:
                    nr_no_improve += 1
                else:
                    nr_no_improve = 0

                if nr_no_improve >= early_stop_history_sz:
                    break

        best_model_idx = np.argmax(models_log["silhouette"])

        # Retrain on training+validation set
        self.model = models_log["model"][best_model_idx]
        self.y_clustering = self.model.labels
        self.n_clusters = self.model.n_clusters

        self.plot_score_evolution(
            models_log["n_clusters"],
            models_log["silhouette"],
            models_log["n_clusters"][best_model_idx],
        )

        image_filepath = (
            self.images_destination_path
            + f"Hierarchical_Clustering_Dendrogram_{type(self).__name__}.jpeg"
        )
        self.model.plot_dendrogram(image_filepath)

        # for label in np.unique(self.y_clustering):
        #     logging.info(f"Cluster: {label}, Size: {np.sum(self.y_clustering==label)}")

        pass

    def plot_score_evolution(
        self,
        k_ls: List[int] = None,
        score_ls: List[float] = None,
        ideal_k: int = None,
    ):
        """
        Plots the evolution of the silhouette score with respect to the number of clusters.

        Args:
            k_ls (List[int]): A list of the number of clusters used in the optimization process.
            score_ls (List[float]): A list of the silhouette scores obtained for each number of clusters.
            ideal_k (int): The number of clusters that gave the best silhouette score.

        Returns:
            None: A plot of the silhouette score versus the number of clusters is saved as an image file.
        """

        plt.plot(k_ls, score_ls)

        plt.ylabel("Silhouette Score")
        plt.xlabel("Nr. Clusters")
        plt.title(f"{type(self).__name__} Clustering Elbow Plot\nIdeal nr. K:{ideal_k}")

        plt.savefig(
            self.images_destination_path + f"Elbow_Plot_{type(self).__name__}.jpeg"
        )
        # plt.show()
        plt.close()

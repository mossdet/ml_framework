import os
import socket
import matplotlib
import pandas as pd
import logging

from collections import defaultdict
from ml_framework.data_clustering.run_eda_clustering import ClusteringEDA
from ml_framework.data_clustering.k_means_clustering import KMeansClustering
from ml_framework.data_clustering.agglomerative_clustering import (
    AgglomerativeClustering,
)
from ml_framework.data_clustering.mean_shift_clustering import MeanShiftClustering
from ml_framework.data_clustering.dbscan_clustering import DBSCAN_Clustering
from ml_framework.tools.helper_functions import get_workspace_path

matplotlib.use("Agg")


def main() -> None:

    # Set Data path
    data_folder_path = ""
    if socket.gethostname() == "LAPTOP-TFQFNF6U":
        data_folder_path = "F:/Weiterbildung/UOC_ML_Bootcamp/Capstone_Projects/Data/"
    elif socket.gethostname() == "DLP":
        data_folder_path = (
            "C:/Users/HFO/Documents/MachineLearning/Capstone_Projects/Data/"
        )

    data_filepath = data_folder_path + "diamonds.csv"

    # Analyzer
    analyzer = ClusteringEDA(data_filepath)
    analyzer.read_data()
    analyzer.clean_data()
    analyzer.encode_data()
    # analyzer.visualize_data(target_col_name=target_col_name)
    train_data, test_data = analyzer.sample_data(train_perc=0.95)

    # Clustering
    clusterings_ls = [
        "KMeansClustering",
        "AgglomerativeClustering",
        "MeanShiftClustering",
        "DBSCAN_Clustering",
    ]

    # clusterings_ls = ["DBSCAN_Clustering"]

    all_models_performance = defaultdict(list)
    for clustering_name in clusterings_ls:
        clustering = eval(clustering_name + "(train_data=train_data)")
        clustering.fit(nr_iterations=100)
        clustering.predict(test_data)
        score_dict = clustering.score()

        all_models_performance["Model"].append(clustering_name)
        for k, v in score_dict.items():
            all_models_performance[k].append(v)
        all_models_performance["NrClusters"].append(clustering.get_num_clusters())

        pass

    perf_df = pd.DataFrame(all_models_performance)
    print(perf_df)

    tables_destination_path = get_workspace_path() + "Tables/"
    os.makedirs(tables_destination_path, exist_ok=True)
    perf_df.to_excel(tables_destination_path + "Clustering_Results.xlsx")
    pass


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s -- %(filename)s %(module)s %(lineno)d",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename="ml_framework_clustering.log",
    )
    # logging.debug("Starting debug logging")
    # logging.info("Starting info logging")
    # logging.warning("Starting warning logging")
    # logging.error("Starting error logging")
    # logging.critical("Starting critical logging")

    main()

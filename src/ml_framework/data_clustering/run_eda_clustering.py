import os
import pandas as pd

from ml_framework.tools.helper_functions import get_workspace_path
from ml_framework.data_analysis import (
    data_ingestion,
    data_cleaning,
    data_encoding,
    data_visualization,
    data_sampling,
)


class ClusteringEDA:
    """
    A class to perform data analysis tasks on a dataset.

    Attributes:
        data_filepath (str): The file path of the dataset.
        images_destination_path (str): The destination path for saving visualization images.
        data (pd.DataFrame): The loaded dataset.
        train_data (pd.DataFrame): The training dataset after sampling.
        valid_data (pd.DataFrame): The validation dataset after sampling.
        test_data (pd.DataFrame): The testing dataset after sampling.
    """

    def __init__(self, data_filepath: str = None):
        """
        Initializes the RunAnalysis object.

        Args:
            data_filepath (str): The file path of the dataset.
        """
        self.data_filepath = data_filepath
        self.images_destination_path = (
            get_workspace_path() + "Images/Clustering/Clustering_EDA_Images/"
        )
        os.makedirs(self.images_destination_path, exist_ok=True)
        self.data = None
        self.train_data = None
        self.test_data = None

    def read_data(self) -> None:
        """
        Reads and ingests the dataset.
        """
        # 1. Ingest Data
        data_ingestor = data_ingestion.DataIngestor(
            self.data_filepath, self.images_destination_path
        )
        self.data = data_ingestor.ingest_data()
        data_ingestor.describe_data(self.data)

    def clean_data(self) -> None:
        """
        Cleans the dataset.
        """
        # 2. Clean Data
        data_cleaner = data_cleaning.DataCleaner()
        # self.data = data_cleaner.replace_missing_data(data=self.data, nan_replace_metrics={0: 'interpolate'})
        self.data = data_cleaner.drop_columns(
            self.data, drop_idxs=[0, 2, 3, 4, 5, 6, 8, 9, 10]
        )  # drop unnamed column
        self.data = data_cleaner.drop_rows(self.data)

    def encode_data(self, target_col_name: str = None) -> None:
        """
        Encodes categorical data and normalizes the dataset.

        Args:
            target_col_name (str): The name of the target column.
        """
        data_encoder = data_encoding.DataEncoder(self.images_destination_path)
        # categ_data_description = data_encoder.describe_categorical_data(self.data)

        # print("Categorical Data Description:")
        # print(pd.DataFrame(categ_data_description))

        # Encode Ordinal Data
        # Worst to best: Fair, Good, Very Good, Premium, Ideal
        # col_name = "cut"
        # cut_encoding = {
        #     "Fair": 0,
        #     "Good": 1,
        #     "Very Good": 2,
        #     "Premium": 3,
        #     "Ideal": 4,
        # }
        # self.data = data_encoder.encode_ordinal_data(
        #     data=self.data, col_name=col_name, categorical_order=cut_encoding
        # )

        # # J (worst) to D (best)
        # col_name = "color"
        # color_encoding = {
        #     "J": 0,
        #     "I": 1,
        #     "H": 2,
        #     "G": 3,
        #     "F": 4,
        #     "E": 5,
        #     "D": 6,
        # }
        # self.data = data_encoder.encode_ordinal_data(
        #     data=self.data, col_name=col_name, categorical_order=color_encoding
        # )

        # # Worst to best: I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF
        # col_name = "clarity"
        # clarity_encoding = {
        #     "I1": 0,
        #     "SI2": 1,
        #     "SI1": 2,
        #     "VS2": 3,
        #     "VS1": 4,
        #     "VVS2": 5,
        #     "VVS1": 6,
        #     "IF": 7,
        # }
        # self.data = data_encoder.encode_ordinal_data(
        #     data=self.data, col_name=col_name, categorical_order=clarity_encoding
        # )

        # Normalize (Z-Score) Data
        self.data = data_encoder.z_score_data(data=self.data)

        pass

    def visualize_data(self, target_col_name: str = None) -> None:
        """
        Visualizes the dataset.

        Args:
            target_col_name (str): The name of the target column.
        """
        # 4. Visualize Data
        data_visualizer = data_visualization.DataVisualizer(
            self.images_destination_path
        )
        data_visualizer.plot_correlation_matrix(data=self.data)
        data_visualizer.plot_histograms_numerical(data=self.data)
        data_visualizer.plot_histograms_categorical(data=self.data)
        # data_visualizer.plot_pairplot(data=self.data, hue=target_col_name)
        data_visualizer.plot_classes_distribution(
            data=self.data, target_col_name=target_col_name
        )

        for class_name in self.data.columns:
            print(class_name)
            data_visualizer.plot_boxplot(
                data=self.data,
                x=target_col_name,
                y=class_name,
                suffix=class_name,
                hue=target_col_name,
            )

    def sample_data(
        self,
        train_perc: float = 0.8,
        valid_perc: float = None,
    ) -> None:
        """
        Samples the dataset into training, validation, and testing sets.

        Args:
            target_col_name (str): The name of the target column.
            train_perc (float): The percentage of data to be used for training.
        """
        # 5. Data Sampling
        data_sampler = data_sampling.DataSampler()

        # df_reduced = data_sampler.sample(self.data, sampling_perc=0.5)
        self.train_data, self.test_data = data_sampler.data_partition(
            self.data, train_perc=train_perc
        )

        self.train_data = data_sampler.shuffle(self.train_data)

        return self.train_data, self.test_data

    def get_data(self) -> pd.DataFrame:
        """
        Returns the loaded dataset.

        Returns:
            pd.DataFrame: The loaded dataset.
        """
        return self.data

    def get_partitioned_data(self) -> pd.DataFrame:
        """
        Returns the partitioned dataset.

        Returns:
            pd.DataFrame: The partitioned dataset.
        """
        return self.train_data, self.test_data


if __name__ == "__main__":
    # Set Data amd Images paths
    data_folder_path = "C:/Users/HFO/Documents/MachineLearning/Capstone_Projects/Data/"
    data_filepath = data_folder_path + "diamonds.csv"

    analyzer = ClusteringEDA(data_filepath)
    df_train, df_test = analyzer.analyze_data()
    pass

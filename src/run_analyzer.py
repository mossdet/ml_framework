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


class Analyzer:
    def __init__(self, data_filepath: str = None):

        self.data_filepath = data_filepath
        self.images_destination_path = (
            get_workspace_path() + "Images/Data_Analysis_Images/"
        )
        os.makedirs(self.images_destination_path, exist_ok=True)
        self.data = None
        self.train_data = None
        self.test_data = None

    def read_data(self) -> None:
        # 1. Ingest Data
        data_ingestor = data_ingestion.DataIngestor(
            self.data_filepath, self.images_destination_path
        )
        self.data = data_ingestor.ingest_data()
        data_ingestor.describe_data(self.data)

    def clean_data(self) -> None:
        # 2. Clean Data
        data_cleaner = data_cleaning.DataCleaner()
        # self.data = data_cleaner.replace_missing_data(data=self.data, nan_replace_metrics={0: 'interpolate'})
        self.data = data_cleaner.drop_columns(
            self.data, drop_idxs=0
        )  # drop unnamed column
        self.data = data_cleaner.drop_rows(self.data)

    def encode_data(self) -> None:
        data_encoder = data_encoding.DataEncoder(self.images_destination_path)
        categ_data_description = data_encoder.describe_categorical_data(self.data)

        # 3.1 Encode Ordinal Data
        # Worst to best: Fair, Good, Very Good, Premium, Ideal
        col_name = "cut"
        cut_encoding = {"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}
        self.data = data_encoder.encode_ordinal_data(
            data=self.data, col_name=col_name, categorical_order=cut_encoding
        )

        # J (worst) to D (best)
        col_name = "color"
        color_encoding = {"J": 1, "I": 2, "H": 3, "G": 4, "F": 5, "E": 6, "D": 7}
        self.data = data_encoder.encode_ordinal_data(
            data=self.data, col_name=col_name, categorical_order=color_encoding
        )

        # Worst to best: I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF
        col_name = "clarity"
        clarity_encoding = {
            "I1": 1,
            "SI2": 2,
            "SI1": 3,
            "VS2": 4,
            "VS1": 5,
            "VVS2": 6,
            "VVS1": 7,
            "IF": 8,
        }
        self.data = data_encoder.encode_ordinal_data(
            data=self.data, col_name=col_name, categorical_order=clarity_encoding
        )

        # 3.2 Encode Nominal Data
        # self.data = data_encoder.encode_nominal_data(data=self.data)

        # 3.3 Encode Target Column Data
        target_col_name = "clarity"
        self.data = data_encoder.encode_target_column(
            data=self.data, target_col_name=col_name
        )

        # 3.4 Normalize (Z-Score) Data
        target_col_name = "clarity"
        self.data = data_encoder.z_score_data(data=self.data, target_col_name=col_name)

        pass

    def visualize_data(self) -> None:
        # 4. Visualize Data
        data_visualizer = data_visualization.DataVisualizer(
            self.images_destination_path
        )
        data_visualizer.plot_correlation_matrix(data=self.data)
        data_visualizer.plot_histograms_numerical(data=self.data)
        data_visualizer.plot_histograms_categorical(data=self.data)
        # data_visualizer.plot_pairplot(data=self.data, hue='clarity')
        data_visualizer.plot_boxplot(data=self.data, x="cut", y="price")
        data_visualizer.plot_classes_distribution(
            data=self.data, target_col_name="clarity"
        )

    def sample_data(self) -> None:

        # 5. Data Sampling
        data_sampler = data_sampling.DataSampler()

        # df_reduced = data_sampler.sample(self.data, sampling_perc=0.5)
        self.train_data, self.test_data = data_sampler.stratified_data_partition(
            self.data, target_col_name="clarity", train_perc=0.8
        )
        # self.train_data = data_sampler.oversample_data(
        #    data=self.train_data, target_col_name="clarity", oversample_factor=-1
        # )
        # self.train_data = data_sampler.synthetic_sampling_SMOTE(
        #    data=self.train_data, target_col_name="clarity"
        # )
        # self.train_data = data_sampler.synthetic_sampling_ADASYN(
        #    data=self.train_data, target_col_name="clarity"
        # )

        self.train_data = data_sampler.shuffle(self.train_data)

        data_visualizer = data_visualization.DataVisualizer(
            self.images_destination_path
        )
        data_visualizer.plot_classes_distribution(
            data=self.train_data, target_col_name="clarity", suffix="train_set"
        )
        data_visualizer.plot_classes_distribution(
            data=self.test_data, target_col_name="clarity", suffix="test_set"
        )

        pass

    def get_data(self) -> pd.DataFrame:
        return self.data

    def get_partitioned_data(self) -> pd.DataFrame:
        return self.train_data, self.test_data


if __name__ == "__main__":
    # Set Data amd Images paths
    data_folder_path = "C:/Users/HFO/Documents/MachineLearning/Capstone_Projects/Data/"
    data_filepath = data_folder_path + "diamonds.csv"

    analyzer = Analyzer(data_filepath)
    df_train, df_test = analyzer.analyze_data()
    pass

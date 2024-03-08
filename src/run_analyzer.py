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

    def analyze_data(self) -> pd.DataFrame:

        ################################################################
        # 1. Ingest Data
        data_ingestor = data_ingestion.DataIngestor(
            self.data_filepath, self.images_destination_path
        )
        df = data_ingestor.read_data()
        data_ingestor.describe_data(df)

        ################################################################
        # 2. Clean Data
        data_cleaner = data_cleaning.DataCleaner()
        # df = data_cleaner.replace_missing_data(data=df, nan_replace_metrics={0: 'interpolate'})
        # df = data_cleaner.drop_rows(df)
        df = data_cleaner.drop_columns(df, drop_idxs=0)

        ################################################################
        # 3. Encode Categorical Data
        data_encoder = data_encoding.DataEncoder(self.images_destination_path)
        categ_data_description = data_encoder.describe_categorical_data(df)

        # 3.1 Encode Ordinal Data
        # Worst to best: Fair, Good, Very Good, Premium, Ideal
        col_name = "cut"
        cut_encoding = {"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}
        df = data_encoder.encode_ordinal_data(
            data=df, col_name=col_name, categorical_order=cut_encoding
        )

        # J (worst) to D (best)
        col_name = "color"
        color_encoding = {"J": 1, "I": 2, "H": 3, "G": 4, "F": 5, "E": 6, "D": 7}
        df = data_encoder.encode_ordinal_data(
            data=df, col_name=col_name, categorical_order=color_encoding
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
        df = data_encoder.encode_ordinal_data(
            data=df, col_name=col_name, categorical_order=clarity_encoding
        )

        # 3.2 Encode Nominal Data
        df_enc = data_encoder.encode_nominal_data(data=df.copy())

        # 3.3 Encode Target Column Data
        col_name = "clarity"
        df_enc = data_encoder.encode_target_column(data=df.copy(), col_name=col_name)

        ################################################################
        # 4. Visualize Data
        data_visualizer = data_visualization.DataVisualizer(
            self.images_destination_path
        )
        data_visualizer.plot_correlation_matrix(data=df)
        data_visualizer.plot_histograms_numerical(data=df)
        data_visualizer.plot_histograms_categorical(data=df)
        # data_visualizer.plot_pairplot(data=df, hue='clarity')
        data_visualizer.plot_boxplot(data=df, x="cut", y="price")
        data_visualizer.plot_classes_distribution(data=df, target_col_name="clarity")

        ################################################################
        # 5. Data Sampling
        data_sampler = data_sampling.DataSampler()
        df_shuffled = data_sampler.shuffle(df)
        df_reduced = data_sampler.sample(df, sampling_perc=0.5)
        df_train, df_test = data_sampler.stratified_data_partition(
            df, target_col_name="clarity", train_perc=0.8
        )

        df_train = data_sampler.oversample_data(
            data=df_train, target_col_name="clarity", oversample_factor=-1
        )

        data_visualizer.plot_classes_distribution(
            data=df_shuffled, target_col_name="clarity", suffix="shuffled"
        )
        data_visualizer.plot_classes_distribution(
            data=df_reduced, target_col_name="clarity", suffix="reduced"
        )
        data_visualizer.plot_classes_distribution(
            data=df_train, target_col_name="clarity", suffix="train"
        )
        data_visualizer.plot_classes_distribution(
            data=df_test, target_col_name="clarity", suffix="test"
        )
        data_visualizer.plot_classes_distribution(
            data=df_train, target_col_name="clarity", suffix="train_oversampled"
        )

        return df


if __name__ == "__main__":
    # Set Data amd Images paths
    data_folder_path = "C:/Users/HFO/Documents/MachineLearning/Capstone_Projects/Data/"
    data_filepath = data_folder_path + "diamonds.csv"

    analyzer = Analyzer(data_filepath)
    df = analyzer.analyze_data()
    pass

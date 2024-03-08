
import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Dict, Union
from ml_framework.tools.helper_functions import get_fileparts


class Analyzer():

    def __init__(self, datafile_path=None):
        self.datafile_path = datafile_path
        self.df = None

    ################################################################
    # Read and describe data
    def read_dataset(self):
        if os.path.isfile(self.datafile_path):
            (path, filename, ext) = get_fileparts(self.datafile_path)
            if ext == 'csv':
                self.df = pd.read_csv(self.datafile_path)
            else:
                print(f"No reader found for file tye: .{ext}")
        else:
            print("Not a valid data file!")

    def describe_data(self):

        print("These are the first 10 entries of the data:")
        print(self.df.head(10))
        print("\n")

        print("Nr. entries:", self.df.shape[0])
        print("Nr. columns:", self.df.shape[1])
        print("\n")

        data_describer = {'ColumnNr': [], 'ColumnName': [],
                          'ColumnType': [], 'NrNaNs': [], 'NrNulls': []}
        for col_idx, col_name in enumerate(self.df.columns):
            data_describer['ColumnNr'].append(col_idx)
            data_describer['ColumnName'].append(col_name)
            data_describer['ColumnType'].append(type(self.df.iloc[0, col_idx]))
            data_describer['NrNaNs'].append(
                self.df.iloc[:, col_idx].isna().sum())
            data_describer['NrNulls'].append(
                self.df.iloc[:, col_idx].isnull().sum())

        describe_df = pd.DataFrame(data_describer)

        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        ax.axis('off')
        table = ax.table(cellText=describe_df.values, colLabels=describe_df.keys(), loc='upper center',
                         cellLoc='center', rowLoc='center', colLoc='center', colWidths=[0.2] * describe_df.shape[1])
        table.set_fontsize(28)
        table.scale(1, 1)
        plt.title(f"Data Description \n\n ({self.datafile_path}) ")
        # mng = plt.get_current_fig_manager()
        # mng.full_screen_toggle()
        plt.show(block=False)

        pass

    ################################################################
    # Clean data
    def clean_data_columns(self, drop_cols_idxs_ls: list, drop_perc: float = 0.2):
        """
        Cleans the data by dropping selected columns and columns with a percentage of missing values higher than drop_perc.
        Parameters:
            None
        Returns:
            None
        """

        for col_idx, col_name in enumerate(self.df.columns):
            nr_rows = self.df.iloc[:, col_idx].shape[0]
            nr_nan = self.df.iloc[:, col_idx].isna().sum(
            ) + self.df.iloc[:, col_idx].isnull().sum()
            nan_perctg = nr_nan/nr_rows
            if nan_perctg > drop_perc:
                drop_cols_idxs_ls.append(col_idx)

        drop_col_names_ls = self.df.columns[drop_cols_idxs_ls]
        print("Drop columns: ", drop_col_names_ls)
        self.df = self.df.drop(columns=drop_col_names_ls)
        pass

    def replace_missing_data(self, column_list: List[str], replace_value: Dict[str, str]):
        pass

    def clean_data_rows(self):
        pre_drop_nr_rows = self.df.shape[0]
        self.df = self.df.dropna()
        self.df = self.df.reset_index(drop=True)
        post_drop_nr_rows = self.df.shape[0]
        print(f"Removed {pre_drop_nr_rows - post_drop_nr_rows} rows")

    ################################################################
    # Encode data

    def describe_categorical_data(self):
        data_describer = {'ColumnNr': [], 'ColumnName': [], 'Categories': []}
        for col_idx, col_name in enumerate(self.df.columns):
            if isinstance(self.df.iloc[0, col_idx], str):
                data_describer['ColumnNr'].append(col_idx)
                data_describer['ColumnName'].append(col_name)
                data_describer['Categories'].append(
                    self.df.iloc[:, col_idx].unique())
        describe_df = pd.DataFrame(data_describer)

        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        ax.axis('off')
        table = ax.table(cellText=describe_df.values, colLabels=describe_df.keys(), loc='upper center',
                         cellLoc='center', rowLoc='center', colLoc='center', colWidths=[0.5] * describe_df.shape[1])
        table.set_fontsize(28)
        table.scale(1, 1)
        plt.title(f"Categorical Data Description \n\n ({self.datafile_path}) ")
        plt.show(block=False)

    def encode_ordinal_data(self, col_name: str = None, categorical_order: List[int] = None):
        pass

    def hot_encode_data(self):
        pass

    def encode_target(self):
        pass

    ################################################################
    # Data Visualization Functions
    def plot_correlation_matrix(self):
        pass

    def plot_pairplot(self):
        pass

    def plot_histograms_numerical(self):
        pass

    def plot_histograms_categorical(self):
        pass

    def plot_boxplot(self):
        pass

    ################################################################
    # Shuffle and sample data
    def shuffle_dataset(self):
        pass

    def sample_dataset(self):
        pass


class DiamondDataAnalyzer(Analyzer):
    def __init__(self, datafile_path=None):
        super().__init__(datafile_path=datafile_path)


if __name__ == '__main__':
    datafile_path = "C:/Users/HFO/Documents/MachineLearning/Capstone_Projects/ml_framework/Data/diamonds.csv"
    analyzer = DiamondDataAnalyzer(datafile_path)
    # Read and describe data
    analyzer.read_dataset()
    analyzer.describe_data()

    # Clean data
    analyzer.clean_data_columns(drop_cols_idxs_ls=[0])
    analyzer.clean_data_rows()

    # Encode data
    analyzer.describe_categorical_data()
    analyzer.encode_ordinal_data()
    analyzer.hot_encode_data()
    analyzer.describe_data()

    pass

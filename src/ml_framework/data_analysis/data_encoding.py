import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from typing import List, Dict, Union
from sklearn.preprocessing import StandardScaler


class DataEncoder:
    """
    A class for encoding and preprocessing data.

    Attributes:
        images_destination_path (str): The destination path to save images.

    Methods:
        describe_categorical_data(data): Describes categorical data and saves a visualization.
        encode_ordinal_data(data, col_name, categorical_order): Encodes ordinal data based on a given order.
        encode_nominal_data(data): Encodes nominal data using one-hot encoding.
        encode_target_column(data, target_col_name): Encodes target column labels.
        z_score_data(data, target_col_name): Performs Z-score normalization on the data.

    """

    def __init__(self, images_destination_path):
        """
        Initializes the DataEncoder instance.

        Args:
            images_destination_path (str): The destination path to save images.
        """
        self.images_destination_path = images_destination_path
        pass

    def describe_categorical_data(self, data):
        """
        Describes categorical data and saves a visualization.

        Args:
            data (pd.DataFrame): The DataFrame containing categorical data.

        Returns:
            dict: A dictionary containing information about the categorical data.
        """
        data_describer = {"ColumnNr": [], "ColumnName": [], "Categories": []}
        for col_idx, col_name in enumerate(data.columns):
            if isinstance(data.iloc[0, col_idx], str):
                data_describer["ColumnNr"].append(col_idx)
                data_describer["ColumnName"].append(col_name)
                data_describer["Categories"].append(data.iloc[:, col_idx].unique())
        describe_df = pd.DataFrame(data_describer)

        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        ax.axis("off")
        table = ax.table(
            cellText=describe_df.values,
            colLabels=describe_df.keys(),
            loc="upper center",
            cellLoc="center",
            rowLoc="center",
            colLoc="center",
            colWidths=[0.2] * describe_df.shape[1],
        )
        table.set_fontsize(28)
        table.scale(1, 1)
        plt.title(f"Categorical Data Description")
        plt.savefig(self.images_destination_path + "categorical_data_description.jpeg")
        # plt.show()
        plt.close()

        nr_categorical_columns = len(data_describer["ColumnNr"])
        for idx in range(nr_categorical_columns):
            print("Column Name:", data_describer["ColumnName"][idx])
            print("Column Number:", data_describer["ColumnNr"][idx])
            print("Categories:")
            categories = data_describer["Categories"][idx]
            cats_ls = data_describer["Categories"][idx].tolist()
            print(sorted(cats_ls))
            print("\n")

        return data_describer

    def encode_ordinal_data(
        self,
        data: pd.DataFrame = None,
        col_name: str = None,
        categorical_order: Dict[str, int] = None,
    ):
        """
        Encodes ordinal data based on a given order.

        Args:
            data (pd.DataFrame): The DataFrame containing the data.
            col_name (str): The name of the column to encode.
            categorical_order (Dict[str, int]): A dictionary mapping categories to their corresponding order.

        Returns:
            pd.DataFrame: The DataFrame with encoded ordinal data.
        """
        data[col_name] = data[col_name].apply(lambda x: categorical_order[x])
        return data

    def encode_nominal_data(self, data: pd.DataFrame = None):
        """
        Encodes nominal data using one-hot encoding.

        Args:
            data (pd.DataFrame): The DataFrame containing the data.

        Returns:
            pd.DataFrame: The DataFrame with encoded nominal data.
        """
        data = pd.get_dummies(data)
        return data

    def encode_target_column(
        self, data: pd.DataFrame = None, target_col_name: str = None
    ):
        """
        Encodes target column labels.

        Args:
            data (pd.DataFrame): The DataFrame containing the data.
            target_col_name (str): The name of the target column.

        Returns:
            pd.DataFrame: The DataFrame with encoded target column labels.
        """
        class_labels = data[target_col_name].unique()
        label_coding = defaultdict(int)
        for idx, class_label in enumerate(class_labels):
            label_coding[class_label] = idx

        data[target_col_name] = data[target_col_name].apply(lambda x: label_coding[x])

        return data

    def z_score_data(self, data: pd.DataFrame = None, target_col_name: str = None):
        """
        Performs Z-score normalization on the data.

        Args:
            data (pd.DataFrame): The DataFrame containing the data.
            target_col_name (str): The name of the target column.

        Returns:
            pd.DataFrame: The DataFrame with Z-score normalized data.
        """
        scaler = StandardScaler()
        data = data.astype("float64")
        data.loc[:, data.columns != target_col_name] = scaler.fit_transform(
            data.loc[:, data.columns != target_col_name]
        )
        return data


if __name__ == "__main__":
    pass

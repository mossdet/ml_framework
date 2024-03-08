import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from typing import List, Dict, Union
from sklearn.preprocessing import StandardScaler


class DataEncoder:

    def __init__(self, images_destination_path):
        self.images_destination_path = images_destination_path
        pass

    def describe_categorical_data(self, data):
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
        data[col_name] = data[col_name].apply(lambda x: categorical_order[x])
        return data

    def encode_nominal_data(self, data: pd.DataFrame = None):
        data = pd.get_dummies(data)
        return data

    def encode_target_column(self, data: pd.DataFrame = None, col_name: str = None):
        """
        Encode target labels with value between 0 and n_classes-1.
        """
        class_labels = data[col_name].unique()
        label_coding = defaultdict(int)
        for idx, class_label in enumerate(class_labels):
            label_coding[class_label] = idx

        data[col_name] = data[col_name].apply(lambda x: label_coding[x])

        return data

    def normalize_data(self, data: pd.DataFrame = None):
        pass
        # scaler = StandardScaler()
        # scaler.fit(np.concatenate((X_train,X_valid),axis=0))
        # X_train = scaler.transform(X_train)


if __name__ == "__main__":
    pass

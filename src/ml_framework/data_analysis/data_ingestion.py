import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Dict, Union
from ml_framework.tools.helper_functions import get_fileparts


class DataIngestor:

    def __init__(self, datafile_path: str = None, images_destination_path: str = None):
        self.datafile_path = datafile_path
        self.images_destination_path = images_destination_path

    # Read and describe data
    def read_data(self) -> pd.DataFrame:
        df = None
        if os.path.isfile(self.datafile_path):
            (path, filename, ext) = get_fileparts(self.datafile_path)
            if ext == "csv":
                df = pd.read_csv(self.datafile_path)
            else:
                print(f"No reader found for file tye: .{ext}")
        else:
            print("Not a valid data file!")

        return df

    def describe_data(self, df: pd.DataFrame = None) -> None:

        # print("These are the first 10 entries of the data:")
        # print(df.head(10))
        # print("\n")

        print("Nr. rows:", df.shape[0])
        print("Nr. columns:", df.shape[1])
        print("\n")

        data_describer = {
            "ColumnNr": [],
            "ColumnName": [],
            "ColumnType": [],
            "NrNaNs": [],
            "NrNulls": [],
        }
        for col_idx, col_name in enumerate(df.columns):
            data_describer["ColumnNr"].append(col_idx)
            data_describer["ColumnName"].append(col_name)
            data_describer["ColumnType"].append(type(df.iloc[0, col_idx]))
            data_describer["NrNaNs"].append(df.iloc[:, col_idx].isna().sum())
            data_describer["NrNulls"].append(df.iloc[:, col_idx].isnull().sum())

        describe_df = pd.DataFrame(data_describer)

        print("\nData Description:")
        print(describe_df)

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
        plt.title(f"Data Description \n\n ({self.datafile_path}) ")
        plt.savefig(self.images_destination_path + "data_description.jpeg")
        # plt.show()
        plt.close()


if __name__ == "__main__":
    pass

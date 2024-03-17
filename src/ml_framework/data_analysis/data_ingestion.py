import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

from typing import List, Dict, Union
from ml_framework.tools.helper_functions import get_fileparts


class DataIngestor:
    """
    A class for ingesting and describing data.

    Attributes:
        datafile_path (str): The path to the data file.
        images_destination_path (str): The destination path to save images.

    Methods:
        ingest_data(): Reads and returns the data from the specified file.
        describe_data(df): Describes the data and saves a visualization.
    """

    def __init__(self, datafile_path: str = None, images_destination_path: str = None):
        """
        Initializes the DataIngestor instance.

        Args:
            datafile_path (str): The path to the data file.
            images_destination_path (str): The destination path to save images.
        """
        self.datafile_path = datafile_path
        self.images_destination_path = images_destination_path

    # Read and describe data
    def ingest_data(self) -> pd.DataFrame:
        """
        Reads and returns the data from the specified file.

        Returns:
            pd.DataFrame: The DataFrame containing the ingested data.
        """
        df = None
        if os.path.isfile(self.datafile_path):
            (path, filename, ext) = get_fileparts(self.datafile_path)
            if ext == "csv":
                df = pd.read_csv(self.datafile_path)
            else:
                logging.info(f"No reader found for file tye: .{ext}")
        else:
            logging.info("Not a valid data file!")

        return df

    def describe_data(self, df: pd.DataFrame = None) -> None:
        """
        Describes the data and saves a visualization.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
        """
        # logging.info("These are the first 10 entries of the data:")
        # logging.info(df.head(10))
        # logging.info("\n")

        logging.info("Nr. rows:", df.shape[0])
        logging.info("Nr. columns:", df.shape[1])
        logging.info("\n")

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

        logging.info("\nData Description:")
        logging.info(describe_df)

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

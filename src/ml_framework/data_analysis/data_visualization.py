import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List, Dict, Union
from ml_framework.tools.helper_functions import get_fileparts


class DataVisualizer:

    def __init__(self, images_destination_path: str = None):
        self.images_destination_path = images_destination_path
        pass

    def plot_correlation_matrix(self, data: pd.DataFrame = None) -> None:
        # plotting correlation heatmap
        corr_mat_df = data.corr()
        plt.figure(figsize=(15, 8))
        sns.heatmap(data=corr_mat_df, annot=True, fmt=".1f", linewidth=0.5)
        plt.savefig(self.images_destination_path + "correlation_matrix.jpeg")
        # plt.show()
        plt.close()

    def plot_pairplot(self, data: pd.DataFrame = None, hue: str = None) -> None:
        # diag_kind="kde" or "hist"
        mrkr_sz = 3
        sns.pairplot(
            data, diag_kind="kde", hue=hue, corner=False, plot_kws={"s": mrkr_sz}
        )
        plt.savefig(self.images_destination_path + "pair_plot.jpeg")
        # plt.show()
        plt.close()

    def plot_histograms_numerical(
        self, data: pd.DataFrame = None, col_name: str = None
    ) -> None:
        df_numerical = data.select_dtypes(include="number")
        columns = df_numerical.columns
        nr_cols = len(columns)
        if nr_cols > 0:
            fig, ax = plt.subplots(nrows=1, ncols=nr_cols, figsize=(20, 10))

            for idx, col_name in enumerate(columns):
                sns.histplot(
                    data=data, x=col_name, kde=True, stat="density", ax=ax[idx]
                )
                if idx > 0:
                    ax[idx].set_ylabel("")
                ax[idx].set_title(col_name)

            fig.suptitle("Numerical Data Distribution Histogramms")
            plt.savefig(self.images_destination_path + "histograms_numerical.jpeg")
            # plt.show()
            plt.close()

    def plot_histograms_categorical(
        self, data: pd.DataFrame = None, col_name: str = None
    ) -> None:
        df_categorical = data.select_dtypes(include="category")
        columns = df_categorical.columns
        nr_cols = len(columns)
        if nr_cols > 0:
            fig, ax = plt.subplots(nrows=1, ncols=nr_cols, figsize=(20, 10))

            for idx, col_name in enumerate(columns):
                sns.histplot(
                    data=data, x=col_name, kde=True, stat="density", ax=ax[idx]
                )
                if idx > 0:
                    ax[idx].set_ylabel("")
                ax[idx].set_title(col_name)

            fig.suptitle("Categorical Data Distribution Histogramms")
            plt.savefig(self.images_destination_path + "histograms_categorical.jpeg")
            # plt.show()
            plt.close()

    def plot_boxplot(
        self,
        data: pd.DataFrame = None,
        x: str = None,
        y: str = None,
        hue: str = None,
    ) -> None:
        sns.boxplot(data=data, x=x, y=y, hue=hue)
        plt.savefig(self.images_destination_path + "boxplot.jpeg")
        # plt.show()
        plt.close()

    def plot_classes_distribution(
        self, data: pd.DataFrame = None, target_col_name: str = None, suffix: str = None
    ) -> None:

        f, axs = plt.subplots(1, 2, figsize=(16, 8))
        # f.tight_layout()

        sns.countplot(
            data=data, x=target_col_name, stat="count", hue=target_col_name, ax=axs[0]
        )
        axs[0].set_ylabel("Sample Count")
        axs[0].set_xlabel("Class")

        sns.countplot(
            data=data, x=target_col_name, stat="percent", hue=target_col_name, ax=axs[1]
        )
        axs[1].set_ylabel("Data Percentage (%)")
        axs[1].set_xlabel("Class")

        plt.suptitle("Samples Distribution across Classes")

        if suffix is None:
            suffix = ""
        else:
            suffix = "_" + suffix

        plt.savefig(
            self.images_destination_path + f"class_distribution_plot{suffix}.jpeg"
        )
        # plt.show()
        plt.close()


if __name__ == "__main__":
    pass

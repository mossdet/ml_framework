import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from typing import List, Dict, Union
from ml_framework.tools.helper_functions import get_fileparts


class DataVisualizer:
    """
    A class for visualizing data using various plotting techniques.

    Attributes:
        images_destination_path (str): The path to save the generated images.

    Methods:
        plot_correlation_matrix(data: pd.DataFrame) -> None:
            Plots a correlation matrix heatmap of the input DataFrame.

        plot_pairplot(data: pd.DataFrame, hue: str) -> None:
            Plots pairwise relationships in the dataset using a pairplot.

        plot_histograms_numerical(data: pd.DataFrame, col_name: str) -> None:
            Plots histograms of numerical data columns.

        plot_histograms_categorical(data: pd.DataFrame, col_name: str) -> None:
            Plots histograms of categorical data columns.

        plot_boxplot(data: pd.DataFrame, x: str, y: str, hue: str, suffix: str) -> None:
            Plots boxplots of data based on specified x and y variables.

        plot_classes_distribution(data: pd.DataFrame, target_col_name: str, suffix: str) -> None:
            Plots the distribution of classes in the dataset.

    """

    def __init__(self, images_destination_path: str = None):
        """
        Initializes the DataVisualizer class with the destination path for images.

        Args:
            images_destination_path (str): The path to save the generated images.
        """
        self.images_destination_path = images_destination_path
        pass

    def plot_correlation_matrix(self, data: pd.DataFrame = None) -> None:
        """
        Plots a correlation matrix heatmap of the input DataFrame.

        Args:
            data (pd.DataFrame): The input DataFrame.

        Returns:
            None
        """
        # plotting correlation heatmap
        corr_mat_df = data.corr()
        plt.figure(figsize=(15, 8))
        sns.heatmap(data=corr_mat_df, annot=True, fmt=".1f", linewidth=0.5)
        plt.savefig(self.images_destination_path + "correlation_matrix.jpeg")
        # plt.show()
        plt.close()

    def plot_pairplot(self, data: pd.DataFrame = None, hue: str = None) -> None:
        """
        Plots pairwise relationships in the dataset using a pairplot.

        Args:
            data (pd.DataFrame): The input DataFrame.
            hue (str): The variable to map plot aspects to different colors.

        Returns:
            None
        """
        # diag_kind="kde" or "hist"
        mrkr_sz = 3
        sns.pairplot(
            data, diag_kind="kde", hue=hue, corner=False, plot_kws={"s": mrkr_sz}
        )
        plt.savefig(self.images_destination_path + "pair_plot.jpeg")
        # plt.show()
        plt.close()

    def plot_histograms_numerical(
        self,
        data: pd.DataFrame = None,
        col_name: str = None,
    ) -> None:
        """
        Plots histograms of numerical data columns.

        Args:
            data (pd.DataFrame): The input DataFrame.
            col_name (str): The name of the column to plot.

        Returns:
            None
        """

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
        self,
        data: pd.DataFrame = None,
        col_name: str = None,
    ) -> None:
        """
        Plots histograms of categorical data columns.

        Args:
            data (pd.DataFrame): The input DataFrame.
            col_name (str): The name of the column to plot.

        Returns:
            None
        """
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
        suffix: str = None,
    ) -> None:
        """
        Plots boxplots of data based on specified x and y variables.

        Args:
            data (pd.DataFrame): The input DataFrame.
            x (str): The variable to be plotted on the x-axis.
            y (str): The variable to be plotted on the y-axis.
            hue (str): Optional variable to plot separate boxes for each level of the hue variable.
            suffix (str): An optional suffix to add to the plot filename.

        Returns:
            None
        """

        if suffix is None:
            suffix = ""
        else:
            suffix = "_" + suffix

        sns.boxplot(data=data, x=x, y=y, hue=hue)
        plt.title(f"Boxplot\n{suffix}")
        plt.savefig(self.images_destination_path + f"boxplot{suffix}.jpeg")
        # plt.show()
        plt.close()

    def plot_classes_distribution(
        self,
        data: pd.DataFrame = None,
        target_col_name: str = None,
        suffix: str = None,
    ) -> None:
        """
        Plots the distribution of classes in the dataset.

        Args:
            data (pd.DataFrame): The input DataFrame.
            target_col_name (str): The name of the target column containing class labels.
            suffix (str): An optional suffix to add to the plot filename.

        Returns:
            None
        """

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

    def plot_performance_radar_chart(
        self,
        performance_dict: Dict[str, Union[float, str]],
    ):
        nr_models = len(performance_dict["Model"])
        categories = list(performance_dict.keys())
        categories.remove("Model")

        fig = go.Figure()

        for model_name in performance_dict["Model"]:
            r = [v[0] for k, v in performance_dict.items() if k != "Model"]
            fig.add_trace(
                go.Scatterpolar(r=r, theta=categories, fill="toself", name=model_name)
            )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
        )

        fig.show()


if __name__ == "__main__":
    pass

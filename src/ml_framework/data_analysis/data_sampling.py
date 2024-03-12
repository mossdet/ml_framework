import os
import pandas as pd
import numpy as np
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE, ADASYN
from typing import List, Dict, Union
from ml_framework.tools.helper_functions import get_fileparts


class DataSampler:
    """
    A class for sampling and preprocessing data.

    Methods:
        shuffle(data): Shuffle the rows of the DataFrame.
        sample(data, sampling_perc): Sample a percentage of rows from the DataFrame.
        stratified_data_partition(data, target_col_name, train_perc): Split the data into train and test sets.
        oversample_data(data, target_col_name, oversample_factor): Oversample the minority class.
        synthetic_sampling_SMOTE(data, target_col_name): Perform Synthetic Minority Over-sampling Technique (SMOTE).
        synthetic_sampling_ADASYN(data, target_col_name): Perform Adaptive Synthetic Sampling (ADASYN).
    """

    def __init__(self):
        """
        Initializes the DataSampler instance.
        """
        pass

    def shuffle(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Shuffle the rows of the DataFrame.

        Args:
            data (pd.DataFrame): The DataFrame to be shuffled.

        Returns:
            pd.DataFrame: The shuffled DataFrame.
        """
        idxs = np.arange(len(data))
        np.random.shuffle(idxs)

        return data.iloc[idxs, :].reset_index(drop=True)

    def sample(
        self,
        data: pd.DataFrame = None,
        sampling_perc: float = None,
    ) -> pd.DataFrame:
        """
        Sample a percentage of rows from the DataFrame.

        Args:
            data (pd.DataFrame): The DataFrame to be sampled.
            sampling_perc (float): The percentage of rows to sample.

        Returns:
            pd.DataFrame: The sampled DataFrame.
        """
        idxs = np.arange(len(data))

        np.random.shuffle(idxs)
        nr_samples = np.int64(np.round(len(data) * sampling_perc))

        idxs = idxs[0:nr_samples]

        return data.iloc[idxs, :].reset_index(drop=True)

    def data_partition(
        self,
        data: pd.DataFrame = None,
        train_perc: float = None,
    ) -> pd.DataFrame:
        """
        Split the data into train and test sets.

        Args:
            data (pd.DataFrame): The DataFrame to be split.
            train_perc (float): The percentage of data to be used for training.

        Returns:
            pd.DataFrame: The train DataFrame.
            pd.DataFrame: The test DataFrame.
        """

        nr_rows = data.shape[0]
        idxs = np.arange(nr_rows)
        np.random.shuffle(idxs)
        train_sel_idxs = idxs[0 : np.int64(nr_rows * train_perc)]
        test_sel_idxs = idxs[np.int64(nr_rows * train_perc) :]

        train_df = data.iloc[train_sel_idxs, :].copy()
        test_df = data.iloc[test_sel_idxs, :].copy()

        train_df.reset_index(drop=True)
        test_df.reset_index(drop=True)

        return train_df, test_df

    def stratified_data_partition(
        self,
        data: pd.DataFrame = None,
        target_col_name: str = None,
        train_perc: float = None,
    ) -> pd.DataFrame:
        """
        Split the data into train and test sets.

        Args:
            data (pd.DataFrame): The DataFrame to be split.
            target_col_name (str): The name of the target column.
            train_perc (float): The percentage of data to be used for training.

        Returns:
            pd.DataFrame: The train DataFrame.
            pd.DataFrame: The test DataFrame.
        """

        classes_ls = np.sort(data[target_col_name].unique())
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()
        for class_name in classes_ls:
            class_size = np.sum(data[target_col_name] == class_name)
            idxs = np.arange(class_size)
            np.random.shuffle(idxs)
            train_sel_idxs = idxs[0 : np.int64(class_size * train_perc)]
            test_sel_idxs = idxs[np.int64(class_size * train_perc) :]
            train_df = pd.concat(
                [
                    train_df,
                    data[data[target_col_name] == class_name].iloc[train_sel_idxs, :],
                ]
            )
            test_df = pd.concat(
                [
                    test_df,
                    data[data[target_col_name] == class_name].iloc[test_sel_idxs, :],
                ]
            )

        train_df.reset_index(drop=True)
        test_df.reset_index(drop=True)

        return train_df, test_df

    def oversample_data(
        self,
        data: pd.DataFrame = None,
        target_col_name: str = None,
        oversample_factor: int = None,
    ) -> pd.DataFrame:
        """
        Oversample the minority class.

        Args:
            data (pd.DataFrame): The DataFrame to be oversampled.
            target_col_name (str): The name of the target column.
            oversample_factor (int): The factor by which to oversample the minority class.

        Returns:
            pd.DataFrame: The oversampled DataFrame.
        """

        # Get majority class label
        classes_ls = np.sort(data[target_col_name].unique())
        majority_class_label = None
        maj_class_size = 0
        for class_name in classes_ls:
            if np.sum(data[target_col_name] == class_name) > maj_class_size:
                maj_class_size = np.sum(data[target_col_name] == class_name)
                majority_class_label = class_name

        # Get array of minority class labels
        minority_class_labels_ls = np.sort(
            classes_ls[
                [class_label != majority_class_label for class_label in classes_ls]
            ]
        )

        # Add oversample_factor samples from each minority class to the original data
        match_majority_class = oversample_factor <= 0
        ovsmpl_data = data.copy()
        for class_name in minority_class_labels_ls:
            class_sel = data[target_col_name] == class_name
            class_size = np.sum(class_sel)
            idxs = np.arange(class_size)

            if match_majority_class:
                oversample_factor = float(maj_class_size / class_size)

            ovsmpl_idxs = np.random.choice(
                class_size,
                np.int64(oversample_factor * class_size) - class_size,
                replace=True,
            )
            ovsmpl_data = pd.concat(
                [ovsmpl_data, ovsmpl_data[class_sel].iloc[ovsmpl_idxs, :]]
            )

            pass

        return ovsmpl_data.reset_index(drop=True)

    def synthetic_sampling_SMOTE(
        self,
        data: pd.DataFrame = None,
        target_col_name: str = None,
    ) -> pd.DataFrame:
        """
        Perform Synthetic Minority Over-sampling Technique (SMOTE).

        Args:
            data (pd.DataFrame): The DataFrame to be oversampled.
            target_col_name (str): The name of the target column.

        Returns:
            pd.DataFrame: The oversampled DataFrame.
        """

        X = data.loc[:, data.columns != target_col_name]
        y = data.loc[:, data.columns == target_col_name]
        oversample = SMOTE()
        X_synthetic, y_synthetic = oversample.fit_resample(X, y)
        # X_synthetic, y_synthetic = SMOTE(random_state=42).fit_resample(X, y)
        synth_data = np.hstack((X_synthetic, y_synthetic))
        synth_cols = X_synthetic.columns.to_list() + y_synthetic.columns.to_list()
        synth_data_df = pd.DataFrame(columns=synth_cols, data=synth_data)
        print(
            "Class distribution pre-SMOTE:\n",
            [
                (class_label, sum(y.to_numpy() == class_label)[0])
                for class_label in np.unique(y)
            ],
        )
        print(
            "Class distribution post-SMOTE:\n",
            [
                (
                    class_label,
                    sum(synth_data_df[target_col_name].to_numpy() == class_label),
                )
                for class_label in np.unique(synth_data_df[target_col_name])
            ],
        )

        return synth_data_df

    def synthetic_sampling_ADASYN(
        self,
        data: pd.DataFrame = None,
        target_col_name: str = None,
    ) -> pd.DataFrame:
        """
        Perform Adaptive Synthetic Sampling (ADASYN).

        Args:
            data (pd.DataFrame): The DataFrame to be oversampled.
            target_col_name (str): The name of the target column.

        Returns:
            pd.DataFrame: The oversampled DataFrame.
        """
        X = data.loc[:, data.columns != target_col_name]
        y = data.loc[:, data.columns == target_col_name]
        oversample = ADASYN()
        X_synthetic, y_synthetic = oversample.fit_resample(X, y)
        # X_synthetic, y_synthetic = SMOTE(random_state=42).fit_resample(X, y)
        synth_data = np.hstack((X_synthetic, y_synthetic))
        synth_cols = X_synthetic.columns.to_list() + y_synthetic.columns.to_list()
        synth_data_df = pd.DataFrame(columns=synth_cols, data=synth_data)
        print(
            "Class distribution pre-SMOTE:\n",
            [
                (class_label, sum(y.to_numpy() == class_label)[0])
                for class_label in np.unique(y)
            ],
        )
        print(
            "Class distribution post-SMOTE:\n",
            [
                (
                    class_label,
                    sum(synth_data_df[target_col_name].to_numpy() == class_label),
                )
                for class_label in np.unique(synth_data_df[target_col_name])
            ],
        )

        return synth_data_df


if __name__ == "__main__":
    pass

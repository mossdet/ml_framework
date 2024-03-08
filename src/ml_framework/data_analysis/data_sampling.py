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

    def __init__(self):
        pass

    def shuffle(self, data: pd.DataFrame = None) -> pd.DataFrame:

        idxs = np.arange(len(data))
        np.random.shuffle(idxs)

        return data.iloc[idxs, :].reset_index(drop=True)

    def sample(
        self, data: pd.DataFrame = None, sampling_perc: float = None
    ) -> pd.DataFrame:
        idxs = np.arange(len(data))

        np.random.shuffle(idxs)
        nr_samples = np.int64(np.round(len(data) * sampling_perc))

        idxs = idxs[0:nr_samples]

        return data.iloc[idxs, :].reset_index(drop=True)

    def stratified_data_partition(
        self,
        data: pd.DataFrame = None,
        target_col_name: str = None,
        train_perc: float = None,
    ) -> pd.DataFrame:
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

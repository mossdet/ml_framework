import pandas as pd
import numpy as np
from typing import List, Dict, Union

# This is an example of a dictionary that can be pass as replace value so that whenever a NaN is encountered, the dictionary defined which action to take
replace_value_dict = {"age": "average", "salary": "median"}


def replace_missing_data(
    df: Union[np.ndarray, pd.DataFrame],
    column_list: List[str],
    replace_value: Dict[str, str],
) -> pd.DataFrame:
    """
    drops row with missing data from dataframe
    :param df: input pandas dataframe
    :param column_list: list of columns to check for missing data
    :return: dataframe with no missing data
    """
    if isinstance(df, np.ndarray):
        pass
    elif isinstance(df, pd.DataFrame):
        pass
    pass


class Analyzer:
    def __init__(self, data_path) -> None:
        self.data_path = data_path
        self.df = None

    def read_dataset(self):
        self.df = pd.read_csv(self, self.data_path)

    def pair_plot(self, left_column_name: List[str], right_column_name: List[str]):
        pass


class Classifier:
    def __init__(self, estimator_name):
        pass

    def fit(self, df: pd.DataFrame):
        pass

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        self.predict(df)

    def score(
        self, predicted_label: nd.array, actual_label: nd.array, list_metrics: List[str]
    ) -> Dict[str, Union[float, np.ndarray]]:
        metrics_dict = dict
        for metric in list_metrics:
            score_val = actual_label - predicted_label
            metrics_dict[metric] = score_val

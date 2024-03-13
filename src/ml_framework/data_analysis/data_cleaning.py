import os
import pandas as pd
import numpy as np

from typing import List, Dict, Union


class DataCleaner:

    def __init__(self):
        pass

    def replace_missing_data(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        nan_replace_metrics: Union[str, Dict[int, str]],
    ) -> pd.DataFrame:
        """
        Replaces missing values in columns of a numpy array or a pandas DataFrame.
        If  data is a numpy array, then nan_replace_metrics must be a string specifying the interpolation-operation used to fill missing values
        If  data is a pandas DataFrame, then nan_replace_metrics must be a dictionary specifying pairs of column_nr and interpolation-operation to replace missing values

        Parameters
        ----------
        data : Union[np.ndarray, pd.DataFrame]
            input numpy array or pandas DataFrame
        nan_replace_metrics : Union[str, Dict[int, str]]
            name of the interpolation-operation used to fill missing values
            or
            dictionary specifying pairs of column_name and interpolation-operation to replace missing values

        Returns
        -------
        pd.DataFrame
            dataframe with no missing data
        """
        if isinstance(data, np.ndarray):
            data = self.__replace_missing_data_array(data, nan_replace_metrics)
        elif isinstance(data, pd.DataFrame):
            if isinstance(nan_replace_metrics, Dict):
                for k, this_metric in nan_replace_metrics.items():
                    data_array = data.iloc[:, k].to_numpy()

                    if np.sum(np.isnan(data_array)) > 0:
                        data_array = self.__replace_missing_data_array(
                            data_array, this_metric
                        )
                        data.iloc[:, k] = data_array
                    pass
            else:
                print("Wrong nan_replace_metrics type!")
        pass

        return data

    def __replace_missing_data_array(
        self, data: np.ndarray, nan_replace_metrics: str
    ) -> np.ndarray:
        """
        Replaces missing values in columns of a numpy array.
        If  data is a numpy array, then nan_replace_metrics must be a string specifying the interpolation-operation used to fill missing values
        """
        if isinstance(nan_replace_metrics, str) and nan_replace_metrics in [
            "mean",
            "median",
            "interpolate",
        ]:
            non_nan_values = data[np.logical_not(np.isnan(data))]
            if not isinstance(non_nan_values[0], str):
                if nan_replace_metrics in ["mean", "median"]:
                    try:
                        op_str = f"pd.Series(data).{nan_replace_metrics}(skipna=True)"
                        replace_value = eval(op_str)
                        data[np.isnan(data)] = replace_value
                    except:
                        this_pyfile_name = os.path.split(os.path.abspath(__file__))[1]
                        print(
                            f"An exception occurred in {this_pyfile_name}, __replace_missing_data_array"
                        )
                else:
                    data = pd.Series(data).interpolate().to_numpy()
                    pass

            else:
                print("Wrong data type in the selected column!")

        else:
            print("Wrong nan_replace_metrics type!")
        return data

    def drop_columns(
        self,
        data: pd.DataFrame = None,
        drop_idxs: Union[int, List[int]] = None,
        drop_perc: float = 1,
    ):
        """
        Cleans the data by dropping selected columns and columns with a percentage of missing values higher than drop_perc.
        Parameters:
            None
        Returns:
            None
        """

        # if int, convert to list of int
        if isinstance(drop_idxs, int):
            drop_idxs = [drop_idxs]

        if drop_perc < 1:
            nr_rows = data.shape[0]
            for col_idx, col_name in enumerate(data.columns):
                nr_nan = (
                    data.iloc[:, col_idx].isna().sum()
                    + data.iloc[:, col_idx].isnull().sum()
                )
                nan_perctg = nr_nan / nr_rows
                if nan_perctg > drop_perc:
                    drop_idxs.append(col_idx)

        if drop_idxs is not None:
            drop_col_names_ls = data.columns[drop_idxs]
            print("Drop columns: ", drop_col_names_ls)
            data = data.drop(columns=drop_col_names_ls)
        else:
            print("No columns to drop!")

        return data

    def drop_rows(self, data: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        """
        Drops rows from a numpy array or pandas DataFrame that contain missing values.

        Parameters
        ----------
        data : Union[np.ndarray, pd.DataFrame]
            Input numpy array or pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            Pandas DataFrame with dropped rows.
        """

        if isinstance(data, np.ndarray):
            data = data[np.logical_not(np.isnan(data))]
        elif isinstance(data, pd.DataFrame):
            data = data.dropna()

        return data


if __name__ == "__main__":
    pass

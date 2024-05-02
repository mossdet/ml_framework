import os
import pandas as pd
import numpy as np
import logging

from typing import List, Dict, Union


class DataCleaner:

    def __init__(self):
        pass

    def replace_missing_data(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        nan_replace_metrics: Union[str, Dict[int, str]],
    ) -> Union[np.ndarray, pd.DataFrame]:
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

        valid_replacement_metrics = ["mean", "median", "interpolate"]

        if isinstance(data, np.ndarray):

            # replace inf by nan
            data[np.logical_or(data == np.inf, data == -np.inf)] = np.nan

            # Test array dimensions
            if data.shape != (data.shape[0],):
                logging.info("Wrong array shape!")
                return None

            # Test replacement metrics correctness for array
            if not isinstance(nan_replace_metrics, str):
                logging.info("Wrong array nan_replace_metrics type!")
                print("Wrong array nan_replace_metrics type!")
                return None
            else:
                if nan_replace_metrics not in valid_replacement_metrics:
                    logging.info("Invalid array nan_replace_metric!")
                    return None

            data = self.__replace_missing_data_array(data, nan_replace_metrics)
        elif isinstance(data, pd.DataFrame):

            # replace inf by nan
            data.replace([np.inf, -np.inf], np.nan)

            # Test replacement metrics correctness for DataFrame
            if not isinstance(nan_replace_metrics, Dict):
                return None
            else:
                if len(nan_replace_metrics.keys()) != data.shape[1]:
                    logging.info(
                        "A replacement metric must be defined for each column!"
                    )
                else:
                    for k, v in nan_replace_metrics.items():
                        if not isinstance(v, str):
                            logging.info("Wrong DataFrame nan_replace_metrics type!")
                            return None
                        else:
                            if v not in valid_replacement_metrics:
                                logging.info("Invalid DataFrame nan_replace_metric!")
                                return None

            for k, this_metric in nan_replace_metrics.items():
                data_array = data.iloc[:, k].to_numpy()
                if np.sum(np.isnan(data_array)) > 0:
                    data_array = self.__replace_missing_data_array(
                        data_array, this_metric
                    )
                    data.iloc[:, k] = data_array

        return data

    def __replace_missing_data_array(
        self, data: np.ndarray, nan_replace_metric: str
    ) -> np.ndarray:
        """
        Replaces missing values in columns of a numpy array.
        If  data is a numpy array, then nan_replace_metric must be a string specifying the interpolation-operation used to fill missing values
        """

        non_nan_values = data[np.logical_not(np.isnan(data))]
        if not isinstance(non_nan_values[0], str):
            if nan_replace_metric in ["mean", "median"]:
                try:
                    op_str = f"pd.Series(data).{nan_replace_metric}(skipna=True)"
                    replace_value = eval(op_str)
                    data[np.isnan(data)] = replace_value
                except:
                    this_pyfile_name = os.path.split(os.path.abspath(__file__))[1]
                    logging.info(
                        f"An exception occurred in {this_pyfile_name}, __replace_missing_data_array"
                    )
                    print(
                        f"An exception occurred in {this_pyfile_name}, __replace_missing_data_array"
                    )
            elif nan_replace_metric == "interpolate":
                if np.isnan(data[0]):
                    data[0] = non_nan_values[0]
                if np.isnan(data[-1]):
                    data[-1] = non_nan_values[-1]
                data = pd.Series(data).interpolate().to_numpy()
                pass

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

        if drop_idxs == None and drop_perc == 1:
            logging.info("No columns to drop!")
            return data

        # replace inf by nan
        data.replace([np.inf, -np.inf], np.nan)

        if drop_idxs is None and drop_perc is not None:
            drop_idxs = []

        # if int, convert to list of int
        if isinstance(drop_idxs, int):
            drop_idxs = [drop_idxs]

        if drop_perc < 1:
            nr_rows = data.shape[0]
            for col_idx, col_name in enumerate(data.columns):
                nr_nan = data[[col_name]].isna().sum().to_numpy()[0]
                nan_perctg = nr_nan / nr_rows
                if nan_perctg > drop_perc:
                    drop_idxs.append(col_idx)

        if drop_idxs is not None:
            drop_idxs = np.unique(drop_idxs)
            drop_col_names_ls = data.columns[drop_idxs]
            logging.info("Drop columns: ", drop_col_names_ls)
            data = data.drop(columns=drop_col_names_ls)

        return data

    def drop_rows(
        self,
        data: pd.DataFrame,
        drop_idxs: Union[int, List[int]] = None,
    ) -> pd.DataFrame:
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

        # replace inf by nan
        data.replace([np.inf, -np.inf], np.nan)

        # if int, convert to list of int
        if isinstance(drop_idxs, int):
            drop_idxs = [drop_idxs]

        if drop_idxs is not None:
            drop_idxs = np.unique(drop_idxs)
            data = data.drop(data.index[drop_idxs][0]).reset_index(drop=True)
            logging.info("Drop Rows")
        else:
            logging.info("No rows to drop!")

        return data


if __name__ == "__main__":
    pass

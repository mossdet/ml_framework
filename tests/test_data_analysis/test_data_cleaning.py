import pytest
import pandas as pd
import numpy as np
from ml_framework.data_analysis.data_cleaning import DataCleaner


@pytest.fixture
def data_cleaner():
    return DataCleaner()


def test_replace_missing_data_dataframe_mean(data_cleaner):
    data = pd.DataFrame([[1.0, np.nan, 3], [4, 5, np.nan], [7, 8, 9]])
    nan_replace_metrics = {0: "mean", 1: "mean", 2: "mean"}
    expected_result = pd.DataFrame([[1.0, 6.5, 3.0], [4, 5, 6], [7, 8, 9]])
    result = data_cleaner.replace_missing_data(data.copy(), nan_replace_metrics)
    pd.testing.assert_frame_equal(result, expected_result)


def test_replace_missing_data_dataframe_median(data_cleaner):
    data = {"col1": [0, 1, 2, np.nan], "col2": [0, 1, 2, 3], "col3": [np.nan, 1, 2, 3]}
    data = pd.DataFrame(data).astype(float)
    nan_replace_metrics = {0: "median", 1: "median", 2: "median"}
    expected_result = pd.DataFrame(
        {"col1": [0, 1, 2, 1], "col2": [0, 1, 2, 3], "col3": [2, 1, 2, 3]}
    ).astype(float)
    result = data_cleaner.replace_missing_data(data.copy(), nan_replace_metrics)
    pd.testing.assert_frame_equal(result, expected_result)


# def test_replace_missing_data_array_mean(data_cleaner):
#     data = np.array([[1, np.nan, 3], [4, 5, np.nan], [7, 8, 9]])
#     nan_replace_metrics = "mean"
#     expected_result = np.array([[1.0, 2, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
#     result = data_cleaner.replace_missing_data(data, nan_replace_metrics)
#     print("Data:\n", data)
#     print("Clean Data:\n", result)
#     np.testing.assert_array_equal(result, expected_result)

# def test_replace_missing_data_array_median(data_cleaner):
#     data = np.array([[1, np.nan, 3], [4, 5, np.nan], [7, 8, 9]])
#     nan_replace_metrics = "median"
#     expected_result = np.array([[1.0, 6.5, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
#     result = data_cleaner.replace_missing_data(data, nan_replace_metrics)
#     np.testing.assert_array_equal(result, expected_result)


# def test_replace_missing_data_array_interpolate(data_cleaner):
#     data = np.array([[1, np.nan, 3], [4, 5, np.nan], [7, 8, 9]])
#     nan_replace_metrics = "interpolate"
#     expected_result = np.array([[1.0, 3.5, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
#     result = data_cleaner.replace_missing_data(data, nan_replace_metrics)
#     np.testing.assert_array_equal(result, expected_result)


# def test_replace_missing_data_dataframe(data_cleaner):
#     data = pd.DataFrame([[1, np.nan, 3], [4, 5, np.nan], [7, 8, 9]])
#     nan_replace_metrics = {1: "median"}
#     expected_result = pd.DataFrame([[1, 8, 3], [4, 5, 6], [7, 8, 9]])
#     result = data_cleaner.replace_missing_data(data, nan_replace_metrics)
#     pd.testing.assert_frame_equal(result, expected_result)


# def test_drop_columns(data_cleaner):
#     data = pd.DataFrame({"A": [1, 2, 3], "B": [4, np.nan, 6], "C": [7, 8, np.nan]})
#     drop_perc = 0.5
#     expected_result = pd.DataFrame({"A": [1, 2, 3], "B": [4, np.nan, 6]})
#     result = data_cleaner.drop_columns(data, drop_perc=drop_perc)
#     pd.testing.assert_frame_equal(result, expected_result)


# def test_drop_rows_array(data_cleaner):
#     data = np.array([[1, np.nan, 3], [4, 5, np.nan], [np.nan, 8, 9]])
#     expected_result = np.array([[4, 5, np.nan]])
#     result = data_cleaner.drop_rows(data)
#     np.testing.assert_array_equal(result, expected_result)


# def test_drop_rows_dataframe(data_cleaner):
#     data = pd.DataFrame([[1, np.nan, 3], [4, 5, np.nan], [np.nan, 8, 9]])
#     expected_result = pd.DataFrame([[4, 5, np.nan]])
#     result = data_cleaner.drop_rows(data)
#     pd.testing.assert_frame_equal(result, expected_result)


# def test_replace_missing_data_wrong_metric(data_cleaner, capsys):
#     data = np.array([[1, np.nan, 3], [4, 5, np.nan], [7, 8, 9]])
#     nan_replace_metrics = "invalid_metric"
#     data_cleaner.replace_missing_data(data, nan_replace_metrics)
#     captured = capsys.readouterr()
#     assert "Wrong nan_replace_metrics type!" in captured.out


# def test_replace_missing_data_wrong_metric_dict(data_cleaner, capsys):
#     data = pd.DataFrame([[1, np.nan, 3], [4, 5, np.nan], [7, 8, 9]])
#     nan_replace_metrics = {1: "invalid_metric"}
#     data_cleaner.replace_missing_data(data, nan_replace_metrics)
#     captured = capsys.readouterr()
#     assert "Wrong nan_replace_metrics type!" in captured.out


# def test_replace_missing_data_wrong_data_type(data_cleaner, capsys):
#     data = "not_a_dataframe_or_array"
#     nan_replace_metrics = "mean"
#     data_cleaner.replace_missing_data(data, nan_replace_metrics)
#     captured = capsys.readouterr()
#     assert "Wrong nan_replace_metrics type!" in captured.out


# def test_drop_columns_no_columns_to_drop(data_cleaner, capsys):
#     data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
#     result = data_cleaner.drop_columns(data)
#     assert result.equals(data)
#     captured = capsys.readouterr()
#     assert "No columns to drop!" in captured.out


if __name__ == "__main__":
    test_replace_missing_data_dataframe_mean(DataCleaner())
    test_replace_missing_data_dataframe_median(DataCleaner())

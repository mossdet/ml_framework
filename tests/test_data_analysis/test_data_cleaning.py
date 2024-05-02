import pytest
import pandas as pd
import numpy as np

# Import the class to be tested
from ml_framework.data_analysis.data_cleaning import DataCleaner


@pytest.fixture
def data_cleaner():
    return DataCleaner()


def test_replace_missing_data_dataframe(data_cleaner):
    # Mean replacement metric
    data = pd.DataFrame([[1.0, np.nan, 3], [4, 5, np.nan], [7, 8, 9]])
    nan_replace_metrics = {0: "mean", 1: "mean", 2: "mean"}
    expected_result = pd.DataFrame([[1.0, 6.5, 3.0], [4, 5, 6], [7, 8, 9]])
    result = data_cleaner.replace_missing_data(data.copy(), nan_replace_metrics)
    pd.testing.assert_frame_equal(result, expected_result)

    # Median replacement metric
    data = {"col1": [0, 1, 2, np.nan], "col2": [0, 1, 2, 3], "col3": [np.nan, 1, 2, 3]}
    data = pd.DataFrame(data).astype(float)
    nan_replace_metrics = {0: "median", 1: "median", 2: "median"}
    expected_result = pd.DataFrame(
        {"col1": [0, 1, 2, 1], "col2": [0, 1, 2, 3], "col3": [2, 1, 2, 3]}
    ).astype(float)
    result = data_cleaner.replace_missing_data(data.copy(), nan_replace_metrics)
    pd.testing.assert_frame_equal(result, expected_result)

    # Interpolate replacement metric
    data = {
        "col1": [0, 1, 2, np.nan, np.nan, np.nan],
        "col2": [0, 1, np.nan, np.nan, np.nan, 5],
        "col3": [np.nan, np.nan, np.nan, 3, 4, 5],
    }
    data = pd.DataFrame(data).astype(float)
    nan_replace_metrics = {0: "interpolate", 1: "interpolate", 2: "interpolate"}
    expected_result = pd.DataFrame(
        {
            "col1": [0, 1, 2, 2, 2, 2],
            "col2": [0, 1, 2, 3, 4, 5],
            "col3": [3, 3, 3, 3, 4, 5],
        }
    ).astype(float)
    result = data_cleaner.replace_missing_data(data.copy(), nan_replace_metrics)
    pd.testing.assert_frame_equal(result, expected_result)

    # Mean replacement metric
    data = pd.DataFrame([[1.0, np.nan, 3], [4, 5, np.nan], [7, 8, 9]])
    nan_replace_metrics = {0: "mean", 1: "mean", 2: "mean"}
    expected_result = pd.DataFrame([[1.0, 6.5, 3.0], [4, 5, 6], [7, 8, 9]])
    result = data_cleaner.replace_missing_data(data.copy(), nan_replace_metrics)
    pd.testing.assert_frame_equal(result, expected_result)


def test_replace_missing_data_ndarray(data_cleaner):

    data = np.array([np.nan, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    nan_replace_metrics = ["mean", "median", "interpolate"]

    for nan_replace_metric in nan_replace_metrics:

        if nan_replace_metric == "mean":
            expected_result = np.array([5, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        elif nan_replace_metric == "median":
            expected_result = np.array([5, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        elif nan_replace_metric == "interpolate":
            expected_result = np.array([1, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        result = data_cleaner.replace_missing_data(data.copy(), nan_replace_metric)
        np.testing.assert_equal(result, expected_result)

    # Wrong array shape
    invalid_data = np.array(
        [[np.nan, 1, 2, 3, 4, 5, 6, 7, 8, 9], [np.nan, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
    )
    result = data_cleaner.replace_missing_data(invalid_data, "mean")
    assert result is None

    # Wrong replacement metric type
    data = np.array([np.nan, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    result = data_cleaner.replace_missing_data(data, nan_replace_metrics=1)
    assert result is None

    # Wrong replacement metric name
    result = data_cleaner.replace_missing_data(data, "means")
    assert result is None


def test_drop_columns(data_cleaner):

    # Nothing to drop
    data = pd.DataFrame({"A": [1, 2, 3], "B": [4, np.nan, 6], "C": [7, np.nan, np.nan]})
    result = data_cleaner.drop_columns(data)
    pd.testing.assert_frame_equal(result, data)

    data = pd.DataFrame({"A": [1, 2, 3], "B": [4, np.nan, 6], "C": [7, np.nan, np.nan]})
    drop_perc = 0.5
    expected_result = pd.DataFrame({"A": [1, 2, 3], "B": [4, np.nan, 6]})
    result = data_cleaner.drop_columns(data, drop_perc=drop_perc)
    pd.testing.assert_frame_equal(result, expected_result)

    data = pd.DataFrame({"A": [1, 2, 3], "B": [4, np.nan, 6], "C": [7, np.nan, np.nan]})
    drop_perc = 0.5
    expected_result = pd.DataFrame({"A": [1, 2, 3]})
    result = data_cleaner.drop_columns(data, drop_idxs=1, drop_perc=drop_perc)
    pd.testing.assert_frame_equal(result, expected_result)


def test_drop_rows(data_cleaner):

    # No rows to drop
    data = pd.DataFrame({"A": [1, 2, 3], "B": [1, 2, 3], "C": [1, 2, 3]})
    result = data_cleaner.drop_rows(data)
    pd.testing.assert_frame_equal(result, data)

    expected_result = pd.DataFrame({"A": [1, 3], "B": [1, 3], "C": [1, 3]})
    result = data_cleaner.drop_rows(data, drop_idxs=1)
    pd.testing.assert_frame_equal(result, expected_result)


if __name__ == "__main__":
    pass
    # pytest.main()

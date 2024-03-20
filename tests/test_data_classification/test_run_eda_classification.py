import pytest
import pandas as pd
from ml_framework.data_classification.run_eda_classification import ClassificationEDA


@pytest.fixture
def classification_eda():
    # Create a ClassificationEDA object with sample data file path
    data_filepath = "sample_data.csv"
    return ClassificationEDA(data_filepath)


def test_read_data(classification_eda):
    # Test the read_data method
    classification_eda.read_data()
    # Check if the data attribute is not None after reading data
    assert classification_eda.data is not None


def test_clean_data(classification_eda):
    # Test the clean_data method
    classification_eda.clean_data()
    # Check if the data attribute is not None after cleaning data
    assert classification_eda.data is not None


def test_encode_data(classification_eda):
    # Test the encode_data method
    classification_eda.encode_data()
    # Check if the data attribute is not None after encoding data
    assert classification_eda.data is not None


def test_visualize_data(classification_eda):
    # Test the visualize_data method
    classification_eda.visualize_data()
    # No assertions here, as we just want to ensure no exceptions are raised


def test_sample_data(classification_eda):
    # Test the sample_data method
    train_data, valid_data, test_data = classification_eda.sample_data()
    # Check if the train_data, valid_data, and test_data attributes are not None
    assert train_data is not None
    assert valid_data is not None
    assert test_data is not None


def test_get_data(classification_eda):
    # Test the get_data method
    data = classification_eda.get_data()
    # Check if the returned data is a DataFrame
    assert isinstance(data, pd.DataFrame)


def test_get_partitioned_data(classification_eda):
    # Test the get_partitioned_data method
    train_data, test_data = classification_eda.get_partitioned_data()
    # Check if the returned train_data and test_data are DataFrames
    assert isinstance(train_data, pd.DataFrame)
    assert isinstance(test_data, pd.DataFrame)


if __name__ == "__main__":
    pytest.main()

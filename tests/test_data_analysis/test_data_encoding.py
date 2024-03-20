import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import pytest

from collections import defaultdict
from typing import List, Dict, Union
from sklearn.preprocessing import StandardScaler

# Import the class to be tested
from ml_framework.data_analysis.data_encoding import DataEncoder


@pytest.fixture
def data_encoder():
    # Initialize DataEncoder with a temporary images destination path
    encoder = DataEncoder(images_destination_path="./test_images/")
    yield encoder


def test_init(data_encoder):
    # Test initialization of DataEncoder class
    assert data_encoder.images_destination_path == "./test_images/"


def test_encode_ordinal_data(data_encoder):
    # Test encode_ordinal_data method

    # Create sample DataFrame with ordinal data
    data = pd.DataFrame({"Ordinal": ["Low", "High", "Medium", "Low"]})

    # Define the categorical order
    categorical_order = {"Low": 0, "Medium": 1, "High": 2}

    # Encode ordinal data
    encoded_data = data_encoder.encode_ordinal_data(
        data=data, col_name="Ordinal", categorical_order=categorical_order
    )

    # Check if the data has been encoded correctly
    assert encoded_data["Ordinal"].tolist() == [0, 2, 1, 0]


def test_encode_nominal_data(data_encoder):
    # Test encode_nominal_data method

    # Create sample DataFrame with nominal data
    data = pd.DataFrame({"Nominal": ["A", "B", "A", "C"]})

    # Encode nominal data
    encoded_data = data_encoder.encode_nominal_data(data=data)

    # Check if the data has been encoded correctly
    assert encoded_data.equals(pd.get_dummies(data))


def test_encode_target_column(data_encoder):
    # Test encode_target_column method

    # Create sample DataFrame with target column
    data = pd.DataFrame({"Target": ["Yes", "No", "Yes", "No"]})

    # Encode target column
    encoded_data = data_encoder.encode_target_column(
        data=data, target_col_name="Target"
    )

    # Check if the data has been encoded correctly
    assert encoded_data["Target"].tolist() == [0, 1, 0, 1]


def test_z_score_data(data_encoder):
    # Test z_score_data method

    # Create sample DataFrame with numerical data
    data = pd.DataFrame({"Feature1": [10, 20, 30], "Feature2": [1, 2, 3]})

    # Z-score normalize data
    normalized_data = data_encoder.z_score_data(data=data, target_col_name="Feature1")

    # Calculate mean and standard deviation for Feature2
    mean = np.mean(normalized_data["Feature2"])
    std = np.std(normalized_data["Feature2"])

    # Check if the mean is approximately 0 and standard deviation is approximately 1
    assert np.isclose(mean, 0, atol=1e-10)
    assert np.isclose(std, 1, atol=1e-10)


if __name__ == "__main__":
    pytest.main()

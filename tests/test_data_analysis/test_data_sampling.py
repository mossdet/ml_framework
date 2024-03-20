import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

# Import the class to be tested
from ml_framework.data_analysis.data_sampling import DataSampler


@pytest.fixture
def data_sampler():
    # Initialize DataSampler
    return DataSampler()


def test_shuffle(data_sampler):
    # Test shuffle method

    # Create sample DataFrame
    data = pd.DataFrame({"A": [1, 2, 3, 4], "B": [5, 6, 7, 8], "C": [9, 10, 11, 12]})

    # Shuffle DataFrame
    shuffled_data = data_sampler.shuffle(data)

    # Check if the number of rows and columns remain the same
    assert shuffled_data.shape == data.shape

    # Check if rows are shuffled
    assert not shuffled_data.equals(data)


def test_sample(data_sampler):
    # Test sample method

    # Create sample DataFrame
    data = pd.DataFrame({"A": range(100), "B": range(100, 200)})

    # Sample 20% of the DataFrame
    sampled_data = data_sampler.sample(data, 0.2)

    # Check if the number of rows in the sampled DataFrame is approximately 20% of the original
    assert abs(sampled_data.shape[0] - 20) <= 2


def test_data_partition(data_sampler):
    # Test data_partition method

    # Create sample DataFrame
    data = pd.DataFrame({"A": range(100), "B": range(100, 200)})

    # Partition data into train and test sets
    train_data, test_data = data_sampler.data_partition(data, 0.8)

    # Check if the number of rows in the train and test sets add up to the original DataFrame
    assert train_data.shape[0] + test_data.shape[0] == data.shape[0]

    # Check if the train and test sets have non-zero sizes
    assert train_data.shape[0] > 0
    assert test_data.shape[0] > 0


def test_stratified_data_partition(data_sampler):
    # Test stratified_data_partition method

    # Create sample DataFrame with a target column
    data = pd.DataFrame(
        {"A": range(100), "B": range(100, 200), "Target": [0] * 50 + [1] * 50}
    )

    # Partition data into train and test sets while maintaining class balance
    train_data, test_data = data_sampler.stratified_data_partition(data, "Target", 0.8)

    # Check if the number of rows in the train and test sets add up to the original DataFrame
    assert train_data.shape[0] + test_data.shape[0] == data.shape[0]

    # Check if the train and test sets have non-zero sizes
    assert train_data.shape[0] > 0
    assert test_data.shape[0] > 0

    # Check if the train and test sets maintain class balance
    assert train_data["Target"].value_counts().min() > 0
    assert test_data["Target"].value_counts().min() > 0


def test_oversample_data(data_sampler):
    # Test oversample_data method

    # Create sample DataFrame with imbalanced classes
    data = pd.DataFrame(
        {"A": range(100), "B": range(100, 200), "Target": [0] * 90 + [1] * 10}
    )

    # Oversample minority class
    oversampled_data = data_sampler.oversample_data(data, "Target", 2)

    # # Check if the oversampled DataFrame has balanced classes
    # assert (
    #     oversampled_data["Target"].value_counts().min()
    #     == oversampled_data["Target"].value_counts().max()
    # )


def test_synthetic_sampling_SMOTE(data_sampler):
    # Test synthetic_sampling_SMOTE method

    # Create sample DataFrame with imbalanced classes
    data = pd.DataFrame(
        {"A": range(100), "B": range(100, 200), "Target": [0] * 90 + [1] * 10}
    )

    # Perform SMOTE
    smote_data = data_sampler.synthetic_sampling_SMOTE(data, "Target")

    # Check if the SMOTE DataFrame has balanced classes
    assert (
        smote_data["Target"].value_counts().min()
        == smote_data["Target"].value_counts().max()
    )


def test_synthetic_sampling_ADASYN(data_sampler):
    # Test synthetic_sampling_ADASYN method

    # Create sample DataFrame with imbalanced classes
    data = pd.DataFrame(
        {"A": range(100), "B": range(100, 200), "Target": [0] * 90 + [1] * 10}
    )

    # Perform ADASYN
    adasyn_data = data_sampler.synthetic_sampling_ADASYN(data, "Target")

    # Check if the ADASYN DataFrame has balanced classes
    assert (
        adasyn_data["Target"].value_counts().min()
        == adasyn_data["Target"].value_counts().max()
    )


if __name__ == "__main__":
    pytest.main()

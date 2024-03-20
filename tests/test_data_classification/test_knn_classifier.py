import os
import pandas as pd
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ml_framework.data_classification.knn_classifier import KNN_Classifier

# Set up logging
import logging

logging.basicConfig(level=logging.INFO)


@pytest.fixture
def knn_classifier():
    # Generate random classification data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_classes=2, random_state=42
    )

    # Split data into train and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)

    # Create KNN_Classifier object
    classifier = KNN_Classifier(
        target_col_name="target",
        train_data=pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(20)]),
        valid_data=pd.DataFrame(X_valid, columns=[f"feature_{i}" for i in range(20)]),
    )
    return classifier


def test_fit(knn_classifier):
    # Test the fit method of KNN_Classifier

    # Check if the fit method does not raise any exceptions
    try:
        knn_classifier.fit(nr_iterations=10)
    except Exception as e:
        pytest.fail(f"Exception raised: {e}")


def test_model_exists_after_fit(knn_classifier):
    # Test if the model exists after calling fit method

    # Call the fit method
    knn_classifier.fit(nr_iterations=10)

    # Check if the model attribute is not None
    assert knn_classifier.model is not None


def test_get_best_k(knn_classifier):
    # Test the get_best_k method of KNN_Classifier

    # Call the fit method before get_best_k
    knn_classifier.fit(nr_iterations=10)

    # Check if the get_best_k method does not raise any exceptions
    try:
        best_k = knn_classifier.get_best_k()
    except Exception as e:
        pytest.fail(f"Exception raised: {e}")

    # Check if the returned best_k is an integer
    assert isinstance(best_k, int)


if __name__ == "__main__":
    pytest.main()

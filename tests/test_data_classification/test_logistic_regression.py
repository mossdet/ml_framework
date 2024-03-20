import os
import pandas as pd
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ml_framework.data_classification.logistic_regression import (
    LogisticRegressionClassifier,
)

# Set up logging
import logging

logging.basicConfig(level=logging.INFO)


@pytest.fixture
def logistic_regression_classifier():
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

    # Create LogisticRegressionClassifier object
    classifier = LogisticRegressionClassifier(
        target_col_name="target",
        train_data=pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(20)]),
        valid_data=pd.DataFrame(X_valid, columns=[f"feature_{i}" for i in range(20)]),
    )
    return classifier


def test_fit(logistic_regression_classifier):
    # Test the fit method of LogisticRegressionClassifier

    # Check if the fit method does not raise any exceptions
    try:
        logistic_regression_classifier.fit(nr_iterations=10)
    except Exception as e:
        pytest.fail(f"Exception raised: {e}")


def test_model_exists_after_fit(logistic_regression_classifier):
    # Test if the model exists after calling fit method

    # Call the fit method
    logistic_regression_classifier.fit(nr_iterations=10)

    # Check if the model attribute is not None
    assert logistic_regression_classifier.model is not None


if __name__ == "__main__":
    pytest.main()

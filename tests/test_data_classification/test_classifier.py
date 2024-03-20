import os
import pandas as pd
import numpy as np
import pytest
import matplotlib.pyplot as plt
import logging
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ml_framework.data_classification.classifier import Classifier


# Set up logging
logging.basicConfig(level=logging.INFO)


@pytest.fixture
def classifier():
    # Generate random classification data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_classes=2, random_state=42
    )

    # Split data into train, validation, and test sets
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_valid, y_valid, test_size=0.5, random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    # Create Classifier object
    classifier = Classifier(
        target_col_name="target",
        train_data=pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(20)]),
        valid_data=pd.DataFrame(X_valid, columns=[f"feature_{i}" for i in range(20)]),
    )

    return classifier


def test_fit(classifier):
    # Test the fit method of Classifier

    # Check if the fit method does not raise any exceptions
    try:
        classifier.fit()
    except Exception as e:
        pytest.fail(f"Exception raised: {e}")


def test_predict(classifier):
    # Test the predict method of Classifier

    # Call the fit method before predict
    classifier.fit()

    # Create random test data
    X_test = np.random.randn(100, 20)

    # Check if the predict method does not raise any exceptions
    try:
        classifier.predict(test_data=pd.DataFrame(X_test))
    except Exception as e:
        pytest.fail(f"Exception raised: {e}")


def test_score(classifier):
    # Test the score method of Classifier

    # Call the fit method before score
    classifier.fit()

    # Call the predict method before score
    classifier.predict(test_data=pd.DataFrame(np.random.randn(100, 20)))

    # Check if the score method does not raise any exceptions
    try:
        score_dict = classifier.score()
    except Exception as e:
        pytest.fail(f"Exception raised: {e}")

    # Check if the returned score_dict contains the necessary keys
    assert "Precision" in score_dict
    assert "Recall" in score_dict
    assert "Accuracy" in score_dict
    assert "F1-Score" in score_dict


if __name__ == "__main__":
    pytest.main()

import os
import pandas as pd
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ml_framework.data_classification.decision_tree_classifier import (
    DecisionTreeClassifier,
)

# Set up logging
import logging

logging.basicConfig(level=logging.INFO)


@pytest.fixture
def decision_tree_classifier():
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

    # Create DecisionTreeClassifier object
    classifier = DecisionTreeClassifier(
        target_col_name="target",
        train_data=pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(20)]),
        valid_data=pd.DataFrame(X_valid, columns=[f"feature_{i}" for i in range(20)]),
    )
    return classifier


def test_fit(decision_tree_classifier):
    # Test the fit method of DecisionTreeClassifier

    # Check if the fit method does not raise any exceptions
    try:
        decision_tree_classifier.fit(nr_iterations=10)
    except Exception as e:
        pytest.fail(f"Exception raised: {e}")


def test_model_exists_after_fit(decision_tree_classifier):
    # Test if the model exists after calling fit method

    # Call the fit method
    decision_tree_classifier.fit(nr_iterations=10)

    # Check if the model attribute is not None
    assert decision_tree_classifier.model is not None


def test_predict(decision_tree_classifier):
    # Test the predict method of DecisionTreeClassifier

    # Call the fit method before predict
    decision_tree_classifier.fit(nr_iterations=10)

    # Create random test data
    X_test = np.random.randn(100, 20)

    # Check if the predict method does not raise any exceptions
    try:
        decision_tree_classifier.predict(test_data=pd.DataFrame(X_test))
    except Exception as e:
        pytest.fail(f"Exception raised: {e}")


def test_score(decision_tree_classifier):
    # Test the score method of DecisionTreeClassifier

    # Call the fit method before score
    decision_tree_classifier.fit(nr_iterations=10)

    # Call the predict method before score
    decision_tree_classifier.predict(test_data=pd.DataFrame(np.random.randn(100, 20)))

    # Check if the score method does not raise any exceptions
    try:
        score_dict = decision_tree_classifier.score()
    except Exception as e:
        pytest.fail(f"Exception raised: {e}")

    # Check if the returned score_dict contains the necessary keys
    assert "Precision" in score_dict
    assert "Recall" in score_dict
    assert "Accuracy" in score_dict
    assert "F1-Score" in score_dict


if __name__ == "__main__":
    pytest.main()

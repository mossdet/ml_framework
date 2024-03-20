import os
import pandas as pd
import numpy as np
import pytest
import matplotlib.pyplot as plt
import logging
import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ml_framework.data_classification.classifier import Classifier
from ann_tf_classifier import ANN_TF_Classifier

# Set up logging
logging.basicConfig(level=logging.INFO)


@pytest.fixture
def ann_tf_classifier():
    # Generate random classification data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_classes=2, random_state=42
    )

    # Split data into train and test sets
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)

    # Create ANN_TF_Classifier object
    classifier = ANN_TF_Classifier(
        target_col_name="target",
        train_data=pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(20)]),
        valid_data=pd.DataFrame(X_valid, columns=[f"feature_{i}" for i in range(20)]),
    )

    return classifier


def test_fit(ann_tf_classifier):
    # Test the fit method of ANN_TF_Classifier

    # Check if the fit method does not raise any exceptions
    try:
        ann_tf_classifier.fit(nr_iterations=1)
    except Exception as e:
        pytest.fail(f"Exception raised: {e}")


def test_predict(ann_tf_classifier):
    # Test the predict method of ANN_TF_Classifier

    # Call the fit method before predict
    ann_tf_classifier.fit(nr_iterations=1)

    # Create random test data
    X_test = np.random.randn(100, 20)

    # Check if the predict method does not raise any exceptions
    try:
        ann_tf_classifier.predict(test_data=pd.DataFrame(X_test))
    except Exception as e:
        pytest.fail(f"Exception raised: {e}")


if __name__ == "__main__":
    pytest.main()

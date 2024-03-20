import pytest
import pandas as pd
from xgboost import XGBClassifier
from ml_framework.data_classification.xgboost_classifier import (
    XGBoostClassifier,
)


@pytest.fixture
def xgboost_classifier():
    # Create an XGBoostClassifier object with sample data
    return XGBoostClassifier()


def test_fit(xgboost_classifier):
    # Test the fit method
    # Create sample data for training and validation
    X_train = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    y_train = pd.Series([0, 1, 0])
    X_valid = pd.DataFrame({"feature1": [4, 5, 6], "feature2": [7, 8, 9]})
    y_valid = pd.Series([1, 0, 1])

    # Set the sample data for the classifier
    xgboost_classifier.X_train = X_train
    xgboost_classifier.y_train = y_train
    xgboost_classifier.X_valid = X_valid
    xgboost_classifier.y_valid = y_valid

    # Call the fit method
    xgboost_classifier.fit(nr_iterations=5)

    # Check if the model attribute is not None after fitting
    assert xgboost_classifier.model is not None
    # Check if the model is an instance of XGBClassifier
    assert isinstance(xgboost_classifier.model, XGBClassifier)


if __name__ == "__main__":
    pytest.main()

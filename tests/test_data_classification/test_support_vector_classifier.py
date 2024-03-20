import pytest
import pandas as pd
from ml_framework.data_classification.support_vector_classifier import (
    SupportVectorClassifier,
)


@pytest.fixture
def support_vector_classifier():
    # Create a SupportVectorClassifier object with sample data
    return SupportVectorClassifier()


def test_fit(support_vector_classifier):
    # Test the fit method
    # Create sample data for training and validation
    X_train = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    y_train = pd.Series([0, 1, 0])
    X_valid = pd.DataFrame({"feature1": [4, 5, 6], "feature2": [7, 8, 9]})
    y_valid = pd.Series([1, 0, 1])

    # Set the sample data for the classifier
    support_vector_classifier.X_train = X_train
    support_vector_classifier.y_train = y_train
    support_vector_classifier.X_valid = X_valid
    support_vector_classifier.y_valid = y_valid

    # Call the fit method
    support_vector_classifier.fit(nr_iterations=5)

    # Check if the model attribute is not None after fitting
    assert support_vector_classifier.model is not None


if __name__ == "__main__":
    pytest.main()

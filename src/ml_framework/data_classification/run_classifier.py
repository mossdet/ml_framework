import pandas as pd

from ml_framework.data_classification.logistic_regression import (
    LogisticRegressionClassifier,
)
from ml_framework.data_classification.knn_classifier import KNN_Classifier
from ml_framework.data_classification.decision_tree_classifier import (
    DecisionTreeClassifier,
)
from ml_framework.data_classification.random_forest_classifier import (
    RandomForestClassifier,
)
from ml_framework.data_classification.support_vector_classifier import (
    SupportVectorClassifier,
)
from ml_framework.data_classification.xgboost_classifier import XGBoostClassifier
from ml_framework.data_classification.ann_tf_classifier import ANN_TF_Classifier


class RunClassification:
    """
    A class for running classification tasks using different classifiers.

    Attributes:
        classifier_name (str): The name of the classifier to use.
        target_col_name (str): The name of the target column in the dataset.
        train_data (pd.DataFrame): The training data for the classifier.
        valid_data (pd.DataFrame): The validation data for the classifier.
        model: The classifier model instance.

    Methods:
        fit(nr_iterations): Fits the classifier model to the training data.
        predict(test_data): Predicts the target values for the test data.
        score(): Computes the score of the classifier model.
        plot_confusion_matrix(): Plots the confusion matrix for the classifier's predictions.
    """

    def __init__(
        self,
        classifier_name: str = None,
        target_col_name: str = None,
        train_data: pd.DataFrame = None,
        valid_data: pd.DataFrame = None,
    ):
        """
        Initializes the RunClassification instance.

        Parameters:
            classifier_name (str): The name of the classifier to use.
            target_col_name (str): The name of the target column in the dataset.
            train_data (pd.DataFrame): The training data for the classifier.
            valid_data (pd.DataFrame): The validation data for the classifier.
        """
        self.classifier_name = classifier_name
        self.target_col_name = target_col_name
        self.model = None
        params = {
            "target_col_name": target_col_name,
            "train_data": train_data,
            "valid_data": valid_data,
        }
        self.model = eval(classifier_name + "(target_col_name, train_data, valid_data)")
        pass

    def fit(self, nr_iterations: int = None):
        """
        Fits the classifier model to the training data.

        Parameters:
            nr_iterations (int): The number of iterations for the fitting process.
        """
        self.model.fit(nr_iterations=nr_iterations)

    def predict(self, test_data: pd.DataFrame = None):
        """
        Predicts the target values for the test data.

        Parameters:
            test_data (pd.DataFrame): The test data for prediction.
        """
        self.model.predict(test_data)

    def score(self):
        """
        Computes the score of the classifier model.

        Returns:
            dict: A dictionary containing the score metrics of the classifier.
        """
        score_dict = self.model.score()
        return score_dict

    def plot_confusion_matrix(self):
        """
        Plots the confusion matrix for the classifier's predictions.
        """
        self.model.plot_confusion_matrix()


if __name__ == "__main__":
    pass

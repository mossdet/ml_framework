import pandas as pd

from ml_framework.data_classification.logistic_regression import (
    LogisticRegressionClassifier,
)
from ml_framework.data_classification.knn import KNN_Classifier


class RunClassification:
    def __init__(
        self,
        classifier_name: str = None,
        target_col_name: str = None,
        train_data: pd.DataFrame = None,
        valid_data: pd.DataFrame = None,
    ):
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
        self.model.fit(nr_iterations=nr_iterations)

    def predict(self, test_data: pd.DataFrame = None):
        self.model.predict(test_data)

    def score(self):
        score_dict = self.model.score()
        return score_dict

    def plot_confusion_matrix(self):
        self.model.plot_confusion_matrix()

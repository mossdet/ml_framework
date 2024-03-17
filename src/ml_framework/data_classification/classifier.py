import os
import pandas as pd
import numpy as np
import optuna
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

from typing import List, Dict, Union
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
)
from ml_framework.tools.helper_functions import get_workspace_path


class Classifier:
    """
    A class for performing classification tasks.

    Attributes:
        target_col_name (str): The name of the target column in the dataset.
        train_data (pd.DataFrame): The training data for the classifier.
        valid_data (pd.DataFrame): The validation data for the classifier.
        X_train (np.ndarray): Features of the training data.
        y_train (np.ndarray): Target labels of the training data.
        X_valid (np.ndarray): Features of the validation data.
        y_valid (np.ndarray): Target labels of the validation data.
        y_test (np.ndarray): Predicted target labels.
        model: The trained classifier model.
        y_predicted (np.ndarray): Predicted target labels.
        confusion_matrix (np.ndarray): Confusion matrix of the predictions.
        images_destination_path (str): Path to store generated images.

    Methods:
        fit(): Trains the classifier model.
        predict(test_data): Predicts target labels for the test data.
        get_predicted_values(): Returns the predicted target labels.
        score(): Computes performance metrics of the classifier model.
        plot_confusion_matrix(): Plots the confusion matrix.
    """

    def __init__(
        self,
        target_col_name: str = None,
        train_data: pd.DataFrame = None,
        valid_data: pd.DataFrame = None,
    ):
        """
        Initializes the Classifier instance.

        Args:
            target_col_name (str): The name of the target column in the dataset.
            train_data (pd.DataFrame): The training data for the classifier.
            valid_data (pd.DataFrame): The validation data for the classifier.
        """
        self.X_train = train_data.loc[
            :, train_data.columns != target_col_name
        ].to_numpy()
        self.y_train = (
            train_data.loc[:, train_data.columns == target_col_name].to_numpy().ravel()
        )

        self.X_valid = valid_data.loc[
            :, valid_data.columns != target_col_name
        ].to_numpy()
        self.y_valid = (
            valid_data.loc[:, valid_data.columns == target_col_name].to_numpy().ravel()
        )

        self.target_col_name = target_col_name
        self.y_test = None
        self.model = None
        self.y_predicted = None
        self.confusion_matrix = None

        self.nr_classes = len(np.unique(self.y_train))

        self.images_destination_path = (
            get_workspace_path()
            + "Images/Classification/Classification_Modelling_Images/"
        )
        os.makedirs(self.images_destination_path, exist_ok=True)
        print("\n\n****************************************************************")
        print("\nRunning ", type(self).__name__)

        pass

    def fit(self):
        """Trains the classifier model."""
        pass

    def predict(self, test_data: pd.DataFrame = None):
        """
        Predicts target labels for the test data.

        Args:
            test_data (pd.DataFrame): The test data for prediction.
        """

        X_test = test_data.loc[:, test_data.columns != self.target_col_name].to_numpy()
        self.y_test = (
            test_data.loc[:, test_data.columns == self.target_col_name]
            .to_numpy()
            .ravel()
        )
        self.y_predicted = self.model.predict(X_test)
        self.confusion_matrix = confusion_matrix(self.y_test, self.y_predicted)

    def get_predicted_values(self):
        """Returns the predicted target labels."""
        return self.y_predicted

    def score(self):
        """
        Computes performance metrics of the classifier model.

        Returns:
            dict: A dictionary containing the computed performance metrics.
        """
        f1_val = f1_score(self.y_test, self.y_predicted, average=None)
        f1_val = np.mean(f1_val)

        prec_val = precision_score(self.y_test, self.y_predicted, average=None)
        prec_val = np.mean(prec_val)

        recall_val = recall_score(self.y_test, self.y_predicted, average=None)
        recall_val = np.mean(recall_val)

        acc_val = accuracy_score(self.y_test, self.y_predicted)

        print(
            f"\n{type(self).__name__}\nPerformance Metrics",
        )
        score_dict = {
            "Precision": prec_val,
            "Recall": recall_val,
            "Accuracy": acc_val,
            "F1-Score": f1_val,
        }

        for k, v in score_dict.items():
            print(f"{k}: {v}")

        return score_dict

    def plot_confusion_matrix(self) -> None:
        """Plots the confusion matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            self.confusion_matrix,
            annot=True,
            fmt=".0f",
            cmap="crest",
            cbar_kws={"label": "# of Events"},
        )
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.title(f"{type(self).__name__}\nConfusion Matrix\n")

        suffix = "_" + type(self).__name__

        plt.savefig(self.images_destination_path + f"Confusion_Matrix{suffix}.jpeg")
        # plt.show()
        plt.close()


if __name__ == "__main__":
    pass

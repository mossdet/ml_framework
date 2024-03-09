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

"""
    The Classifier class performs three main functions:
        - Fit: a function that takes the training set and the name of the needed model as input and trains the model.
        - Predict: a function that takes a testing set and predicts the output.
        - Score: a function that takes two sets: one for input features and the other for their true labels. The function 
                 also takes an argument called metric to decide what to return of different classification metrics.
"""


class Classifier:
    def __init__(
        self,
        target_col_name: str = None,
        train_data: pd.DataFrame = None,
        valid_data: pd.DataFrame = None,
    ):
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

        self.images_destination_path = get_workspace_path() + "Images/Modelling_Images/"
        os.makedirs(self.images_destination_path, exist_ok=True)
        print("\n\n****************************************************************")
        print("\nRunning ", type(self).__name__)

        pass

    def fit(self):
        pass

    def predict(self, test_data: pd.DataFrame = None):

        X_test = test_data.loc[:, test_data.columns != self.target_col_name].to_numpy()
        self.y_test = (
            test_data.loc[:, test_data.columns == self.target_col_name]
            .to_numpy()
            .ravel()
        )
        self.y_predicted = self.model.predict(X_test)
        self.confusion_matrix = confusion_matrix(self.y_predicted, self.y_test)

    def get_predicted_values(self):
        return self.y_predicted

    def score(self):
        f1_val = f1_score(self.y_predicted, self.y_test, average=None)
        f1_val = np.mean(f1_val)

        prec_val = precision_score(self.y_predicted, self.y_test, average=None)
        prec_val = np.mean(prec_val)

        recall_val = recall_score(self.y_predicted, self.y_test, average=None)
        recall_val = np.mean(recall_val)

        acc_val = accuracy_score(self.y_predicted, self.y_test)

        print(
            f"\n{type(self).__name__}\nPerformance Metrics",
        )
        score_dict = {
            "Precision:": prec_val,
            "Recall:": recall_val,
            "Accuracy:": acc_val,
            "F1-Score": f1_val,
        }

        for k, v in score_dict.items():
            print(f"{k}: {v}")

        return score_dict

    def plot_confusion_matrix(self) -> None:
        sns.heatmap(self.confusion_matrix, annot=True, fmt=".0f", cmap="crest")
        plt.ylabel("True label")
        plt.ylabel("Predicted label")
        plt.title("Confusion Matrix")

        suffix = "_" + type(self).__name__

        plt.savefig(self.images_destination_path + f"Confusion_Matrix{suffix}.jpeg")
        # plt.show()
        plt.close()


if __name__ == "__main__":
    pass

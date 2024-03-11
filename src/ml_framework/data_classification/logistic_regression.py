import os
import pandas as pd
import numpy as np
import optuna
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from ml_framework.data_classification.classifier import Classifier

from ml_framework.tools.helper_functions import get_workspace_path
from typing import List, Dict, Union
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
)


class LogisticRegressionClassifier(Classifier):
    def __init__(
        self,
        target_col_name: str = None,
        train_data: pd.DataFrame = None,
        valid_data: pd.DataFrame = None,
    ):
        super().__init__(
            target_col_name=target_col_name,
            train_data=train_data,
            valid_data=valid_data,
        )

    def fit(self, nr_iterations: int = 10):

        def optuna_objective_func(trial, X_train, y_train, X_valid, y_valid):
            params = {
                "solver": trial.suggest_categorical(
                    "solver", ["lbfgs", "liblinear", "sag", "saga"]
                ),
                "max_iter": trial.suggest_categorical("max_iter", [500]),
                "n_jobs": trial.suggest_categorical("n_jobs", [-1]),
                "random_state": trial.suggest_categorical("random_state", [42]),
            }

            model = sklearn.linear_model.LogisticRegression(**params).fit(
                X_train, y_train
            )

            y_predicted = model.predict(X_valid)
            f1_val = f1_score(y_valid, y_predicted, average=None)
            f1_val = np.mean(f1_val)

            return f1_val

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")

        # Start optimizing with 100 trials
        study.optimize(
            lambda trial: optuna_objective_func(
                trial, self.X_train, self.y_train, self.X_valid, self.y_valid
            ),
            n_trials=nr_iterations,
        )

        # Retrain on training+validation set
        X_train_valid = np.concatenate((self.X_train, self.X_valid))
        y_train_valid = np.concatenate((self.y_train, self.y_valid))
        best_trial = study.best_trial
        self.model = sklearn.linear_model.LogisticRegression(**best_trial.params).fit(
            X_train_valid, y_train_valid
        )

        pass


if __name__ == "__main__":
    pass

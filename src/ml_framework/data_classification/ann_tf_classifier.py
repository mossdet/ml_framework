import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import logging

from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Dropout, BatchNormalization
from keras import regularizers
from ml_framework.data_classification.classifier import Classifier
from ml_framework.tools.helper_functions import get_workspace_path
from typing import List, Dict, Union
from sklearn.metrics import confusion_matrix


class ANN_TF_Classifier(Classifier):
    """
    ANN_TF_Classifier class for fitting an ANN classifier model as implemented in tensor-flow.

    Attributes:
        target_col_name (str): The name of the target column.
        train_data (pd.DataFrame): The training data.
        valid_data (pd.DataFrame): The validation data.
        model: The xgboost classifier model.

    Methods:
        fit(nr_iterations: int = 10): Fit the ANN classifier model.
    """

    def __init__(
        self,
        target_col_name: str = None,
        train_data: pd.DataFrame = None,
        valid_data: pd.DataFrame = None,
    ):
        """
        Initialize the ANN_TF_Classifier object.

        Args:
            target_col_name (str): The name of the target column.
            train_data (pd.DataFrame): The training data.
            valid_data (pd.DataFrame): The validation data.
        """
        super().__init__(
            target_col_name=target_col_name,
            train_data=train_data,
            valid_data=valid_data,
        )

    def fit(self, nr_iterations: int = 10):
        """
        Fit the ANN classifier model.

        Args:
            nr_iterations (int): The number of iterations or epochs to train the ANN parameters.
        """

        plt.switch_backend("agg")

        # Define the ANN architecture

        nr_layers = 2
        nr_hidden_nodes = self.X_train.shape[1] * 10  # 32
        reg_val = 1e-6
        batch_size = 32
        nr_epochs = 100  # nr_iterations

        self.model = Sequential()

        # Hidden layers
        for _ in range(nr_layers):
            self.model.add(
                Dense(
                    units=nr_hidden_nodes,
                    activation="relu",
                    kernel_regularizer=regularizers.L2(reg_val),
                    bias_regularizer=regularizers.L2(reg_val),
                    activity_regularizer=regularizers.L2(reg_val),
                )
            )

        # Output layer
        self.model.add(Dense(units=self.nr_classes, activation="softmax"))

        # Compile model
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        training_result = self.model.fit(
            self.X_train,
            self.y_train,
            batch_size=batch_size,
            epochs=nr_epochs,
            validation_data=(self.X_valid, self.y_valid),
            callbacks=[],
            verbose=1,
        )

        self.plot_training_results(training_result)

        logging.info(self.model.summary())

        pass

    def plot_training_results(
        self, training_result: keras.src.callbacks.History = None
    ):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 9))
        axes[0].plot(training_result.history["loss"], label="Loss")
        axes[0].plot(training_result.history["val_loss"], label="Validation loss")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        # Accuracy
        axes[1].plot(training_result.history["accuracy"], label="Accuracy")
        axes[1].plot(
            training_result.history["val_accuracy"], label="Validation accuracy"
        )
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        plt.tight_layout()
        plt.show()
        plt.savefig(
            self.images_destination_path + f"ANN_TF_Classifier_TrainingResult.jpeg"
        )
        # plt.show()
        plt.close()

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

        self.y_predicted = self.model.predict(X_test, verbose=1)
        self.y_predicted = np.argmax(self.y_predicted, axis=1)

        self.confusion_matrix = confusion_matrix(self.y_predicted, self.y_test)

    def save_model(self, stored_model_path:str=None)->None:
        # store the Keras model
        self.model.save(stored_model_path + type(self).__name__ + "SavedModel.keras")

    def load_model(self, stored_model_path:str=None)->None:
        # load the Keras model
        self.model = keras.models.load_model(stored_model_path + type(self).__name__ + "SavedModel.keras")

if __name__ == "__main__":
    pass

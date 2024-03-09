from typing import List, Dict, Union

"""
    The Classifier class performs three main functions:
        - Fit: a function that takes the training set and the name of the needed model as input and trains the model.
        - Predict: a function that takes a testing set and predicts the output.
        - Score: a function that takes two sets: one for input features and the other for their true labels. The function 
                 also takes an argument called metric to decide what to return of different classification metrics.
"""


class Classifier:
    def __init__(self, estimator_name):
        pass

    def fit(self, df: pd.DataFrame):
        pass

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        self.predict(df)

    def score(
        self, predicted_label: np.array, actual_label: np.array, list_metrics: List[str]
    ) -> Dict[str, Union[float, np.ndarray]]:
        metrics_dict = dict
        for metric in list_metrics:
            score_val = actual_label - predicted_label
            metrics_dict[metric] = score_val


if __name__ == "__main__":
    pass

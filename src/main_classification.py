import os
import socket
import matplotlib
import pandas as pd
import logging

from collections import defaultdict
from ml_framework.data_classification.run_eda_classification import ClassificationEDA

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
from ml_framework.tools.helper_functions import get_workspace_path


this_filename = os.path.split(os.path.abspath(__file__))[1]
logger = logging.getLogger(this_filename)
logging.basicConfig(
    filename=this_filename.replace(".py", ".log"),
    encoding="utf-8",
    level=logging.DEBUG,
    format="%(parent)s - %(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

matplotlib.use("Agg")
logger.debug(
    'matplotlib.use("Agg") must be called to avoid crashing because of image plotting'
)


logger.info("info level log message test")
logger.warning("warning level log message test")
logger.error("error level log message test")


# Set Data path
data_folder_path = ""
if socket.gethostname() == "LAPTOP-TFQFNF6U":
    data_folder_path = "F:/Weiterbildung/UOC_ML_Bootcamp/Capstone_Projects/Data/"
elif socket.gethostname() == "DLP":
    data_folder_path = "C:/Users/HFO/Documents/MachineLearning/Capstone_Projects/Data/"

data_filepath = data_folder_path + "diamonds.csv"

# Analyzer
target_col_name = "cut"
analyzer = ClassificationEDA(data_filepath)
analyzer.read_data()
analyzer.clean_data()
analyzer.encode_data(target_col_name=target_col_name)
# analyzer.visualize_data(target_col_name=target_col_name)
train_data, valid_data, test_data = analyzer.sample_data(
    target_col_name=target_col_name, train_perc=0.8, valid_perc=0.2
)


# Classifier
classifiers_ls = [
    "LogisticRegressionClassifier",
    "KNN_Classifier",
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "XGBoostClassifier",
    "ANN_TF_Classifier",
    "SupportVectorClassifier",
]

all_models_performance = defaultdict(list)
for classifier_name in classifiers_ls:
    classifier = eval(classifier_name + "(target_col_name, train_data, valid_data)")
    classifier.fit(nr_iterations=100)
    classifier.predict(test_data)
    score_dict = classifier.score()
    classifier.plot_confusion_matrix()

    all_models_performance["Model"].append(classifier_name)
    for k, v in score_dict.items():
        all_models_performance[k].append(v)

    pass

# DataVisualizer().plot_performance_radar_chart(performance_dict=all_models_performance)

perf_df = pd.DataFrame(all_models_performance)
print(perf_df)

tables_destination_path = get_workspace_path() + "Tables/"
os.makedirs(tables_destination_path, exist_ok=True)
perf_df.to_excel(tables_destination_path + "Classification_Results.xlsx")
pass

if __name__ == "__main__":
    pass

import socket
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot
from collections import defaultdict
from ml_framework.data_classification.run_eda_classification import ClassificationEDA
from ml_framework.data_classification.run_classifier import RunClassification
from ml_framework.data_analysis.data_visualization import DataVisualizer

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
# classifiers_ls = ["SupportVectorClassifier"]

params = {
    "classifier_name": "",
    "target_col_name": target_col_name,
    "train_data": train_data,
    "valid_data": valid_data,
}
all_models_performance = defaultdict(list)
for classifier_name in classifiers_ls:
    params["classifier_name"] = classifier_name
    classifier = RunClassification(**params)
    classifier.fit(nr_iterations=50)
    classifier.predict(test_data)
    score_dict = classifier.score()
    classifier.plot_confusion_matrix()

    all_models_performance["Model"].append(classifier_name)
    for k, v in score_dict.items():
        all_models_performance[k].append(v)

    pass

DataVisualizer().plot_performance_radar_chart(performance_dict=all_models_performance)

pass

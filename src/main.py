import socket
import run_analyzer
from ml_framework.data_classification.logistic_regression import (
    LogisticRegressionClassifier,
)
from ml_framework.data_classification.knn import KNN_Classifier

# Set Data path
data_folder_path = ""
if socket.gethostname() == "LAPTOP-TFQFNF6U":
    data_folder_path = "F:/Weiterbildung/UOC_ML_Bootcamp/Capstone_Projects/Data/"
elif socket.gethostname() == "DLP":
    data_folder_path = "C:/Users/HFO/Documents/MachineLearning/Capstone_Projects/Data/"

data_filepath = data_folder_path + "diamonds.csv"


# Analyzer
target_col_name = "cut"
analyzer = run_analyzer.Analyzer(data_filepath)
analyzer.read_data()
analyzer.clean_data()
analyzer.encode_data(target_col_name=target_col_name)
# analyzer.visualize_data(target_col_name=target_col_name)
train_data, valid_data, test_data = analyzer.sample_data(
    target_col_name=target_col_name, train_perc=0.8, valid_perc=0.2
)

# Classifier

classifier = LogisticRegressionClassifier(target_col_name, train_data, valid_data)
classifier.fit(nr_iterations=50)
classifier.predict(test_data, target_col_name=target_col_name)
classifier.score()
classifier.plot_confusion_matrixx()


classifier = KNN_Classifier(target_col_name, train_data, valid_data)
classifier.fit(nr_iterations=50)
print("Best K= ", classifier.get_best_k())
classifier.predict(test_data, target_col_name=target_col_name)
classifier.score()
classifier.plot_confusion_matrixx()

pass

import socket
import run_analyzer
import run_classifier

# Set Data path
data_folder_path = ""
if socket.gethostname() == "LAPTOP-TFQFNF6U":
    data_folder_path = "F:/Weiterbildung/UOC_ML_Bootcamp/Capstone_Projects/Data/"
elif socket.gethostname() == "DLP":
    data_folder_path = "C:/Users/HFO/Documents/MachineLearning/Capstone_Projects/Data/"

data_filepath = data_folder_path + "diamonds.csv"


# Analyzer
target_col_name = "cut"
analyzer = run_analyzer.RunAnalysis(data_filepath)
analyzer.read_data()
analyzer.clean_data()
analyzer.encode_data(target_col_name=target_col_name)
# analyzer.visualize_data(target_col_name=target_col_name)
train_data, valid_data, test_data = analyzer.sample_data(
    target_col_name=target_col_name, train_perc=0.8, valid_perc=0.2
)

# Classifier

classifiers_ls = ["LogisticRegressionClassifier", "KNN_Classifier"]
params = {
    "classifier_name": "",
    "target_col_name": target_col_name,
    "train_data": train_data,
    "valid_data": valid_data,
}
for classifier_name in classifiers_ls:
    params["classifier_name"] = classifier_name
    classifier = run_classifier.RunClassification(**params)
    classifier.fit(nr_iterations=50)
    classifier.predict(test_data)
    classifier.score()
    classifier.plot_confusion_matrix()


pass

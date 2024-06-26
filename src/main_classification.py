import os
import socket
import matplotlib
import pandas as pd
import logging
import pickle

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

matplotlib.use("Agg")


def main() -> None:

    # Set Data path
    data_folder_path = ""
    if socket.gethostname() == "LAPTOP-TFQFNF6U":
        if os.name == 'nt':
            data_folder_path = "F:/Weiterbildung/UOC_ML_Bootcamp/Capstone_Projects/Data/"
            stored_models_path = "F:/Weiterbildung/UOC_ML_Bootcamp/Capstone_Projects/ml_framework/Stored_Models/"
        elif os.name == 'posix':
            data_folder_path = "/mnt/f/Weiterbildung/UOC_ML_Bootcamp/Capstone_Projects/Data/"
            stored_models_path = "/mnt/f/Weiterbildung/UOC_ML_Bootcamp/Capstone_Projects/ml_framework/Stored_Models/"
    elif socket.gethostname() == "DLP":
        if os.name == 'nt':
            data_folder_path = "C:/Users/HFO/Documents/MachineLearning/Capstone_Projects/Data/"
            stored_models_path = "C:/Users/HFO/Documents/MachineLearning/Capstone_Projects/ml_framework/Stored_Models/"
        elif os.name == 'posix':
            data_folder_path = "mnt/c/Users/HFO/Documents/MachineLearning/Capstone_Projects/Data/"
            stored_models_path = "mnt/c/Users/HFO/Documents/MachineLearning/Capstone_Projects/ml_framework/Stored_Models/"
        
    os.makedirs(stored_models_path, exist_ok=True)

    data_filepath = data_folder_path + "diamonds.csv"

    # Analyzer
    target_col_name = "cut"
    analyzer = ClassificationEDA(data_filepath)
    analyzer.read_data()
    analyzer.clean_data()
    analyzer.encode_data(target_col_name=target_col_name)
    analyzer.visualize_data(target_col_name=target_col_name)
    train_data, valid_data, test_data = analyzer.sample_data(
        target_col_name=target_col_name, train_perc=0.8, valid_perc=0.2
    )
    # store the analyzer as pickle file
    analyzer_filepath = stored_models_path+"Classification_Analyzer.bin"
    with open(analyzer_filepath, 'wb') as f_out:
        pickle.dump(analyzer, f_out)
                
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

    # load the classifiers
    # for classifier_name in classifiers_ls:
    #     classifier_filepath = stored_models_path + f"{classifier_name}.bin"
    #     with open(classifier_filepath, 'rb') as f_in:
    #         classifier = pickle.load(f_in)

    #     classifier.load_model(stored_model_path=stored_models_path)
    #     classifier.predict(test_data)
    #     score_dict = classifier.score()
    #     pass

    all_models_performance = defaultdict(list)
    for classifier_name in classifiers_ls:
        classifier = eval(classifier_name + "(target_col_name, train_data, valid_data)")
        classifier.fit(nr_iterations=100)
        classifier.predict(test_data)
        score_dict = classifier.score()
        classifier.plot_confusion_matrix()
        classifier.save_model(stored_model_path=stored_models_path)

        all_models_performance["Model"].append(classifier_name)
        for k, v in score_dict.items():
            all_models_performance[k].append(v)

        # # store the classifier as pickle file
        # classifier_filepath = stored_models_path + f"{classifier_name}.bin"
        # with open(classifier_filepath, 'wb') as f_out:
        #     regressor.reset_attributes()
        #     pickle.dump(classifier, f_out)

        pass

    perf_df = pd.DataFrame(all_models_performance)
    logging.info(perf_df)

    tables_destination_path = get_workspace_path() + "Tables/"
    os.makedirs(tables_destination_path, exist_ok=True)
    perf_df.to_excel(tables_destination_path + "Classification_Results.xlsx")
    pass


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s -- %(filename)s %(module)s %(lineno)d",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename="ml_framework_classification.log",
    )
    main()

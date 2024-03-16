import os
import socket
import matplotlib
import pandas as pd
from collections import defaultdict
from ml_framework.tools.helper_functions import get_workspace_path
from ml_framework.data_regression.run_eda_regression import RegressionEDA
from ml_framework.data_regression.linear_regression import (
    LinearRegressor,
)
from ml_framework.data_regression.knn_regressor import KNN_Regressor
from ml_framework.data_regression.decision_tree_regressor import (
    DecisionTreeRegressor,
)
from ml_framework.data_regression.random_forest_regressor import (
    RandomForestRegressor,
)
from ml_framework.data_regression.support_vector_regressor import (
    SupportVectorRegressor,
)
from ml_framework.data_regression.xgboost_regressor import XGBoostRegressor
from ml_framework.data_regression.ann_tf_regressor import ANN_TF_Regressor

matplotlib.use("Agg")

# Set Data path
data_folder_path = ""
if socket.gethostname() == "LAPTOP-TFQFNF6U":
    data_folder_path = "F:/Weiterbildung/UOC_ML_Bootcamp/Capstone_Projects/Data/"
elif socket.gethostname() == "DLP":
    data_folder_path = "C:/Users/HFO/Documents/MachineLearning/Capstone_Projects/Data/"

data_filepath = data_folder_path + "diamonds.csv"


# Analyzer
target_col_name = "price"
analyzer = RegressionEDA(data_filepath)
analyzer.read_data()
analyzer.clean_data()
analyzer.encode_data(target_col_name=target_col_name)
# analyzer.visualize_data(target_col_name=target_col_name)
train_data, valid_data, test_data = analyzer.sample_data(
    target_col_name=target_col_name, train_perc=0.8, valid_perc=0.2
)


# Regression
regressors_ls = [
    "LinearRegressor",
    "KNN_Regressor",
    "DecisionTreeRegressor",
    "RandomForestRegressor",
    "XGBoostRegressor",
    "ANN_TF_Regressor",
    "SupportVectorRegressor",
]

all_models_performance = defaultdict(list)
for regressor_name in regressors_ls:
    regressor = eval(regressor_name + "(target_col_name, train_data, valid_data)")
    regressor.fit(nr_iterations=100)
    regressor.predict(test_data)
    score_dict = regressor.score()
    regressor.plot_scatterplot()

    all_models_performance["Model"].append(regressor_name)
    for k, v in score_dict.items():
        all_models_performance[k].append(v)

perf_df = pd.DataFrame(all_models_performance)
print(perf_df)

tables_destination_path = get_workspace_path() + "Tables/"
os.makedirs(tables_destination_path, exist_ok=True)
perf_df.to_excel(tables_destination_path + "Regression_Results.xlsx")
pass

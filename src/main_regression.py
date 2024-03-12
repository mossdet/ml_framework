import socket
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot
from collections import defaultdict
from ml_framework.data_regression.run_eda_regression import RegressionEDA
from ml_framework.data_regression.run_regressor import RunRegression

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
]
# regressors_ls = ["ANN_TF_Regressor"]
params = {
    "regressor_name": "",
    "target_col_name": target_col_name,
    "train_data": train_data,
    "valid_data": valid_data,
}
alll_models_perforance = defaultdict(list)
for regressor_name in regressors_ls:
    params["regressor_name"] = regressor_name
    regressor = RunRegression(**params)
    regressor.fit(nr_iterations=50)
    regressor.predict(test_data)
    score_dict = regressor.score()
    regressor.plot_scatterplot()

    for k, v in score_dict.items():
        alll_models_perforance[k].append(v)


pass

import os
import run_analyzer


# Set Data path
data_folder_path = "C:/Users/HFO/Documents/MachineLearning/Capstone_Projects/Data/"
data_filepath = data_folder_path + "diamonds.csv"


# Analyzer
analyzer = run_analyzer.Analyzer(data_filepath)
analyzer.read_data()
analyzer.clean_data()
analyzer.encode_data()
analyzer.visualize_data()
analyzer.sample_data()
df_data = analyzer.get_data()
df_train, df_test = analyzer.get_partitioned_data()

# Classifier


# Clustering

pass

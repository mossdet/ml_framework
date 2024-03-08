
import os
import analyzer


# Set Data amd Images paths
data_folder_path = "C:/Users/HFO/Documents/MachineLearning/Capstone_Projects/Data/"
data_filepath = data_folder_path + "diamonds.csv"

analyzer = analyzer.Analyzer(data_filepath)
df = analyzer.analyze_data()

pass
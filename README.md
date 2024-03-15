# ml_framework
This project defines the framework for future ML projects

## Package Description:
<br />

![Alt text](Flowchart/ml_framework_diagram.drawio.png "")
<br /><br /><br /><br />


# Machine Learning Framework

This Python library facilitates the steps needed for the training and testing of machine-learning models dealing with structured data. The framework provides methods for data analysis, classification regression and clustering.

## Installation

The library was developed using [poetry](https://python-poetry.org/) for dependency, environment and package management. A wheel file can be generated with the command:
```bash
poetry build
```

The wheel file itself can be used to install the library using [pip](https://pip.pypa.io/en/stable/):
```bash
pip install ml_framework-0.1.0-py3-none-any.whl
```

## Usage

### Data Analysis
```python
import ml_framework

# Analyzer
target_col_name = "target"
analyzer = ClassificationEDA(data_filepath)
analyzer.read_data()
analyzer.clean_data()
analyzer.encode_data(target_col_name=target_col_name)
analyzer.visualize_data(target_col_name=target_col_name)
train_data, valid_data, test_data = analyzer.sample_data(
  target_col_name=target_col_name, train_perc=0.8, valid_perc=0.2
)
```


### Development of Classifiers
```python
# Classifier
import ml_framework

classifiers_ls = [
    "LogisticRegressionClassifier",
    "KNN_Classifier",
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "XGBoostClassifier",
    "ANN_TF_Classifier",
    "SupportVectorClassifier",
]

classifiers_performance = defaultdict(list)
for classifier_name in classifiers_ls:
    classifier = eval(classifier_name + "(target_col_name, train_data, valid_data)")
    classifier.fit(nr_iterations=100)
    classifier.predict(test_data)
    score_dict = classifier.score()
    classifier.plot_confusion_matrix()

    classifiers_performance["Model"].append(classifier_name)
    for k, v in score_dict.items():
        classifiers_performance[k].append(v)
```
```console
(ml-framework-py3.11) PS C:\Documents\ml_framework\src> python .\main_classification.py
                          Model  Precision    Recall  Accuracy  F1-Score
0  LogisticRegressionClassifier   0.636091  0.520693  0.651469  0.541070
1                KNN_Classifier   0.684462  0.637773  0.686313  0.654432
2        DecisionTreeClassifier   0.693999  0.645946  0.717820  0.655758
3        RandomForestClassifier   0.801956  0.759292  0.779817  0.769552
4             XGBoostClassifier   0.819495  0.793395  0.810861  0.805071
5             ANN_TF_Classifier   0.806076  0.777483  0.800297  0.790336
6       SupportVectorClassifier   0.782612  0.747490  0.780743  0.762896
```


### Development of Regressors
```python
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

regressors_performance = defaultdict(list)
for regressor_name in regressors_ls:
    regressor = eval(regressor_name + "(target_col_name, train_data, valid_data)")
    regressor.fit(nr_iterations=100)
    regressor.predict(test_data)
    score_dict = regressor.score()
    regressor.plot_scatterplot()

    regressors_performance["Model"].append(regressor_name)
    for k, v in score_dict.items():
        regressors_performance[k].append(v)
```
```console
(ml-framework-py3.11) PS C:\Documents\ml_framework\src> python .\main_regression.py
                    Model  R2_Score  ...  Mean_Absolute_Error  Mean_Absolute_Percentage_Error
0         LinearRegressor  0.904075  ...           822.070293                        0.437571
1           KNN_Regressor  0.959968  ...           400.844642                        0.113064
2   DecisionTreeRegressor  0.952484  ...           435.645532                        0.109457
3   RandomForestRegressor  0.982113  ...           274.680662                        0.070769
4        XGBoostRegressor  0.982173  ...           276.939906                        0.073591
5        ANN_TF_Regressor  0.975925  ...           322.883121                        0.088762
6  SupportVectorRegressor  0.965810  ...           513.631381                        0.250324
```


### Development of Clustering
```python
# Clustering
clusterings_ls = [
    "KMeansClustering",
    "AgglomerativeClustering",
    "MeanShiftClustering",
    "DBSCAN_Clustering",
]

clustering_performance = defaultdict(list)
for clustering_name in clusterings_ls:
    clustering = eval(clustering_name + "(train_data=train_data)")
    clustering.fit(nr_iterations=100)
    clustering.predict(test_data)
    score_dict = clustering.score()

    clustering_performance["Model"].append(clustering_name)
    for k, v in score_dict.items():
        clustering_performance[k].append(v)
    clustering_performance["NrClusters"].append(clustering.get_num_clusters())
```
```console
(ml-framework-py3.11) PS C:\Documents\ml_framework\src> python .\main_clustering.py
                    Model  R2_Score  ...  Mean_Absolute_Error  Mean_Absolute_Percentage_Error
0         LinearRegressor  0.904075  ...           822.070293                        0.437571
1           KNN_Regressor  0.959968  ...           400.844642                        0.113064
2   DecisionTreeRegressor  0.952484  ...           435.645532                        0.109457
3   RandomForestRegressor  0.982113  ...           274.680662                        0.070769
4        XGBoostRegressor  0.982173  ...           276.939906                        0.073591
5        ANN_TF_Regressor  0.975925  ...           322.883121                        0.088762
6  SupportVectorRegressor  0.965810  ...           513.631381                        0.250324
```


## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
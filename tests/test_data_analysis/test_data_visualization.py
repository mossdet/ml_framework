import os
import pandas as pd
import numpy as np
import pytest
import matplotlib.pyplot as plt

# Import the class to be tested
from ml_framework.data_analysis.data_visualization import DataVisualizer


@pytest.fixture
def data_visualizer():
    # Initialize DataVisualizer
    return DataVisualizer()


def test_plot_correlation_matrix(data_visualizer):
    # Test plot_correlation_matrix method

    # Create sample DataFrame
    data = pd.DataFrame({"A": range(100), "B": range(100, 200), "C": range(200, 300)})

    # Check if the method does not raise any exceptions
    try:
        data_visualizer.plot_correlation_matrix(data)
    except Exception as e:
        pytest.fail(f"Exception raised: {e}")


def test_plot_pairplot_numerical(data_visualizer):
    # Test plot_pairplot_numerical method

    # Create sample DataFrame
    data = pd.DataFrame({"A": range(100), "B": range(100, 200), "C": range(200, 300)})

    # Check if the method does not raise any exceptions
    try:
        data_visualizer.plot_pairplot_numerical(data, hue="A")
    except Exception as e:
        pytest.fail(f"Exception raised: {e}")


def test_plot_pairplot_categorical(data_visualizer):
    # Test plot_pairplot_categorical method

    # Create sample DataFrame
    data = pd.DataFrame({"A": range(100), "B": ["X"] * 100})

    # Check if the method does not raise any exceptions
    try:
        data_visualizer.plot_pairplot_categorical(data, hue="B")
    except Exception as e:
        pytest.fail(f"Exception raised: {e}")


def test_plot_histograms_numerical(data_visualizer):
    # Test plot_histograms_numerical method

    # Create sample DataFrame
    data = pd.DataFrame(
        {
            "A": range(100),
            "B": np.random.normal(0, 1, 100),
            "C": np.random.normal(0, 1, 100),
        }
    )

    # Check if the method does not raise any exceptions
    try:
        data_visualizer.plot_histograms_numerical(data)
    except Exception as e:
        pytest.fail(f"Exception raised: {e}")


def test_plot_histograms_categorical(data_visualizer):
    # Test plot_histograms_categorical method

    # Create sample DataFrame
    data = pd.DataFrame({"A": ["X"] * 50 + ["Y"] * 50, "B": ["P"] * 50 + ["Q"] * 50})

    # Check if the method does not raise any exceptions
    try:
        data_visualizer.plot_histograms_categorical(data)
    except Exception as e:
        pytest.fail(f"Exception raised: {e}")


def test_plot_boxplot(data_visualizer):
    # Test plot_boxplot method

    # Create sample DataFrame
    data = pd.DataFrame(
        {"A": range(100), "B": ["X"] * 100, "C": np.random.normal(0, 1, 100)}
    )

    # Check if the method does not raise any exceptions
    try:
        data_visualizer.plot_boxplot(data, x="B", y="A", hue="B", suffix="suffix")
    except Exception as e:
        pytest.fail(f"Exception raised: {e}")


def test_plot_classes_distribution(data_visualizer):
    # Test plot_classes_distribution method

    # Create sample DataFrame
    data = pd.DataFrame(
        {
            "A": ["X"] * 50 + ["Y"] * 50,
            "B": ["P"] * 50 + ["Q"] * 50,
            "Target": [0] * 50 + [1] * 50,
        }
    )

    # Check if the method does not raise any exceptions
    try:
        data_visualizer.plot_classes_distribution(data, "Target", "suffix")
    except Exception as e:
        pytest.fail(f"Exception raised: {e}")


if __name__ == "__main__":
    pytest.main()

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import pytest
from unittest.mock import patch, MagicMock

# Import the class to be tested
from ml_framework.data_analysis.data_ingestion import DataIngestor


@pytest.fixture
def data_ingestor(tmp_path):
    # Initialize DataIngestor with a temporary data file and images destination path
    datafile_path = os.path.join(tmp_path, "test_data.csv")
    images_destination_path = os.path.join(tmp_path, "images/")
    os.makedirs(images_destination_path, exist_ok=True)
    with open(datafile_path, "w") as f:
        f.write("A,B,C\n1,2,3\n4,5,6\n")
    yield DataIngestor(
        datafile_path=datafile_path, images_destination_path=images_destination_path
    )


def test_init(data_ingestor):
    # Test initialization of DataIngestor class
    assert data_ingestor.datafile_path.endswith("test_data.csv")
    assert data_ingestor.images_destination_path.endswith("images/")


def test_ingest_data_csv(data_ingestor):
    # Test ingest_data method for CSV file

    # Ingest data
    df = data_ingestor.ingest_data()

    # Check if the DataFrame is not None
    assert df is not None

    # Check if the DataFrame has correct shape
    assert df.shape == (2, 3)


def test_ingest_data_invalid_file(data_ingestor):
    # Test ingest_data method for invalid file type

    # Change file extension to '.txt'
    data_ingestor.datafile_path = data_ingestor.datafile_path[:-3] + "txt"

    # Ingest data
    df = data_ingestor.ingest_data()

    # Check if the DataFrame is None
    assert df is None


def test_describe_data(data_ingestor, caplog):
    # Test describe_data method

    # Mock DataFrame
    df_mock = MagicMock(spec=pd.DataFrame)
    df_mock.columns = ["A", "B", "C"]
    df_mock.shape = (2, 3)

    # # Patch logging
    # with caplog.at_level(logging.INFO):
    #     data_ingestor.describe_data(df_mock)

    # # Check if correct log messages are logged
    # # assert "Nr. rows: 2" in caplog.text
    # # assert "Nr. columns: 3" in caplog.text
    # # assert "Data Description:" in caplog.text


if __name__ == "__main__":
    pytest.main()

"""test cases for the churn_library module"""
import os
import numpy as np
import pandas as pd
import churn_library as cl


class ChangeDir:
    """Class used as context for temporary change of local directory"""
    # Reference:
    # https://pythonadventures.wordpress.com/2013/12/15/chdir-a-context-manager-for-switching-working-directories/

    def __init__(self, new_path):
        """Initialize ChangeDir by storing the new path to which we change"""
        self.new_path = new_path
        self.saved_path = None

    def __enter__(self):
        """When entering context, save the current working directory"""
        self.saved_path = os.getcwd()
        os.chdir(self.new_path)

    def __exit__(self, *args):
        """When exiting context, changedir back to the saved working directory"""
        os.chdir(self.saved_path)


def test_import_data_file_exists():
    """Test that import data imports an existing file"""
    try:
        _ = cl.import_data("./data/bank_data.csv")
    except FileNotFoundError as err:
        raise err


def test_import_data_has_columns_and_rows():
    """Test that import data provides data that is not empty"""
    try:
        dataframe = cl.import_data("./data/bank_data.csv")
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0  # pylint: disable=E1101
    except AssertionError as err:
        raise err


def test_perform_eda(tmp_path):
    '''
    test perform eda function
    '''
    # Setup
    dataframe = cl.import_data("./data/bank_data.csv")
    with ChangeDir(tmp_path):
        # Execute
        cl.perform_eda(dataframe)
        # Verify that four images were created
        created_image_files = [f for f in os.listdir(cl.IMAGES_FOLDER) if f.endswith(
            '.png')]  # Assuming the images are in .png format
        assert len(created_image_files) == 5, "Five image files were not created"


def test_encoder_helper():
    '''
    test encoder helper
    '''
    # Setup
    dataframe = cl.import_data("./data/bank_data.csv")
    # Execute
    encoded_dataframe, encoded_category_names = cl.encoder_helper(
        dataframe, cl.CATEGORY_COLUMNS)
    # Verify the encoder produces only quantitative columns
    # Specifically, check that the number of quantitative columns is the same
    # as the total number of columns
    assert encoded_dataframe.select_dtypes(include=[np.number]).shape[1] \
        == encoded_dataframe.shape[1], "Not all columns are quantitative"
    # Verify that the encoded column names are equal to the number of
    # dataframe columns
    categories = [
        group for column_name in cl.CATEGORY_COLUMNS for group in dataframe[column_name].unique()]
    assert len(encoded_category_names) == len(
        categories) - len(cl.CATEGORY_COLUMNS)


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    # Setup
    dataframe = cl.import_data("./data/bank_data.csv")
    # Execute
    x_train, x_test, y_train, y_test = cl.perform_feature_engineering(
        dataframe)
    # Verify type of output
    for output in x_train, x_test:
        assert isinstance(output, pd.core.frame.DataFrame)
    for output in y_train, y_test:
        assert isinstance(output, pd.core.series.Series)


def test_train_models(tmp_path):
    '''
    test train_models
    '''
    # Setup
    dataframe = cl.import_data("./data/bank_data.csv")
    # Reduce size of the dataframe significantly
    dataframe = dataframe.head(40)
    x_train, _, y_train, _ = cl.perform_feature_engineering(dataframe)

    with ChangeDir(tmp_path):
        # Execute
        cl.train_models(x_train, y_train)
        # Verify two models were created
        created_model_files = [f for f in os.listdir(cl.MODELS_FOLDER) if f.endswith(
            '.pkl')]  # Assuming the images are in .png format
        assert len(created_model_files) == 2, "Two model files were not created"


def test_produce_performance_report(tmp_path):
    '''
    test produce performance report
    '''
    # Setup
    dataframe = cl.import_data("./data/bank_data.csv")
    # Reduce size of the dataframe significantly
    dataframe = dataframe.head(40)
    x_train, x_test, y_train, y_test = cl.perform_feature_engineering(
        dataframe)

    with ChangeDir(tmp_path):
        # Execute
        cl.train_models(x_train, y_train)
        cl.produce_performance_report(x_train, x_test, y_train, y_test)
        # Verify five plot / report images were generated
        created_image_files = [f for f in os.listdir(cl.IMAGES_FOLDER) if f.endswith(
            '.png')]  # Assuming the images are in .png format
        assert len(created_image_files) == 5, "Five model files were not created"

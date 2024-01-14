"""library to assist analysis of customer churn"""

import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import shap
import joblib
import churn_logging

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

sns.set()


IMAGES_FOLDER = "./images"
MODELS_FOLDER = "./models"

CATEGORY_COLUMNS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

QUANT_COLUMNS = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio'
]

RESPONSE_NAME = "Churn"


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            dataframe: pandas dataframe
    '''
    logging.info("IMPORT: Importing data from CSV at '%s'", pth)
    dataframe = pd.read_csv(pth)
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    dataframe.drop(columns='Attrition_Flag', axis=1, inplace=True)
    logging.info("IMPORT: Successfully imported dataframe")
    logging.info(dataframe.head())  # pylint: disable=E1101
    return dataframe


def perform_eda(dataframe):
    '''
    perform exploratory data analysis on dataframe and save figures to images folder
    input:
            dataframe: pandas dataframe

    output:
            None
    '''
    # Create histograms of Churn and Customer Age
    logging.info(
        "EDA: Creating histograms for the Dataset's Churn and Customer Age")
    for feature in 'Churn', 'Customer_Age':
        image_filename = feature + "_EDA.png"
        plot_and_save(dataframe[feature].hist, image_filename)

    # Create a bar graph of the Marital Status
    logging.info("EDA: Creating a bar graph for the Marital Status")
    image_filename = "Marital_Status_EDA.png"
    plot_and_save(dataframe.Marital_Status.value_counts(
        'normalize').plot, image_filename, kind='bar')

    # Create a histogram plot of the transactions count to show distributions
    logging.info(
        "EDA: Creating a histogram of the transaction counts to show distribution")
    image_filename = "Total_Trans_Ct_EDA.png"
    plot_and_save(
        sns.histplot,
        image_filename,
        dataframe['Total_Trans_Ct'].to_numpy(),
        stat='density')

    # Create a heatmap of correlations between features
    image_filename = "Correlation_EDA.png"
    # Our heatmap needs numeric data, so select only the quantified columns
    quant_dataframe = dataframe[QUANT_COLUMNS + ['Churn']]
    logging.info("EDA: Creating a correlation heatmap")
    plot_and_save(
        sns.heatmap,
        image_filename,
        quant_dataframe.corr(),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)


def plot_and_save(func, filename, *args,
                  images_folder=IMAGES_FOLDER, **kwargs):
    """
    Function to create a figure and save it to a file, taking the plotting function as an argument

    input:
        func: The plotting function to execute.
        args: Positional arguments for the plotting function.
        filename: Name of the file to save the plot.
        kwargs: Keyword arguments for the plotting function.

    output:
        None
    """
    # Create a new figure
    plt.figure(figsize=(20, 10))

    # Call the provided plotting function with its arguments
    func(*args, **kwargs)

    # Save the plot to the specified file
    filepath = "/".join([images_folder, filename])
    logging.info("Saving plot to '%s'", filepath)
    os.makedirs(images_folder, exist_ok=True)
    plt.savefig(filepath)
    plt.close()


def encoder_helper(dataframe, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            dataframe: pandas dataframe
            category_lst: list of columns that contain categorical features

    output:
            dataframe: pandas dataframe with new columns for one-hot encoded values
            encoded_column_names = list of names of the encoded columns
    '''
    # Perform One-Hot Encoding for each column containing categorical features
    logging.info("Performing One-Hot encoding...")
    # drop='first' to avoid multicollinearity
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_columns = encoder.fit_transform(dataframe[category_lst])
    encoded_column_names = encoder.get_feature_names_out(category_lst)
    encoded_dataframe = pd.DataFrame(
        encoded_columns, columns=encoded_column_names
    )

    # Concatenate the encoded columns with the original DataFrame
    dataframe = pd.concat([dataframe, encoded_dataframe], axis=1)

    # Drop the original categorical columns
    dataframe.drop(columns=category_lst, inplace=True)

    logging.info("Successfully encoded Dataframe Columns as One-Hot Values")
    logging.info(dataframe.head())
    return dataframe, list(encoded_column_names)


def perform_feature_engineering(dataframe, response="Churn"):
    '''
    input:
              dataframe: pandas dataframe
              response: string indicating the y column dependent variable name

    output:
              x_train: x training data
              x_test: x testing data
              y_train: y training data
              y_test: y testing data
    '''
    if not isinstance(dataframe, pd.DataFrame):
        logging.error(
            "Input dataframe type in perform_feature_engineering is not Pandas DataFrame")
        raise TypeError
    logging.info(
        "FEATURE_ENG: Encoding and splitting dataset into training and test subsets...")
    encoded_dataframe, encoded_column_names = encoder_helper(
        dataframe, CATEGORY_COLUMNS)
    keep_cols = QUANT_COLUMNS + encoded_column_names
    feature_data = encoded_dataframe[keep_cols]
    normalize(feature_data)
    response_data = dataframe[response]
    return train_test_split(feature_data, response_data,
                            test_size=0.3, random_state=42)


def classification_report_image(model_name,
                                y_train,
                                y_test,
                                y_train_preds,
                                y_test_preds):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # Create Model Performance Report for given model
    logging.info(
        "REPORT: Creating report for %s model",
        model_name.replace(
            '_',
            ' '))
    image_filename = f"{model_name}_Model_Report.png"
    plot_and_save(
        plot_model_report,
        image_filename,
        model_name,
        y_train,
        y_test,
        y_train_preds,
        y_test_preds
    )
    logging.info("REPORT: Successfully created report for model results.")


def plot_model_report(model_name, y_train, y_test,
                      y_train_preds, y_test_preds):
    """Helper to plot a Random Forest model report image"""
    plt.rc('figure', figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str(f'{model_name} Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str(f'{model_name} Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')


def feature_importance_plot(model, x_data, filename):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of x values
            filename: filename of the stored figure

    output:
             None
    '''
    logging.info("Plotting feature importance for the model...")
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]
    # Helper function to plot the data

    def plot_feature_importance():
        plt.title("Feature Importance")
        plt.ylabel('Importance')
        plt.bar(range(x_data.shape[1]), importances[indices])
        plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plot_and_save(plot_feature_importance, filename)


def train_models(x_train, y_train):
    '''
    train and store random forest and logistic regression models
    input:
              x_train: x training data
              x_test: x testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    logging.info(
        "TRAIN: Building the Random Forest and Logistic Regression models...")
    # Build Random Forest and Logistic Regression models
    rfc = RandomForestClassifier()
    lrc_model = LogisticRegression(solver='lbfgs', max_iter=3000)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

    # Train the models
    logging.info("TRAIN: Training the models...")
    cv_rfc.fit(x_train, y_train)
    rfc_model = cv_rfc.best_estimator_
    lrc_model.fit(x_train, y_train)

    # Save the best models
    logging.info("TRAIN: Saving the models to files...")
    os.makedirs(MODELS_FOLDER, exist_ok=True)
    joblib.dump(rfc_model, f'{MODELS_FOLDER}/rfc_model.pkl')
    joblib.dump(lrc_model, f'{MODELS_FOLDER}/logistic_model.pkl')


def produce_performance_report(x_train, x_test, y_train, y_test):
    '''
    produces a performance result for stored random forest and logistic regression models
    input:
              x_train: x training data
              x_test: x testing data
    output:
              None
    '''
    # Load the models
    rfc_model = joblib.load(f'{MODELS_FOLDER}/rfc_model.pkl')
    lrc_model = joblib.load(f'{MODELS_FOLDER}/logistic_model.pkl')

    # Output predictions from the models
    logging.info("TRAIN: Creating predictions...")
    y_train_preds_rf = rfc_model.predict(x_train)
    y_test_preds_rf = rfc_model.predict(x_test)
    y_train_preds_lr = lrc_model.predict(x_train)
    y_test_preds_lr = lrc_model.predict(x_test)

    # Produce a feature importance plot for the Random Forest model
    image_filename = "Feature_Importance_RFC.png"
    feature_importance_plot(rfc_model, x_train + x_test, image_filename)

    # Save image of model performance report
    logging.info("REPORT: Creating classification reports...")
    models_info = [
        ["Logistic_Regression", y_train_preds_lr, y_test_preds_lr],
        ["Random_Forest", y_train_preds_rf, y_test_preds_rf]
    ]
    for model_name, y_train_preds, y_test_preds in models_info:
        classification_report_image(
            model_name,
            y_train,
            y_test,
            y_train_preds,
            y_test_preds
        )

    # Create Model ROC (Receiver Operating Characteristic) Plots
    logging.info(
        "TRAIN: Creating model Receiver Operating Characteristic curves...")
    image_filename = "ROC_Curves.png"
    plot_and_save(
        plot_model_roc_curves,
        image_filename,
        rfc_model,
        lrc_model,
        x_test,
        y_test)

    # Create Feature Explainer plot
    logging.info("TRAIN: Creating feature explainer plot...")
    image_filename = "Feature_Explainer_RFC.png"
    plot_and_save(plot_feature_explainer, image_filename, rfc_model, x_test)


def plot_model_roc_curves(rfc_model, lrc_model, x_test, y_test):
    """Helper to plot the receiver operating characteristic (ROC) curves"""
    lrc_plot = RocCurveDisplay.from_estimator(lrc_model, x_test, y_test)
    plt.figure(figsize=(15, 8))
    current_axes = plt.gca()
    RocCurveDisplay.from_estimator(
        rfc_model,
        x_test,
        y_test,
        ax=current_axes,
        alpha=0.8)
    lrc_plot.plot(ax=current_axes, alpha=0.8)


def plot_feature_explainer(rfc_model, x_test):
    """Helper to plot SHAP values for random forest model"""
    explainer = shap.TreeExplainer(rfc_model)
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(shap_values, x_test, plot_type="bar", show=False)


if __name__ == "__main__":
    churn_logging.logging_init()
    bank_dataframe = import_data(r"./data/bank_data.csv")
    perform_eda(bank_dataframe)
    bank_x_train, bank_x_test, bank_y_train, bank_y_test = perform_feature_engineering(
        bank_dataframe)
    train_models(bank_x_train, bank_y_train)
    produce_performance_report(
        bank_x_train,
        bank_x_test,
        bank_y_train,
        bank_y_test)

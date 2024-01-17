# Predict Customer Churn

- Eric Koch's Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project serves to demonstrate refactoring code to best practices, including implementing linting, testing, and logging.

The original Jupyter notebook uses scikit-learn to examine a bank's data on customer churn first through exploratory data analysis (EDA), and later through training logistic regression and random forest classifiers that predict customer churn based on the available features.

In short summary of findings, the most influential factors on customer churn were their credit card usage (e.g. transaction numbers, rotating balance, etc). Other factors like marital status, education, or gender had far less impact. The random forest classifier proved the best predictor for churn, with ~95% precision, ~95% recall, and ~95% F1-score.

We can explain this result by recognizing that frequent users of credit cards are by definition more active with those cards, and therefore more likely to make changes (i.e. switch banks). In contrast, infrequent users of a card would have little reason to stop their relationship with the bank, as they do not pay that relationship much attention.

## Files and data description
`./images/` - Our EDA and report plot images are saved here
`./models/` - Our trained models are saved here
`./logs/` - Our logs are saved here

## Running Files
To run, first set up your python venv environment through our Makefile with:
`make initvenv`
Subsequently, run:
`make venv`
to enter that virtual environment.
Next, run:
`make install`
to install necessary packages for the project.
To lint and test the package, run:
`make lint`
and subsequently
`make test`
To run the core functionality of the churn library, run:
`make run`





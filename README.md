# Titanic-MachineLearning

## Titanic Survival Analysis
This repository contains analyses of the Titanic dataset aimed at predicting passenger survival using two machine learning approaches: Logistic Regression and K-Nearest Neighbors (KNN).

## Repository Contents
### Data Files
gender_submission.csv: A sample submission file in the correct format for Kaggle's Titanic competition.
test.csv: The test dataset containing passenger information without the survival labels.
train.csv: The training dataset containing passenger information with the survival labels.

### Analysis Notebooks
Titanic_LogisticRegression.ipynb: A Jupyter Notebook that performs logistic regression analysis on the Titanic dataset. This notebook includes data preprocessing, feature engineering, model training, and evaluation.

Titanic_KNN.ipynb: A Jupyter Notebook that implements the K-Nearest Neighbors algorithm to classify passengers as survivors or non-survivors based on their attributes.

## Analysis Overview
### Logistic Regression Analysis
The logistic regression analysis aims to model the probability of survival as a function of several features, such as age, sex, class, and more. The steps included in this analysis are:

### Data Cleaning and Preprocessing: Handling missing values, encoding categorical variables, and scaling features.
Exploratory Data Analysis (EDA): Visualizing the data to understand the distribution and relationships between features.
### Feature Engineering: Creating new features from existing ones to improve model performance.
### Model Training: Training a logistic regression model on the training dataset.
### Model Evaluation: Evaluating model performance using metrics like accuracy, precision, recall, and F1 score.
K-Nearest Neighbors (KNN) Analysis
### The KNN analysis involves the following steps:

#### Data Preprocessing and Normalization: Handling missing values, encoding categorical variables, and normalizing features.
#### Optimal K Selection: Using cross-validation to select the optimal value of K for the KNN algorithm.
#### Model Training: Training the KNN model on the training dataset.
#### Model Prediction: Predicting survival on the test dataset.
#### Model Evaluation: Evaluating model performance using metrics like accuracy, precision, recall, and F1 score.

## Conclusions
### Logistic Regression
#### Feature Importance: The logistic regression model highlighted key features influencing survival, such as passenger class (Pclass), sex, and age.
#### Performance: The model demonstrated good accuracy and provided insights into the likelihood of survival based on different attributes.
K-Nearest Neighbors
#### Impact of K Value: The choice of K significantly impacted model performance, with an optimal K providing the best balance between bias and variance.
#### Performance: The KNN model effectively classified passengers, though its performance was sensitive to feature scaling and the chosen value of K.

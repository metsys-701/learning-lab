"""
Topic: Telco Customer Churn Prediction

Description:
This file predicts customer churn using the Telco dataset. Fixed a
messy TotalCharges column, encoded all categorical columns with
get_dummies, then trained and compared Logistic Regression and
Random Forest.
"""
#=========================
#1. Imports
#=========================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

#=========================
#2. Load and Explore Data
#=========================
# Load the dataset and check its structure, data types, and missing values

df = pd.read_csv("telco_churn.csv")
print(df.head(5))
print(df.info())
print(df.describe())
print(df.isnull().sum())


#=========================
#3. Fix TotalCharges Column
#=========================
# TotalCharges looked numeric but was actually stored as text (str),
# because some rows have a space character instead of a real number
# (likely new customers with tenure=0 who haven't been billed yet).
# isnull().sum() didn't catch this because a space isn't technically NaN.

# Check which rows have the space character
print(df[df['TotalCharges'].str.strip() == ''])

# Convert to numeric; any value that can't convert (the spaces) becomes NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Now that they're real NaN values, drop those rows (only 11 out of 7043)
df.dropna(subset=['TotalCharges'], inplace=True)

print(df.info())

#=========================
#4. Encode Categorical Columns
#=========================
# Convert categorical text columns into 0/1 columns using one-hot encoding

df.drop(columns=['customerID'], inplace=True)
df = pd.get_dummies(df, columns=['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod','Churn'], drop_first=True)

print(df.info())

#=========================
#5. Split Features and Target
#=========================

X = df.drop(columns=['Churn_Yes'])
y = df['Churn_Yes']


#=========================
#6. Train/Test Split
#=========================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#=========================
#7. Model Training
#=========================

logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

random_forest_model = RandomForestClassifier(n_estimators=100, max_depth=3)
random_forest_model.fit(X_train, y_train)

#=========================
#8. Prediction and Evaluation
#=========================
logistic_prediction = logistic_model.predict(X_test)
random_forest_prediction = random_forest_model.predict(X_test)

print("Logistic regression prediction:", logistic_prediction)
print("Random forest prediction:", random_forest_prediction)

logistic_Acc = accuracy_score(y_test, logistic_prediction)
random_forest_Acc = accuracy_score(y_test, random_forest_prediction)

print("Logistic regression Accuracy:", logistic_Acc)
print("Random forest Accuracy:", random_forest_Acc)

#=========================
#9. Result Note
#=========================
# Logistic Regression accuracy: ~79.6%
# Random Forest accuracy: ~78.7%
# Pretty close, Logistic Regression did slightly better. This dataset
# had more columns than Titanic (a lot of Yes/No service columns) and
# needed get_dummies instead of map() since some columns had more than
# 2 categories. Also had to fix TotalCharges since it looked numeric
# but was actually stored as text because of empty-space values.
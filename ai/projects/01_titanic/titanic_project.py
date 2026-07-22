#=========================
#0. Imports
#=========================
# Import pandas, numpy, and the tools you'll need for train/test split, modeling, and evaluation

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score

#=========================
#1. Load and Explore Data
#=========================
# Read the dataset and look at its structure

df = pd.read_csv("train.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

#=========================
#2. Feature Engineering
#=========================
# Drop irrelevant columns, handle missing data, create FamilySize feature

df.drop(columns=['Name', 'Ticket', 'PassengerId', 'Cabin', 'Fare', 'Embarked'], inplace=True)
df.dropna(axis=0, how='any', inplace=True)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

#=========================
#3. Split Features and Target
#=========================
# Separate the target variable (Survived) from the features

X = df.drop(columns=['Survived'])
y = df['Survived']

#=========================
#4. Train/Test Split
#=========================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#=========================
#5. Model Training
#=========================
# Train both Logistic Regression and Decision Tree models

model = LogisticRegression()
model.fit(X_train, y_train)

tree_model = DecisionTreeClassifier(max_depth= 3)
tree_model.fit(X_train, y_train)

#=========================
#6. Prediction and Evaluation
#=========================
# Compare accuracy between the two models

predictions = model.predict(X_test)
print("Predictions:", predictions)

predict_tree = tree_model.predict(X_test)
print("Tree predict:", predict_tree)

Acc_model = accuracy_score(y_test, predictions)
Acc_tree_model = accuracy_score(y_test, predict_tree)

print("Acc model:", Acc_model)
print("Acc tree model:", Acc_tree_model)

#=========================
#7. Model Comparison Note
#=========================
# Logistic Regression accuracy: ~81%
# Decision Tree (max_depth=3) accuracy: ~71%
# Logistic Regression did better this time. Maybe max_depth=3 was
# too small for the tree and it couldn't split the data enough,
# or maybe this dataset just fits Logistic Regression's approach better. 
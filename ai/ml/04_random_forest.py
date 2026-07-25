"""
Topic: Random Forest

Description:
This file contains my practice exercises for Random Forest,
using the same Titanic dataset from my mini project, and
comparing its accuracy with Decision Tree.
"""

#=========================
#1. Imports
#=========================
# Import pandas, train_test_split, RandomForestClassifier, and accuracy_score

import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#=========================
#2. Load and Prepare Data
#=========================
# Load the dataset and repeat the same feature engineering steps as before

df = pd.read_csv("../projects/01_titanic/train.csv")
print(df.head(5))
print(df.info())
print(df.describe())
print(df.isnull().sum())

df.drop(columns=['Name', 'Ticket', 'PassengerId', 'Cabin', 'Fare', 'Embarked'], inplace=True)
df.dropna(axis= 0 , how='any', inplace=True)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['Sex'] = df['Sex'].map({'male':0, 'female': 1})


#=========================
#3. Split Features and Target
#=========================

X = df.drop(columns=['Survived'])
y = df['Survived']

#=========================
#4. Train/Test Split
#=========================

X_train , X_test , y_train, y_test = train_test_split(X, y , test_size=0.2 )


#=========================
#5. Random Forest Model
#=========================
# Train a Random Forest model with 100 trees and limited depth

model = RandomForestClassifier(n_estimators= 100, max_depth=3)
model.fit(X_train, y_train)

#=========================
#6. Prediction and Evaluation
#=========================
# Predict and calculate accuracy

predictions = model.predict(X_test)
print("Predictions:", predictions)

Acc_forest = accuracy_score(y_test, predictions)
print("Acc forest:", Acc_forest)

#=========================
#7. Result Note
#=========================
# Logistic Regression: ~81%, Decision Tree: ~71%, Random Forest: ~80%
# Random Forest did much better than a single Decision Tree, which makes
# sense since it combines many trees instead of relying on just one.
# It's close to Logistic Regression but slightly behind here, maybe
# because this dataset is small and simple enough that Logistic
# Regression's approach already works well.
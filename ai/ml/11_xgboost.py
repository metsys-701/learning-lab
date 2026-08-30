"""
Topic: XGBoost

Description:
This file trains an XGBoost classifier on the Titanic dataset and
compares its accuracy with the other models I've tried so far.
"""
#=========================
#1. Imports
#=========================
# Import pandas, train_test_split, XGBClassifier, and accuracy_score

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#=========================
#2. Load and Prepare Data
#=========================
# Same Titanic feature engineering as before

df = pd.read_csv("../projects/01_titanic/train.csv")

df.drop(columns=['Name', 'Ticket', 'PassengerId', 'Cabin', 'Fare', 'Embarked'], inplace=True)
df.dropna(axis=0, how='any', inplace=True)
df["FamilySize"] = df['SibSp'] + df['Parch'] + 1
df['Sex'] = df['Sex'].map({'male':0, 'female':1})

#=========================
#3-4. Split Features/Target, Train/Test Split
#=========================

X = df.drop(columns=['Survived'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#=========================
#5. XGBoost Model
#=========================
# Train an XGBoost model

model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
model.fit(X_train, y_train)


#=========================
#6. Prediction and Evaluation
#=========================
# Predict and calculate accuracy

prediction = model.predict(X_test)
print("model prediction:", prediction )

Acc = accuracy_score(y_test, prediction)
print("Accuracy score:", Acc)

#=========================
#7. Result Note
#=========================
# Accuracy: ~80.4%
# Exactly the same as Random Forest (~80.4%). Makes sense that boosting
# didn't show a big advantage here — Titanic is a small dataset (~700
# rows), and XGBoost's strength usually shows up more on bigger, more
# complex data like Telco.
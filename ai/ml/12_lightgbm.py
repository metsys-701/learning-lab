"""
Topic: LightGBM

Description:
This file trains a LightGBM classifier on the Titanic dataset and
compares its accuracy with XGBoost and the other models I've tried.
"""
#=========================
#1. Imports
#=========================
# Import pandas, train_test_split, LGBMClassifier, and accuracy_score

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#=========================
#2. Load and Prepare Data
#=========================
# Same Titanic feature engineering as before

df = pd.read_csv("../projects/01_titanic/train.csv")

df.drop(columns=['Name', 'Ticket', 'PassengerId', 'Cabin', 'Fare', 'Embarked'], inplace=True)
df.dropna(axis=0, how='any', inplace=True)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['Sex'] = df['Sex'].map({'male':0, 'female':1})

#=========================
#3-4. Split Features/Target, Train/Test Split
#=========================

X = df.drop(columns=['Survived'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#=========================
#5. LightGBM Model
#=========================
# Train a LightGBM model

model = LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
model.fit(X_train,y_train)

#=========================
#6. Prediction and Evaluation
#=========================
# Predict and calculate accuracy

predict = model.predict(X_test)
Acc = accuracy_score(y_test, predict)

print("Prediction", predict)
print("Accuracy score", Acc)

#=========================
#7. Result Note
#=========================
# Accuracy: ~79%
# Slightly lower than XGBoost (~80.4%) here. Got a warning about "no
# further splits with positive gain" during training, which makes
# sense on a small dataset like Titanic — LightGBM's leaf-wise growth
# and speed advantages are built for bigger data, so it doesn't really
# show an edge here, and might even be slightly worse.
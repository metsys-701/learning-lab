"""
Topic: Naive Bayes

Description:
This file contains my practice exercises for Naive Bayes, using the
same Titanic dataset. No scaling needed here since Naive Bayes works
with probabilities, not distances.
"""

#=========================
#1. Imports
#=========================
# Import pandas, train_test_split, GaussianNB, and accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

#=========================
#2. Load and Prepare Data
#=========================
# Load the dataset and repeat the same feature engineering steps as before

df = pd.read_csv("../projects/01_titanic/train.csv")

df.drop(columns=['Name', 'Ticket', 'PassengerId', 'Cabin', 'Fare', 'Embarked'], inplace=True)
df.dropna(axis=0, how='any', inplace=True)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1 
df['Sex'] = df['Sex'].map({'male':0, 'female':1})

print(df.head(5))

#=========================
#3. Split Features and Target
#=========================

X = df.drop(columns=['Survived'])
y = df['Survived']

#=========================
#4. Train/Test Split
#=========================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#=========================
#5. Naive Bayes Model
#=========================
# Train a Gaussian Naive Bayes model

model = GaussianNB()
model.fit(X_train, y_train)

#=========================
#6. Prediction and Evaluation
#=========================
# Predict and calculate accuracy

prediction = model.predict(X_test)
print("Predict:", prediction)

Acc = accuracy_score(y_test, prediction)
print("Accuracy score:", Acc)

#=========================
#7. Result Note
#=========================
# Accuracy: ~80%
# Similar to Random Forest (~80%), a bit behind SVM (~85%). Naive Bayes
# assumes all features are independent, which isn't really true here
# (like Sex and Pclass probably aren't fully independent), but it
# still gave a solid result anyway.
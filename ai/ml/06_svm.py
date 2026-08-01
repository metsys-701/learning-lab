"""
Topic: SVM (Support Vector Machine)

Description:
This file contains my practice exercises for SVM, using the same
Titanic dataset, with feature scaling added before training since
SVM is distance-based.
"""

#=========================
#1. Imports
#=========================
# Import pandas, train_test_split, SVC, StandardScaler, and accuracy_score

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#=========================
#2. Load and Prepare Data
#=========================
# Load the dataset and repeat the same feature engineering steps as before

df = pd.read_csv("../projects/01_titanic/train.csv")

df.drop(columns= ['Name', 'Ticket', 'PassengerId', 'Cabin', 'Fare', 'Embarked'], inplace=True)
df.dropna(axis=0, how='any', inplace= True)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['Sex'] = df['Sex'].map({'male':0, 'female':1})

#print(df.head(5))

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
#5. Feature Scaling
#=========================
# Scale the features (same approach as KNN)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#=========================
#6. SVM Model
#=========================
# Train an SVM model using the rbf kernel

model = SVC(kernel='rbf')
model.fit(X_train_scaled, y_train)

#=========================
#7. Prediction and Evaluation
#=========================
# Predict and calculate accuracy

prediction = model.predict(X_test_scaled)
print("Predict:", prediction)

Acc = accuracy_score(y_test, prediction)
print("Accuracy score:", Acc)


#=========================
#8. Result Note
#=========================
# Accuracy: ~85%
# This is the best result so far compared to Logistic Regression (~81%),
# Decision Tree (~71%), Random Forest (~80%), and KNN (~81%). I think
# the rbf kernel might be catching some patterns in the data that the
# other models couldn't, but I'd need to learn more to really know why.
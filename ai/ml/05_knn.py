"""
Topic: KNN (K-Nearest Neighbors)

Description:
This file contains my practice exercises for KNN, using the same
Titanic dataset, with feature scaling added before training.
"""

#=========================
#1. Imports
#=========================
# Import pandas, train_test_split, KNeighborsClassifier, StandardScaler, and accuracy_score
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#=========================
#2. Load and Prepare Data
#=========================
# Load the dataset and repeat the same feature engineering steps as before

df = pd.read_csv("../projects/01_titanic/train.csv")

df.drop(columns=['Name', 'Ticket', 'PassengerId', 'Cabin', 'Fare', 'Embarked'], inplace=True )
df.dropna(axis=0, how='any', inplace=True)
df['FamilySize']= df['SibSp'] + df['Parch'] + 1
df['Sex'] = df['Sex'].map({'male':0, 'female':1})

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
# Scale the features so that no single feature dominates the distance calculation

scaler= StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#=========================
#6. KNN Model
#=========================
# Train a KNN model with 5 neighbors

model = KNeighborsClassifier(n_neighbors = 5)
model.fit(X_train_scaled, y_train)


#=========================
#7. Prediction and Evaluation
#=========================
# Predict and calculate accuracy
predictions = model.predict(X_test_scaled)
print("Predict:", predictions)

Acc = accuracy_score(y_test, predictions)
print("Acc knn: ", Acc)

#=========================
#8. Result Note
#=========================
# Logistic Regression: ~81%, Decision Tree: ~71%, Random Forest: ~80%, KNN: ~81%
# KNN did just as well as Logistic Regression here. I think scaling
# the features really helped, since KNN uses distance and Age/Pclass
# were on very different scales before. Without scaling this probably
# would have done worse.
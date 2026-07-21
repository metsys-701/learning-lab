#=========================
#1. Imports
#=========================
# Import numpy, train_test_split, DecisionTreeClassifier, and accuracy_score

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#=========================
#2. Dataset
#=========================
# study_hours -> first feature
# absences -> second feature
# passed -> target, 0 = failed, 1 = passed

study_hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 2, 6])
absences = np.array([5, 4, 6, 2, 1, 0, 3, 1, 0, 0, 8, 2])
passed = np.array([0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1])

X = np.column_stack((study_hours,absences))

#=========================
#3. Train/Test Split
#=========================
# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, passed, test_size=0.2)

#=========================
#4. Decision Tree Model
#=========================
# Train the model with a limited depth to avoid overfitting

model = DecisionTreeClassifier(max_depth = 2)
model.fit(X_train,y_train)

#=========================
#5. Prediction
#=========================
# Predict on the test data

prediction = model.predict(X_test)
print("Predict:", prediction)

#=========================
#6. Model Evaluation
#=========================
# Calculate the accuracy

Acc = accuracy_score(y_test, prediction)
print("Acc:", Acc)
#=========================
#1. Imports
#=========================
# Import numpy, train_test_split, LogisticRegression, and accuracy_score

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#=========================
#2. Dataset
#=========================
# study_hours -> independent variable (x)
# passed -> dependent variable (y), 0 = failed, 1 = passed

study_hours = np.array([1, 2, 3, 3.5, 4, 5, 5.5, 6, 7, 8, 9, 10 ]).reshape(-1,1)
passed = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1 ])



#=========================
#3. Train/Test Split
#=========================
# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(study_hours, passed, test_size=0.25)

#=========================
#4. Logistic Regression Model
#=========================
# Train the model using only the training data

model = LogisticRegression()
model.fit(X_train,y_train)

#=========================
#5. Prediction
#=========================
# Predict on the test data

predictions = model.predict(X_test)
print("predictions:", predictions)


#=========================
#6. Prediction Probabilities
#=========================
# See the probability behind each prediction

probabilities = model.predict_proba(X_test)
print("Probability:", probabilities)

#=========================
#7. Model Evaluation
#=========================
# Calculate how many predictions were correct

Acc = accuracy_score(y_test, predictions)
print("Acc:", Acc)

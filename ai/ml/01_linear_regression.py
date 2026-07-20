"""
Topic: Linear Regression (Manual + scikit-learn)

Description:
This file contains my practice exercises for simple linear regression,
implemented manually with numpy and then compared with scikit-learn.
"""

#=========================
#1. Imports
#=========================
# Import numpy for manual calculations, and scikit-learn for the built-in model
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#=========================
#2. Dataset
#=========================
# study_hours -> independent variable (x)
# exam_scores -> dependent variable (y)

study_hours = np.array([2, 4, 5, 7, 8, 9, 10, 11])
exam_scores = np.array([40, 45, 50, 55, 60, 65, 70, 75])

#=========================
#3. Manual Linear Regression - Means
#=========================
# Calculate the mean of x and y

x_mean = study_hours.mean()
y_mean = exam_scores.mean()

print(f"Study hours mean: {x_mean}")
print(f"Exam scores mean : {y_mean}")

#=========================
#4. Manual Linear Regression - Slope
#=========================
# numerator: how much x and y change together
# denominator: how much x varies on its own

numerator = np.sum((study_hours - x_mean) * (exam_scores - y_mean))
denominator = np.sum((study_hours - x_mean) ** 2)

slope = numerator / denominator
print(f"Slope: {slope}")

#=========================
#5. Manual Linear Regression - Intercept
#=========================
# Calculate the intercept using the mean point (x_mean, y_mean)

intercept = y_mean - slope * x_mean
print(f"Intercept: {intercept}")

#=========================
#6. Manual Prediction
#=========================
# Predict the exam score for a new study hour value

new_hour = 7.5
manual_prediction = slope * new_hour + intercept

print(f"Manual prediction for {new_hour} hours: {manual_prediction}")

#=========================
#7. Linear Regression with scikit-learn
#=========================
# sklearn expects x to be a 2D array (matrix), so we reshape it

X = study_hours.reshape(-1, 1)
y = exam_scores

model = LinearRegression()
model.fit(X, y)
print(f"sklearn slope: {model.coef_[0]}")
print(f"sklearn intercept: {model.intercept_}")

#=========================
#8. sklearn Prediction
#=========================
# Predict using the trained model and compare with the manual result

sklearn_prediction = model.predict([[new_hour]])
print(f"sklearn prediction for {new_hour} hours: {sklearn_prediction}")

#=========================
#9. Model Evaluation
#=========================
# Mean Squared Error: average squared difference between actual and predicted values

y_pred_all = model.predict(X)
mse = mean_squared_error(y, y_pred_all)

print(f"MSE: {mse}")